"""CV^2: what it is, how it is distributed, how it predicts miscalibration, and how each proposed
fix does across CV^2.

    python docs/cv2/cv2_marginals.py           # first: per-unit CV^2 of the noise bins, ephys + imaging
    python docs/cv2/cv2_report.py              # ~5 min
    python docs/cv2/cv2_report.py --figures-only

Write-up: docs/cv2.md.

Two versions of CV^2 (variance / mean^2 of a unit's periodogram powers; 1 for Gaussian noise):
  whole-spectrum  imaging only. Every bin from 0.05 Hz up, away from the stimulus window, each
                  power divided by the mean of its 51 neighbours. Hundreds of bins per ROI, so
                  little sampling noise. guard_band.ordinate_shape.
  noise-bin       every unit with an NWB file, ephys and imaging: the 2M noise-bin powers at
                  the stimulus frequency, as persisted in per_unit_fourier_results. Few bins, so
                  noisy; cv2_marginals.py pairs each unit with an exact-null draw of the same M.
For Gaussian noise the noise-bin CV^2 is independent of the unit's own p-value (a scale-free
statistic of exponential samples is independent of their sum, and the analysis bin is
independent of the noise bins), so binning units by it does not by itself bend their p-values.

Imaging (zebrafish 0.4 Hz / 0.3 Hz / 0.1 Hz): every ROI's production p-value at every test
frequency (guard_band.test_bins: every 3rd bin from 0.05 Hz, away from the stimulus), under
production's null F(2, 4M) and the dispersion-matched null F(2k, 4Mk), k = 1 / whole-spectrum
CV^2. Each ROI's ECDF over its test frequencies is stored, so any grouping of ROIs is the mean
of its ROIs' ECDFs (each ROI weighted equally). Population: production before the coverage
threshold (P(iscell) > 0.5, npix >= 10, inside the outline, flatline removal), coverage
computed alongside so the coverage threshold can be evaluated as one of the fixes.

Outputs (next to this script): results_cv2_rois.csv, results_cv2_ecdf.npz (gitignored),
fig_cv2_*.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE.parent / "guard_band_null"))
import guard_band as gb  # noqa: E402
from guard_band import sv, P_GRID, C_INK, C_MUTED, _band, _style, ecdf_dev  # noqa: E402
from pipeline.roi_coverage import activity  # noqa: E402

SETS = gb.SETS
SHORT = {"zebrafish 0.4 Hz (2022 Q1)": "0.4 Hz (2022 Q1)", "zebrafish 0.3 Hz": "0.3 Hz",
         "zebrafish 0.1 Hz": "0.1 Hz"}
CV2_BINS = [0, 0.5, 0.7, 0.85, 1.15, np.inf]
CV2_LABELS = ["< 0.5", "0.5-0.7", "0.7-0.85", "0.85-1.15", "> 1.15"]
CV2_COLOURS = ["#6b2a0d", "#eb6834", "#f2a36b", "#8a8a8a", "#2a78d6"]
CV2_MIN = 0.85
COVERAGE_MIN = 0.1
OUT_ROIS = _HERE / "results_cv2_rois.csv"
OUT_ECDF = _HERE / "results_cv2_ecdf.npz"
MARGINALS = _HERE / "results_cv2_marginals.csv"


# ------------------------------------------------------------------ compute
def ecdf_rows(P):
    """Per row of P (ROI x test frequency), the ECDF of its p-values at P_GRID."""
    P = np.sort(P, axis=1)
    return np.stack([np.searchsorted(row, P_GRID, side="right") / P.shape[1] for row in P])


def compute():
    recs, _ = sv.pool()
    rows, E_prod, E_disp = [], [], []
    for name, batch, _ in recs:
        if batch not in SETS:
            continue
        cfg, F, roi = sv.load(name, outline=True)
        T, N = cfg.sample_period, F.shape[1]
        f0, M = sv.window_bins(N, T, cfg.analysis.f, cfg.analysis.Q_frac)
        res = 1.0 / (N * T)
        Y = np.fft.rfft(F - F.mean(axis=1, keepdims=True), axis=1)
        lo = max(int(np.ceil(gb.F_MIN / res)) - gb.MED_WIN, 1)
        excl = np.arange(f0 - gb.G_MAX - M, f0 + gb.G_MAX + M + 1)
        k, cv2 = gb.ordinate_shape(Y, lo, Y.shape[1] - 1, excl)
        ks = gb.test_bins(Y.shape[1], f0, M, res)
        X = np.stack([gb.nfc(Y, b, M, 0) for b in ks], 1)
        ok = np.isfinite(X).all(1)
        Pp = gb.p_from_nfc(X, M)
        Pd = stats.f.sf(X ** 2 / 2, 2 * k[:, None], 4 * M * k[:, None])
        x0 = gb.nfc(Y, f0, M, 0)
        cov = activity(F, T)["coverage"]
        rows.append(pd.DataFrame(dict(
            experiment=name, batch=batch, roi=roi, M=M, cv2=cv2, coverage=cov,
            p_stim=gb.p_from_nfc(x0, M), p_stim_disp=stats.f.sf(x0 ** 2 / 2, 2 * k, 4 * M * k),
            dev_prod=np.mean(Pp <= 0.5, 1) - 0.5, dev_disp=np.mean(Pd <= 0.5, 1) - 0.5))[ok])
        E_prod.append(ecdf_rows(Pp[ok]))
        E_disp.append(ecdf_rows(Pd[ok]))
        print(f"  {name}: {ok.sum()} ROIs, M={M}, {len(ks)} test frequencies", flush=True)
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(OUT_ROIS, index=False)
    np.savez_compressed(OUT_ECDF, prod=np.concatenate(E_prod), disp=np.concatenate(E_disp))
    return d, np.concatenate(E_prod), np.concatenate(E_disp)


def cv2_bin(c):
    return pd.cut(c, CV2_BINS, labels=CV2_LABELS, right=False)


# ------------------------------------------------------------------ figures
def fig_binned(d, Ep, Ed, out):
    """Figure 2: ECDF deviation where nothing was presented, ROIs binned by CV^2, under the
    production null (top) and the dispersion-matched null (bottom)."""
    fig, axes = plt.subplots(2, len(SETS), figsize=(4.6 * len(SETS), 7.6), sharey=True)
    b = cv2_bin(d["cv2"])
    for i, (E, null) in enumerate([(Ep, "production null"), (Ed, "dispersion-matched null")]):
        for ax, s in zip(axes[i], SETS):
            for lab, c in zip(CV2_LABELS, CV2_COLOURS):
                sel = ((d["batch"] == s) & (b == lab)).values
                if sel.sum() < 20:
                    continue
                y = E[sel].mean(0) - P_GRID
                ax.plot(P_GRID, y, color=c, lw=2, label=f"CV$^2$ {lab}: {sel.sum()} ROIs, "
                        f"dev@0.5 {np.interp(0.5, P_GRID, y):+.3f}")
            ax.axhline(0, color=C_INK, lw=0.6)
            ax.set_title(f"zebrafish {SHORT[s]}, {null}", fontsize=9)
            ax.set_xlabel("p", fontsize=8)
            ax.legend(fontsize=6.5, frameon=False, loc="lower center")
            _style(ax)
        axes[i, 0].set_ylabel("ECDF(p) - p over test frequencies", fontsize=8)
    fig.suptitle("Where nothing was presented, ROIs binned by whole-spectrum CV$^2$ "
                 "(flat at 0 = calibrated)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


MARG_GROUPS = [("zebra finch", "ephys"), ("Pigeon", "ephys"), ("Quail", "ephys"),
               ("zebrafish 0.4 Hz (2022 Q1)", "imaging"), ("zebrafish 0.3 Hz", "imaging"),
               ("zebrafish 0.1 Hz", "imaging"), ("medaka 0.1 Hz", "imaging")]
MARG_BINS = [0, 0.7, 0.85, 1.0, 1.15, np.inf]
MARG_LABELS = ["< 0.7", "0.7-0.85", "0.85-1.0", "1.0-1.15", "> 1.15"]
MARG_COLOURS = ["#b8431a", "#f2a36b", "#b0b0b0", "#6fa3e3", "#1c5cab"]


def fig_stimulus_binned(m, out):
    """Figure 3: every dataset at the stimulus frequency, units binned by noise-bin CV^2."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.6), sharey=True)
    b = pd.cut(m["cv2"], MARG_BINS, labels=MARG_LABELS, right=False)
    for ax, (g, mod) in zip(axes.flat, MARG_GROUPS):
        for lab, c in zip(MARG_LABELS, MARG_COLOURS):
            p = m.loc[(m["group"] == g) & (b == lab), "p_value"].dropna().values
            if len(p) < 30:
                continue
            ax.plot(P_GRID, ecdf_dev(p), color=c, lw=1.8,
                    label=f"CV$^2$ {lab}: n = {len(p)}, {np.mean(p <= 0.5) - 0.5:+.3f}")
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_title(f"{g} ({mod})", fontsize=9)
        ax.set_xlabel("p at the stimulus frequency (production)", fontsize=8)
        ax.legend(fontsize=6, frameon=False, loc="lower center", title="legend: dev@0.5",
                  title_fontsize=6)
        _style(ax)
    axes.flat[-1].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("ECDF(p) - p", fontsize=8)
    fig.suptitle("Fig 2 magnetic pool at the stimulus frequency, units binned by noise-bin "
                 "CV$^2$ (the 2M bins the test uses)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fixes(d, Ep, Ed):
    """Each fix: which ROIs it keeps and which null it uses."""
    return [("production: all ROIs, production null", np.ones(len(d), bool), Ep, "#8a8a8a", "o"),
            (f"coverage >= {COVERAGE_MIN}, production null (current pipeline)",
             (d["coverage"] >= COVERAGE_MIN).values, Ep, "#eb6834", "s"),
            (f"CV$^2$ >= {CV2_MIN}, production null", (d["cv2"] >= CV2_MIN).values, Ep,
             "#2a78d6", "D"),
            ("all ROIs, dispersion-matched null", np.ones(len(d), bool), Ed, "#0d366b", "^")]


def fig_fixes(d, Ep, Ed, out):
    """Figure 4: each fix across CV^2 bins, and the trade-off between calibration and ROIs kept."""
    fig, axes = plt.subplots(2, len(SETS), figsize=(4.8 * len(SETS), 7.6))
    b = cv2_bin(d["cv2"])
    for j, s in enumerate(SETS):
        ins = (d["batch"] == s).values
        ax = axes[0, j]
        lab_dev = []
        for lab in CV2_LABELS:
            sel = ins & (b == lab).values
            lab_dev.append((lab, sel.sum(), (Ep[sel].mean(0) - P_GRID) if sel.sum() else None,
                            (Ed[sel].mean(0) - P_GRID) if sel.sum() else None))
        x = np.arange(len(CV2_LABELS))
        for k_, (name, c) in enumerate([("production null", "#8a8a8a"),
                                        ("dispersion-matched null", "#0d366b")]):
            y = [np.interp(0.5, P_GRID, r[2 + k_]) if r[1] >= 20 else np.nan for r in lab_dev]
            ax.plot(x, y, color=c, lw=2, marker="o", ms=5, label=name)
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_xticks(x, [f"{l}\n({r[1]})" for l, r in zip(CV2_LABELS, lab_dev)], fontsize=7)
        ax.set_xlabel("whole-spectrum CV$^2$ bin (ROIs)", fontsize=8)
        ax.set_ylabel("dev@0.5 over test frequencies", fontsize=8)
        ax.set_title(f"zebrafish {SHORT[s]}: calibration by CV$^2$", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        _style(ax)
        ax = axes[1, j]
        for name, keep, E, c, mk in fixes(d, Ep, Ed):
            sel = ins & keep
            dev = np.interp(0.5, P_GRID, E[sel].mean(0) - P_GRID)
            ax.plot(sel.sum() / ins.sum(), dev, mk, color=c, ms=9, label=name)
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("share of ROIs kept", fontsize=8)
        ax.set_ylabel("dev@0.5 over test frequencies", fontsize=8)
        ax.set_title(f"zebrafish {SHORT[s]}: calibration vs ROIs kept", fontsize=9)
        _style(ax)
    axes[1, 0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_curves_fixes(d, Ep, Ed, out):
    """Figure 5: the whole p-value distribution under each fix."""
    fig, axes = plt.subplots(1, len(SETS), figsize=(4.6 * len(SETS), 3.8), sharey=True)
    for ax, s in zip(axes, SETS):
        ins = (d["batch"] == s).values
        for name, keep, E, c, _ in fixes(d, Ep, Ed):
            sel = ins & keep
            ax.plot(P_GRID, E[sel].mean(0) - P_GRID, color=c, lw=1.8,
                    label=f"{name}: {sel.sum()} ROIs")
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_title(f"zebrafish {SHORT[s]}", fontsize=9)
        ax.set_xlabel("p", fontsize=8)
        _style(ax)
    axes[0].set_ylabel("ECDF(p) - p over test frequencies", fontsize=8)
    axes[0].legend(fontsize=6, frameon=False, loc="lower center")
    fig.suptitle("Where nothing was presented, under each fix (flat at 0 = calibrated)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def summary(d, Ep, Ed):
    out = []
    for s in SETS:
        ins = (d["batch"] == s).values
        for name, keep, E, _, _ in fixes(d, Ep, Ed):
            sel = ins & keep
            out.append(dict(set=SHORT[s], fix=name, rois=int(sel.sum()),
                            kept=sel.sum() / ins.sum(),
                            dev=np.interp(0.5, P_GRID, E[sel].mean(0) - P_GRID)))
    print(pd.DataFrame(out).round(3).to_string(index=False))
    b = cv2_bin(d["cv2"])
    t = d.assign(bin=b).groupby(["batch", "bin"], observed=True).agg(
        n=("cv2", "size"), dev_prod=("dev_prod", "mean"), dev_disp=("dev_disp", "mean"))
    print(t.round(3).to_string())


def main():
    if "--figures-only" in sys.argv:
        d = pd.read_csv(OUT_ROIS)
        z = np.load(OUT_ECDF)
        Ep, Ed = z["prod"], z["disp"]
    else:
        d, Ep, Ed = compute()
    summary(d, Ep, Ed)
    fig_binned(d, Ep, Ed, _HERE / "fig_cv2_binned.png")
    fig_stimulus_binned(pd.read_csv(MARGINALS), _HERE / "fig_cv2_stimulus_binned.png")
    fig_fixes(d, Ep, Ed, _HERE / "fig_cv2_fixes.png")
    fig_curves_fixes(d, Ep, Ed, _HERE / "fig_cv2_fixes_curves.png")
    print("  wrote fig_cv2_*.png")


if __name__ == "__main__":
    main()
