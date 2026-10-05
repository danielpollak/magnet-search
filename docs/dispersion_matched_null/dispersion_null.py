"""Prototype: does a null matched to each ROI's own ordinate spread (CV^2) restore the imaging null?

    python docs/dispersion_matched_null/dispersion_null.py                  # ~5 min
    python docs/dispersion_matched_null/dispersion_null.py --figures-only   # replot from the CSVs

Write-up: docs/dispersion_matched_null.md. Zebrafish only: the 0.3 Hz and 0.1 Hz sets of
recordings (floor-clipped, miscalibrated) and the 2022 Q1 0.4 Hz set (unclipped, calibrated)
as a control.

NFC^2 / 2 = R, the power at the analysis bin over the mean power of its 2M noise bins.
  nominal (production)  every bin's power is exponential (Gamma, shape 1): R ~ F(2, 4M);
  dispersion-matched    every bin's power is Gamma with shape k = 1 / CV^2, CV^2 being this
                        ROI's own (variance / mean^2 of its ordinates over its whole spectrum
                        away from the stimulus, each divided by the mean of its 51 neighbours):
                        R ~ F(2k, 4Mk). k = 1 gives production back.

The test frequencies, population, simulated traces and power test are the guard-band
prototype's (docs/guard_band_null/guard_band.py), whose machinery this script reuses with the
noise bins right next to the analysis bin (G = 0, as in production) and its nulls swapped for
the two above.

Outputs (next to this script): results_dm_*.csv/.npz (gitignored), fig_dm_*.png.
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
sys.path.insert(0, str(_HERE.parent / "guard_band_null"))
sys.path.insert(0, str(_HERE.parent / "cv2"))
import guard_band as gb  # noqa: E402
import cv2_report as cr  # noqa: E402
from guard_band import SETS, CONFIGS, SIM_DEPTHS, P_GRID, C_INK, C_MUTED, _band, _style, ecdf_dev  # noqa: E402

C_SETS = {"zebrafish 0.4 Hz (2022 Q1)": "#8a8a8a", "zebrafish 0.3 Hz": "#2a78d6",
          "zebrafish 0.1 Hz": "#eb6834"}
NULLS = [("nominal", "#8a8a8a", "-", "production null, F(2, 4M)"),
         ("dispersion", "#2a78d6", "-", "dispersion-matched null, F(2k, 4Mk)")]
M_EXAMPLE = 32                       # the 0.1 Hz recordings' M, for Figure 2
CV2_EXAMPLES = [(1.0, "#8a8a8a"), (0.8, "#86b6ef"), (0.5, "#2a78d6"), (0.3, "#0d366b")]


def nulls(x, M, me, k):
    return (("nominal", gb.p_from_nfc(x, M)),
            ("dispersion", stats.f.sf(x ** 2 / 2, 2 * k, 4 * M * k)))


def compute():
    gb.GS = [0]
    gb.nulls = nulls
    real, curves_real, _, meff, stim = gb.run_real()
    sim, power, curves_sim = gb.run_sim()
    curves = pd.concat([curves_real, curves_sim], ignore_index=True)
    out = dict(real=real, meff=meff, stim=stim, sim=sim, power=power, curves=curves)
    for n, d in out.items():
        d.to_csv(_HERE / f"results_dm_{n}.csv", index=False)
    return out


def compute_by_cv2():
    """Per ROI, the ECDF of its test-frequency p-values under both nulls, with its CV^2, for
    Figure 4 (ROIs binned by CV^2; cv2_report.fig_binned)."""
    rows, Ep, Ed = [], [], []
    for name, batch in cr.recordings():
        cfg, F, _ = gb.sv.load(name, outline=True)
        s = cr.spectrum(F, cfg)
        ok = s["ok"]
        X, k = s["X"][ok], s["k"][ok][:, None]
        Ep.append(cr.ecdf_rows(gb.p_from_nfc(X, s["M"])))
        Ed.append(cr.ecdf_rows(stats.f.sf(X ** 2 / 2, 2 * k, 4 * s["M"] * k)))
        rows.append(pd.DataFrame(dict(experiment=name, batch=batch, cv2=s["cv2"][ok])))
    d = pd.concat(rows, ignore_index=True)
    Ep, Ed = np.concatenate(Ep), np.concatenate(Ed)
    d.to_csv(_HERE / "results_dm_by_cv2.csv", index=False)
    np.savez_compressed(_HERE / "results_dm_by_cv2.npz", prod=Ep, disp=Ed)
    return d, Ep, Ed


def summary_by_cv2(d, Ep, Ed):
    b = cr.cv2_bin(d["cv2"])
    out = []
    for (s, lab), g in d.groupby([d["batch"], b], observed=True):
        sel = g.index.values
        out.append(dict(set=s, cv2=lab, rois=len(sel),
                        prod=np.interp(0.5, P_GRID, Ep[sel].mean(0) - P_GRID),
                        disp=np.interp(0.5, P_GRID, Ed[sel].mean(0) - P_GRID)))
    print(pd.DataFrame(out).round(3).to_string(index=False))


# ------------------------------------------------------------------ figures
def fig_diagnose(meff, out):
    """Figure 1: the spread of a ROI's ordinates (CV^2) and its miscalibration."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for b, c in C_SETS.items():
        x = np.sort(meff.loc[meff["batch"] == b, "cv2"].values)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=2, label=b)
    ax.axvline(1, color=C_MUTED, lw=0.6, ls=":")
    ax.set_xlim(0, 2)
    ax.set_xlabel("CV$^2$ of the ROI's ordinates (1 = Gaussian noise)", fontsize=8)
    ax.set_ylabel("ECDF over ROI traces", fontsize=8)
    ax.set_title("A. Most floor-clipped ROIs have CV$^2$ < 1", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    ax = axes[1]
    for b, c in C_SETS.items():
        d = meff[meff["batch"] == b].copy()
        d["bin"] = pd.qcut(d["cv2"], 10, labels=False, duplicates="drop")
        g = d.groupby("bin").agg(cv2=("cv2", "median"), dev=("dev_roi", "mean"))
        ax.plot(g["cv2"], g["dev"], color=c, lw=2, marker="o", ms=4)
    ax.axhline(0, color=C_INK, lw=0.6)
    ax.axvline(1, color=C_MUTED, lw=0.6, ls=":")
    ax.set_xlabel("CV$^2$ of the ROI's ordinates", fontsize=8)
    ax.set_ylabel("ROI's dev@0.5 over test frequencies, production null", fontsize=8)
    ax.set_title("B. The lower the CV$^2$, the more miscalibrated", fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_gamma(stim, out):
    """Figure 2: what the dispersion-matched null is, and how far it moves p."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    x = np.linspace(0, 4, 400)
    for cv2, c in CV2_EXAMPLES:
        k = 1 / cv2
        lab = f"CV$^2$ = {cv2:g} (k = {k:.2g})" + ("  production" if cv2 == 1 else "")
        ax.plot(x, stats.gamma.pdf(x, k, scale=1 / k), color=c, lw=2, label=lab)
    ax.set_ylim(0, 2.2)
    ax.set_xlabel("power at a bin / its mean", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.set_title("A. Assumed distribution of a bin's power: Gamma(k)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    ax = axes[1]
    cv = np.linspace(0.2, 1.5, 200)
    for p, ls in ((0.05, "-"), (0.5, "--")):
        r = stats.f.isf(p, 2 / cv, 4 * M_EXAMPLE / cv)
        ax.plot(cv, np.sqrt(2 * r), color="#2a78d6", lw=2, ls=ls, label=f"NFC needed for p = {p:g}")
        ax.axhline(np.sqrt(2 * stats.f.isf(p, 2, 4 * M_EXAMPLE)), color=C_MUTED, lw=0.8, ls=ls)
    ax.axvline(1, color=C_MUTED, lw=0.6, ls=":")
    ax.set_xlabel("ROI's CV$^2$", fontsize=8)
    ax.set_ylabel("NFC", fontsize=8)
    ax.set_title(f"B. Thresholds under the matched null (M = {M_EXAMPLE}; gray: production)",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    ax = axes[2]
    for b, c in C_SETS.items():
        d = stim[stim["batch"] == b]
        dp = d[d["null"] == "dispersion"]["p"].values - d[d["null"] == "nominal"]["p"].values
        y = np.sort(dp)
        ax.plot(y, np.arange(1, len(y) + 1) / len(y), color=c, lw=2, label=b)
    ax.axvline(0, color=C_INK, lw=0.6)
    ax.set_xlim(-0.4, 0.4)
    ax.set_xlabel("p (matched null) - p (production), at the stimulus frequency", fontsize=8)
    ax.set_ylabel("ECDF over ROI traces", fontsize=8)
    ax.set_title("C. How far each ROI's p moves", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_curves(curves, out):
    """Figure 3: the whole p-value distribution where nothing was presented, real and simulated."""
    panels = [("real", b, None, f"real: {b}") for b in SETS] + \
             [("simulated", b, d, f"simulated {b.split()[1]} Hz config, log-rate SD {d:g}"
               + (" (stationary)" if d == 0 else ""))
              for b in CONFIGS for d in SIM_DEPTHS]
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 9.6), sharey=True)
    for ax, (src, b, depth, title) in zip(axes.flat, panels):
        d = curves[(curves["source"] == src) & (curves["batch"] == b)]
        if depth is not None:
            d = d[d["depth"] == depth]
        for null, c, ls, lab in NULLS:
            row = d[d["null"] == null].iloc[0]
            y = row[[f"c{i}" for i in range(len(P_GRID))]].values.astype(float)
            ax.plot(P_GRID, y, color=c, ls=ls, lw=1.8,
                    label=f"{lab}: dev@0.5 {np.interp(0.5, P_GRID, y):+.3f}")
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("p", fontsize=8)
        ax.legend(fontsize=6, frameon=False, loc="lower center")
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("ECDF(p) - p, pooled over test freqs", fontsize=8)
    fig.suptitle("p-value distribution where nothing was presented (flat at 0 = calibrated)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_power(power, out):
    """Figure 5: detection of a known stimulus-locked modulation in simulated traces."""
    fig, axes = plt.subplots(2, len(CONFIGS), figsize=(5.6 * len(CONFIGS), 7.4), squeeze=False)
    for j, b in enumerate(CONFIGS):
        for i, depth in enumerate([2.0, 3.0]):
            ax = axes[i, j]
            d = power[(power["batch"] == b) & (power["depth"] == depth)]
            for null, c, ls, lab in NULLS:
                e = d[d["null"] == null].sort_values("mod")
                ax.plot(e["mod"], e["detected"], color=c, ls=ls, lw=2, marker="o", ms=4,
                        label=lab)
            ax.axhline(0.05, color=C_INK, lw=0.6, ls=":")
            ax.set_xlabel("stimulus-locked modulation depth m: rate x (1 + m sin 2 pi f t)",
                          fontsize=8)
            ax.set_ylabel("share of traces with p < 0.05", fontsize=8)
            ax.set_title(f"simulated {b.split()[1]} Hz config, floor-clipped, log-rate SD "
                         f"{depth:g}", fontsize=9)
            _style(ax)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Power. At m = 0 the share should be 0.05 (dotted)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_stimulus(stim, out):
    """Figure 6: read-out at the stimulus frequency."""
    fig, axes = plt.subplots(1, len(SETS), figsize=(4.4 * len(SETS), 3.6), sharey=True)
    for ax, b in zip(axes, SETS):
        d = stim[stim["batch"] == b]
        for null, c, ls, lab in NULLS:
            p = d[d["null"] == null]["p"].values
            ax.plot(P_GRID, ecdf_dev(p), color=c, ls=ls, lw=1.8,
                    label=f"{lab}: dev@0.5 {np.mean(p <= 0.5) - 0.5:+.3f}")
        _band(ax, len(p))
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_title(f"{b} (n = {len(p)} ROI traces)", fontsize=9)
        ax.set_xlabel("p", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False, loc="lower center")
        _style(ax)
    axes[0].set_ylabel("ECDF(p) - p at the stimulus frequency", fontsize=8)
    fig.suptitle("Read-out at the stimulus frequency. Gray: 95% binomial band", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    names = ["real", "meff", "stim", "sim", "power", "curves"]
    if "--figures-only" in sys.argv:
        r = {n: pd.read_csv(_HERE / f"results_dm_{n}.csv") for n in names}
        d = pd.read_csv(_HERE / "results_dm_by_cv2.csv")
        z = np.load(_HERE / "results_dm_by_cv2.npz")
        Ep, Ed = z["prod"], z["disp"]
    else:
        r = compute()
        d, Ep, Ed = compute_by_cv2()
    summary_by_cv2(d, Ep, Ed)
    print(gb._summary(r["real"]).round(4).to_string(index=False))
    print(gb._summary(r["sim"].assign(batch=r["sim"]["batch"] + " d"
                                      + r["sim"]["depth"].astype(str))).round(4).to_string(index=False))
    fig_diagnose(r["meff"], _HERE / "fig_dm_diagnose.png")
    fig_gamma(r["stim"], _HERE / "fig_dm_gamma.png")
    fig_curves(r["curves"], _HERE / "fig_dm_curves.png")
    cr.fig_binned(d, [(Ep, "production null"), (Ed, "dispersion-matched null")],
                  _HERE / "fig_dm_by_cv2.png")
    fig_power(r["power"], _HERE / "fig_dm_power.png")
    fig_stimulus(r["stim"], _HERE / "fig_dm_stimulus.png")
    print("  wrote fig_dm_*.png")


if __name__ == "__main__":
    main()
