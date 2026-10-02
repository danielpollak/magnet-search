"""Can floor-clipped ROIs with long dead time be excluded by a stimulus-blind activity rule?

    python docs/nfc_finite_sample_bias/activity_coverage.py                 # ~5 min
    python docs/nfc_finite_sample_bias/activity_coverage.py --figures-only  # replot (needs local cache)

Companion to slow_variation.py. In the floor-clipped recordings (0.3 Hz and 0.1 Hz zebrafish,
medaka) many ROIs are "on" only for part of the recording and sit at the floor for the
rest. This script measures, per ROI trace:

  floor             the trace's most common value, if it holds >= FLOOR_SHARE of frames
                    (a clipped trace); otherwise the trace has no floor. The minimum is
                    NOT used: a single outlier frame below the floor would make every
                    floor frame count as active.
  active fraction   share of frames above the floor (1 for an unclipped trace)
  coverage          share of WIN_S-second windows containing >= MIN_ACTIVE active frames

and asks whether excluding low-coverage ROIs restores p-value calibration. Calibration is
judged at ONE sham frequency per recording, where nothing was presented: the nearest bin
whose noise window does not overlap the stimulus window (stimulus bin + 2M + 1, or - 2M - 1
if that runs past Nyquist). One p-value per ROI at the sham frequency, as at the stimulus
frequency, so the two are directly comparable. The threshold is not tuned on the magnetic
result. Neither metric uses
stimulus timing or power at any frequency, so excluding on them cannot create or remove a
stimulus-locked response.

Outputs (next to this script): results_activity_rois.csv, results_activity_sweep.csv,
results_activity_curves.csv, fig_ac_*.png. activity_coverage_cache.npz (example traces
for the gallery) is gitignored.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import slow_variation as sv  # noqa: E402
from slow_variation import BATCHES, P_GRID, C_INK, C_MUTED, _band, _style, ecdf_dev  # noqa: E402

CACHE = _HERE / "activity_coverage_cache.npz"
FLOOR_SHARE = 0.2
WIN_S = 60.0
MIN_ACTIVE = 3
THRESHOLDS = np.round(np.arange(0, 1.0001, 0.05), 2)
CURVE_THRESHOLDS = [0.0, 0.25, 0.5, 0.75]
GALLERY_N = 8                   # traces per set of recordings in the gallery
CLIPPED = BATCHES[1:]           # the three floor-clipped sets of recordings
C_SET = {BATCHES[0]: "#8a8a8a", BATCHES[1]: "#2a78d6", BATCHES[2]: "#eb6834",
         BATCHES[3]: "#1baf7a"}
THR_RAMP = ["#b0b0b0", "#86b6ef", "#2a78d6", "#0d366b"]


# ------------------------------------------------------------------ compute
def activity(F, T):
    N = F.shape[1]
    floor = np.full(len(F), -np.inf)
    for i, row in enumerate(F):
        v, c = np.unique(row, return_counts=True)
        if c.max() >= FLOOR_SHARE * N:
            floor[i] = v[c.argmax()]
    active = F > floor[:, None]
    w = int(round(WIN_S / T))
    nw = N // w
    A = active[:, :nw * w].reshape(len(F), nw, w)
    return dict(clipped=np.isfinite(floor), active_frac=active.mean(1),
                coverage=(A.sum(2) >= MIN_ACTIVE).mean(1))


def analyse(name, batch, rec):
    cfg, F, roi_idx = sv.load(name)
    if name.startswith("medaka"):
        roi_idx = np.arange(len(F))
    T, f, Q_frac = cfg.sample_period, cfg.analysis.f, cfg.analysis.Q_frac
    N = F.shape[1]
    f0, M = sv.window_bins(N, T, f, Q_frac)
    a = activity(F, T)
    Y = np.fft.rfft(F - F.mean(axis=1, keepdims=True), axis=1)
    p_stim = sv.pvals(sv.nfc_from_fft(Y, f0, M), M)
    k_sham = f0 + 2 * M + 1
    if k_sham + M >= Y.shape[1] - 1:          # no room above: take the mirror bin below
        k_sham = f0 - 2 * M - 1
    p_sham = sv.pvals(sv.nfc_from_fft(Y, k_sham, M), M)
    rows = pd.DataFrame(dict(experiment=name, rec=rec, batch=batch, id=roi_idx, **a,
                             p_stim=p_stim, p_sham=p_sham, f_stim=f,
                             f_sham=k_sham / (N * T)))
    return rows, F, T


def sweep(rois):
    out = []
    for b in BATCHES:
        sel = (rois["batch"] == b).values
        d = rois[sel]
        for thr in THRESHOLDS:
            k = (d["coverage"] >= thr).values
            n = int(k.sum())
            if n == 0:
                out.append(dict(batch=b, threshold=thr, n=0))
                continue
            out.append(dict(batch=b, threshold=thr, n=n, n_total=len(d),
                            dev_sham=float(np.mean(d.loc[k, "p_sham"] <= 0.5) - 0.5),
                            dev_stim=float(np.mean(d.loc[k, "p_stim"] <= 0.5) - 0.5)))
    return pd.DataFrame(out)


def curves(rois):
    out = []
    for b in BATCHES:
        sel = (rois["batch"] == b).values
        d = rois[sel]
        for thr in CURVE_THRESHOLDS:
            k = (d["coverage"] >= thr).values
            if k.sum() == 0:
                continue
            for kind, p in (("sham", d.loc[k, "p_sham"].values),
                            ("stimulus", d.loc[k, "p_stim"].values)):
                out.append(dict(batch=b, threshold=thr, kind=kind, n_roi=int(k.sum()),
                                **{f"c{i}": v for i, v in enumerate(ecdf_dev(p))}))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ figures
def fig_distributions(rois, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for r, (col, lab) in enumerate((("active_frac", "fraction of frames above the floor"),
                                   ("coverage", f"coverage: share of {WIN_S:.0f} s windows with "
                                                f">= {MIN_ACTIVE} active frames"))):
        ax_h, ax_e = axes[r]
        bins = np.linspace(0, 1, 41)
        for b in BATCHES:
            x = rois.loc[rois["batch"] == b, col].values
            ax_h.hist(x, bins=bins, histtype="step", lw=1.5, color=C_SET[b], density=True,
                      label=f"{b} (n={len(x)})")
            xs = np.sort(x)
            ax_e.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=C_SET[b], lw=1.5)
        ax_h.set_xlabel(lab, fontsize=8)
        ax_h.set_ylabel("density (ROI traces)", fontsize=8)
        ax_e.set_xlabel(lab, fontsize=8)
        ax_e.set_ylabel("fraction of ROI traces <= x", fontsize=8)
        for ax in (ax_h, ax_e):
            ax.set_xlim(0, 1.0)
            _style(ax)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Activity per ROI trace, by set of recordings (2022 Q1 traces are unclipped: every frame "
                 "counts as active, so they pile up at 1)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_sweep(sw, out):
    fig, axes = plt.subplots(2, len(CLIPPED), figsize=(3.8 * len(CLIPPED), 6.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1]))
    for c, b in enumerate(CLIPPED):
        d = sw[(sw["batch"] == b) & (sw["n"] > 0)]
        ax = axes[0, c]
        se = 1.96 * np.sqrt(0.25 / d["n"])
        ax.fill_between(d["threshold"], -se, se, color="#d9d9d9", lw=0)
        ax.plot(d["threshold"], d["dev_sham"], color=C_SET[b], lw=2, marker="o", ms=3,
                label="sham frequency (calibration)")
        ax.plot(d["threshold"], d["dev_stim"], color=C_INK, lw=1, ls="--", marker="o", ms=2.5,
                label="stimulus frequency (read-out)")
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_title(b, fontsize=9)
        if c == 0:
            ax.set_ylabel("dev@0.5", fontsize=8)
        _style(ax)
        ax2 = axes[1, c]
        ax2.plot(d["threshold"], d["n"], color=C_SET[b], lw=2, marker="o", ms=3)
        ax2.set_ylim(0, None)
        ax2.set_xlabel("coverage threshold (keep ROI if coverage >= x)", fontsize=8)
        if c == 0:
            ax2.set_ylabel("ROI traces kept", fontsize=8)
        _style(ax2)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Calibration vs coverage threshold, one sham frequency per recording. "
                 "Gray: 95% binomial band for the ROIs kept", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_curves(cv, out):
    fig, axes = plt.subplots(2, len(BATCHES), figsize=(3.5 * len(BATCHES), 6.4), sharey=True)
    for c, b in enumerate(BATCHES):
        for r, kind in enumerate(("sham", "stimulus")):
            ax = axes[r, c]
            d = cv[(cv["batch"] == b) & (cv["kind"] == kind)]
            n_min = int(d["n_roi"].min())
            _band(ax, n_min)
            for i, (_, row) in enumerate(d.iterrows()):
                y = row[[f"c{j}" for j in range(len(P_GRID))]].values.astype(float)
                ax.plot(P_GRID, y, color=THR_RAMP[i], lw=1.5,
                        label=f"coverage >= {row['threshold']:.2f} (n={int(row['n_roi'])})")
            ax.axhline(0, color=C_INK, lw=0.6)
            ax.set_title(f"{b}\n{kind} frequency" if r == 0 else f"{kind} frequency", fontsize=8)
            if c == 0:
                ax.set_ylabel("ECDF(p) - p", fontsize=8)
            if r == 1:
                ax.set_xlabel("p", fontsize=8)
            ax.legend(fontsize=5.5, frameon=False, loc="lower center")
            _style(ax)
    fig.suptitle("ECDF deviation after excluding low-coverage ROIs. Gray: 95% binomial band for "
                 "the smallest subset shown", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_gallery(gal, out):
    fig, axes = plt.subplots(GALLERY_N, len(CLIPPED), figsize=(5 * len(CLIPPED), 1.25 * GALLERY_N),
                             squeeze=False)
    for c, b in enumerate(CLIPPED):
        g = gal[b]
        for r in range(GALLERY_N):
            ax = axes[r, c]
            y = g["F"][r]
            t = np.arange(len(y)) * g["T"]
            ax.plot(t, y, color=C_INK, lw=0.4)
            ax.set_title(f"{g['name'][r]} ROI {g['id'][r]}   coverage {g['coverage'][r]:.2f}   "
                         f"active {g['active_frac'][r]:.2f}", fontsize=6.5)
            ax.tick_params(labelsize=6)
            _style(ax)
            if r == GALLERY_N - 1:
                ax.set_xlabel("time (s)", fontsize=7)
        axes[0, c].annotate(b, (0.5, 1.45), xycoords="axes fraction", ha="center", fontsize=9)
    fig.suptitle("Raw traces at evenly spaced coverage quantiles (top = lowest coverage)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ driver
def _gallery(rois, traces):
    """GALLERY_N traces per clipped set, at evenly spaced coverage quantiles (deterministic)."""
    gal = {}
    for b in CLIPPED:
        d = rois[rois["batch"] == b].sort_values("coverage", kind="stable")
        pick = d.iloc[np.linspace(0, len(d) - 1, GALLERY_N).round().astype(int)]
        gal[b] = dict(F=[traces[(e, i)][0] for e, i in zip(pick["experiment"], pick["id"])],
                      T=traces[(pick["experiment"].iloc[0], pick["id"].iloc[0])][1],
                      name=pick["experiment"].tolist(), id=pick["id"].tolist(),
                      coverage=pick["coverage"].tolist(), active_frac=pick["active_frac"].tolist())
    return gal


def _save_gallery(gal):
    flat = {}
    for i, b in enumerate(CLIPPED):
        g = gal[b]
        L = min(len(x) for x in g["F"])
        flat[f"g{i}_F"] = np.stack([x[:L] for x in g["F"]])
        flat[f"g{i}_T"] = np.array(g["T"])
        flat[f"g{i}_batch"] = np.array(b)
        for k in ("name", "id", "coverage", "active_frac"):
            flat[f"g{i}_{k}"] = np.array(g[k])
    np.savez_compressed(CACHE, **flat)


def _load_gallery():
    z = np.load(CACHE)
    gal, i = {}, 0
    while f"g{i}_F" in z:
        b = str(z[f"g{i}_batch"])
        gal[b] = dict(F=z[f"g{i}_F"], T=float(z[f"g{i}_T"]),
                      **{k: z[f"g{i}_{k}"].tolist() for k in ("name", "id", "coverage", "active_frac")})
        i += 1
    return gal


def figures(rois, sw, cv, gal):
    fig_distributions(rois, _HERE / "fig_ac_distributions.png")
    fig_gallery(gal, _HERE / "fig_ac_gallery.png")
    fig_sweep(sw, _HERE / "fig_ac_threshold_sweep.png")
    fig_curves(cv, _HERE / "fig_ac_curves.png")
    print("  wrote fig_ac_*.png")


def main():
    if "--figures-only" in sys.argv:
        figures(pd.read_csv(_HERE / "results_activity_rois.csv"),
                pd.read_csv(_HERE / "results_activity_sweep.csv"),
                pd.read_csv(_HERE / "results_activity_curves.csv"), _load_gallery())
        return
    recs, _ = sv.pool()
    rows, traces = [], {}
    for name, batch, rec in recs:
        print(f"  {name}  [{batch}]", flush=True)
        r, F, T = analyse(name, batch, rec)
        rows.append(r)
        for i, roi in enumerate(r["id"]):
            traces[(name, roi)] = (F[i], T)
    rois = pd.concat(rows, ignore_index=True)
    sw, cv, gal = sweep(rois), curves(rois), _gallery(rois, traces)
    print(rois.groupby("batch")[["f_stim", "f_sham"]].agg(["min", "max"]).round(3).to_string())
    rois.to_csv(_HERE / "results_activity_rois.csv", index=False)
    sw.to_csv(_HERE / "results_activity_sweep.csv", index=False)
    cv.to_csv(_HERE / "results_activity_curves.csv", index=False)
    _save_gallery(gal)
    print(sw[sw["threshold"].isin([0, 0.25, 0.5, 0.75, 0.9])].round(3).to_string(index=False))
    figures(rois, sw, cv, gal)


if __name__ == "__main__":
    main()
