"""Compare suite2p's segmentation of 20220915_fish1 in three runs: production, and the two runs of
segment.py (x64 tiffs with suite2p's registration off; raw frames, or frames motion corrected).

    python docs/custom_registration/compare_segmentation.py     # magneto2 env, ~1 min
    python docs/custom_registration/compare_segmentation.py --figures-only

Production's ROIs are in suite2p's registered coordinates, which here are the raw frame moved by
suite2p's constant shift (yoff + yoff1, xoff + xoff1 = -52.6, 20.4 px). Every production ROI is
moved by it, rounded to whole pixels, for the figures and the overlap counts.

"Kept" means P(iscell) > 0.5 and npix >= 10, the analysis' ROI thresholds (the analysis also
drops ROIs outside the fish's outline and flat traces; for production that leaves the same 182
ROIs here).

The cell next to production ROI 58 (docs/custom_registration.md section 4) is followed in each
run: every ROI sharing pixels with production ROI 58, 22 or 113 is listed, and the traces of
the largest overlapping kept ROI are plotted against production ROI 58's. A trace is the ROI's
weighted mean (suite2p's F.npy) in raw photon units: production's F_full.npy as is, the
segment.py runs' F.npy divided by SCALE / 2 (suite2p halves uint16 input).

Outputs:
    results_segmentation.csv    one row per ROI per run: npix, P(iscell), kept, centre, overlap
                                with production ROIs 58 / 22 / 113
    fig_segmentation.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import measure as ms  # noqa: E402

SESSION = "2022_09_15-fish1_cytoGCaMP"
ROOT = Path(f"C:/Users/dan/suite2p_v1_custom_reg/{SESSION}")
RUNS = {"production": Path(f"C:/Users/dan/suite2p_v1_production/{SESSION}/output/suite2p/plane0"),
        "raw_x64": ROOT / "s2p_raw_x64/output/suite2p/plane0",
        "corrected_x64": ROOT / "s2p_corrected_x64/output/suite2p/plane0"}
LABELS = {"production": "production (unscaled, suite2p registration)",
          "raw_x64": "x64, no motion correction", "corrected_x64": "x64, motion corrected"}
COLORS = {"production": "#8a8a8a", "raw_x64": "#e8590c", "corrected_x64": "#2a78d6"}
SCALE = 64
OFFSET = 100.0
FOCUS = (58, 22, 113)        # production ROIs that split the active cell (section 4)
HALF = 48
NEAR = 15                    # px; search radius for the ROI that carries ROI 58's signal
TRIAL = ("2022_09_15_fish1_magneto_0.tif", 0, 1260)   # tiff, first frame, n frames


def offset_to_corrected():
    """Per run, the offset to subtract from its ROI pixels to land in the segment.py runs'
    (raw-frame) coordinates. Production's registered frame q shows raw pixel
    q + (yoff + yoff1, xoff + xoff1) (one value on every frame and block here); the check is
    that ROI 58's trace computed from raw pixels at that offset matches F_full.npy (r = 0.99 at
    the rounded offset, the best of every integer offset within +-40 px). The two segment.py runs
    share the raw coordinates (their mean images cross-correlate at (0, 0))."""
    o = np.load(RUNS["production"] / "ops.npy", allow_pickle=True).item()
    for k in ("yoff", "xoff", "yoff1", "xoff1"):
        assert np.ptp(o[k]) == 0, f"production's {k} is not constant"
    s = np.array([o["yoff"].flat[0] + o["yoff1"].flat[0], o["xoff"].flat[0] + o["xoff1"].flat[0]])
    return {"production": -np.round(s).astype(int), "raw_x64": np.zeros(2, int),
            "corrected_x64": np.zeros(2, int)}


def load():
    off = offset_to_corrected()
    rows, pix = [], {}
    prod = np.load(RUNS["production"] / "stat.npy", allow_pickle=True)
    focus = {f: set(zip(prod[f]["ypix"] - off["production"][0], prod[f]["xpix"] - off["production"][1]))
             for f in FOCUS}
    for run, p in RUNS.items():
        stat = np.load(p / "stat.npy", allow_pickle=True)
        ic = np.load(p / "iscell.npy")[:, 1]
        for k, s in enumerate(stat):
            y, x = s["ypix"] - off[run][0], s["xpix"] - off[run][1]
            pix[(run, k)] = (y, x)
            px = set(zip(y, x))
            rows.append(dict(run=run, roi=k, npix=int(s["npix"]), p_iscell=float(ic[k]),
                             kept=bool(ic[k] > 0.5 and s["npix"] >= 10),
                             med_y=float(np.median(y)), med_x=float(np.median(x)),
                             **{f"overlap_{f}": len(px & focus[f]) for f in FOCUS}))
    return pd.DataFrame(rows), pix, off


def traces(D):
    """Production ROI 58's trace and, per segment.py run, the trace of the ROI (kept or not)
    within NEAR px of it whose magneto_0 trace correlates best with ROI 58's."""
    _, start, n = TRIAL
    f58 = np.load(RUNS["production"] / "F_full.npy", mmap_mode="r")[58, start:start + n] - OFFSET
    p58 = D[(D.run == "production") & (D.roi == 58)].iloc[0]
    out = {"production": (58, np.asarray(f58), 1.0)}
    for run in ("raw_x64", "corrected_x64"):
        d = D[(D.run == run) & (np.hypot(D.med_y - p58.med_y, D.med_x - p58.med_x) < NEAR)]
        F = np.load(RUNS[run] / "F.npy", mmap_mode="r")
        r = [np.corrcoef(F[k, start:start + n], f58)[0, 1] for k in d.roi]
        k = int(d.roi.values[int(np.argmax(r))])
        out[run] = (k, np.asarray(F[k, start:start + n]) / (SCALE / 2) - OFFSET, float(np.max(r)))
    return out


def main():
    D, pix, off = load()
    D.to_csv(_HERE / "results_segmentation.csv", index=False)
    print(f"  offsets to the corrected run's coordinates: {off}")
    s = D.groupby("run", sort=False).agg(
        rois=("roi", "size"), npix_median=("npix", "median"),
        share_under_10px=("npix", lambda v: np.mean(v < 10)),
        p_iscell_over_half=("p_iscell", lambda v: int(np.sum(v > 0.5))),
        kept=("kept", "sum"))
    s["kept_npix_median"] = D[D.kept].groupby("run").npix.median()
    # production detected only inside its registered field (ops yrange/xrange, the frame minus
    # the constant shift's margins); count the kept ROIs whose centre lies inside that field too
    o = np.load(RUNS["production"] / "ops.npy", allow_pickle=True).item()
    (ya, yb), (xa, xb) = np.array(o["yrange"]) - off["production"][0], np.array(o["xrange"]) - off["production"][1]
    inside = D.med_y.between(ya, yb - 1) & D.med_x.between(xa, xb - 1)
    s["kept_in_production_field"] = D[D.kept & inside].groupby("run").size()
    print(s.round(2).to_string())
    _, start, n = TRIAL
    f58 = np.load(RUNS["production"] / "F_full.npy", mmap_mode="r")[58, start:start + n]
    for run in RUNS:
        d = D[(D.run == run) & (D[[f"overlap_{f}" for f in FOCUS]].sum(1) > 0)].copy()
        F = np.load(RUNS[run] / ("F_full.npy" if run == "production" else "F.npy"), mmap_mode="r")
        d["r_with_58"] = [np.corrcoef(F[k, start:start + n], f58)[0, 1] for k in d.roi]
        print(f"  {run}: ROIs sharing pixels with production ROIs {FOCUS} "
              f"(r_with_58: magneto_0 trace vs production ROI 58's):")
        print(d[["roi", "npix", "p_iscell", "kept", "r_with_58"] + [f"overlap_{f}" for f in FOCUS]]
              .round(2).to_string(index=False))
    tr = traces(D)
    for run in ("raw_x64", "corrected_x64"):
        print(f"  {run}: best-correlated ROI within {NEAR} px of ROI 58: {tr[run][0]} (r = {tr[run][2]:.2f})")
    figure(D, pix)


def figure(D, pix=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if pix is None:
        pix = {}
        off = offset_to_corrected()
        for run, p in RUNS.items():
            for k, s in enumerate(np.load(p / "stat.npy", allow_pickle=True)):
                pix[(run, k)] = (s["ypix"] - off[run][0], s["xpix"] - off[run][1])
    o = np.load(RUNS["corrected_x64"] / "ops.npy", allow_pickle=True).item()
    bg = o["max_proj"] if o["max_proj"].shape == o["meanImg"].shape else o["meanImg"]
    yr, xr = o.get("yrange", [0, bg.shape[0]]), o.get("xrange", [0, bg.shape[1]])
    full = np.full(o["meanImg"].shape, np.nan)
    full[yr[0]:yr[0] + bg.shape[0], xr[0]:xr[0] + bg.shape[1]] = bg
    p58 = D[(D.run == "production") & (D.roi == 58)].iloc[0]
    cy, cx = int(p58.med_y), int(p58.med_x)
    y0, x0 = cy - HALF, cx - HALF
    crop = full[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF]
    lo, hi = np.nanpercentile(crop, [1, 99.5])

    def mask(run, k):
        m = np.zeros((2 * HALF, 2 * HALF), bool)
        y, x = pix[(run, k)]
        ok = (y >= y0) & (y < y0 + 2 * HALF) & (x >= x0) & (x < x0 + 2 * HALF)
        m[y[ok] - y0, x[ok] - x0] = True
        return m

    tr = traces(D)
    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.3, wspace=0.25)
    for i, run in enumerate(RUNS):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(crop, cmap="magma", vmin=lo, vmax=hi, interpolation="nearest")
        d = D[D.run == run]
        n_kept = n_other = 0
        for _, r in d.iterrows():
            if abs(r.med_y - cy) > HALF + 20 or abs(r.med_x - cx) > HALF + 20:
                continue
            m = mask(run, r.roi)
            if not m.any():
                continue
            n_kept += r.kept
            n_other += not r.kept
            ax.contour(m, levels=[0.5], colors="#ff4d4d" if r.kept else "#ffb3b3",
                       linewidths=0.8 if r.kept else 0.4, linestyles="-" if r.kept else ":")
        ax.contour(mask("production", 58), levels=[0.5], colors="#5ad1ff", linewidths=1.2)
        if run != "production":
            ax.contour(mask(run, tr[run][0]), levels=[0.5], colors="#5ad1ff", linewidths=1.0,
                       linestyles="--")
        ax.set_title(f"{LABELS[run]}\n{n_kept} kept ROIs (solid), {n_other} others (dotted)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    for run in RUNS:
        v = np.sort(D[D.run == run].npix.values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COLORS[run], lw=1.2,
                label=f"{LABELS[run]} ({len(v)} ROIs)")
    ax.axvline(10, color="k", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("ROI size, npix (dotted: the analysis' minimum, 10)", fontsize=8)
    ax.set_ylabel("share of ROIs at or below", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax = fig.add_subplot(gs[1, 1])
    for run in RUNS:
        v = np.sort(D[(D.run == run) & D.kept].npix.values)
        ax.step(v, np.arange(1, len(v) + 1), where="post", color=COLORS[run], lw=1.2,
                label=f"{LABELS[run]}: {len(v)} kept")
    ax.set_xscale("log")
    ax.set_xlabel("ROI size, npix (kept ROIs only)", fontsize=8)
    ax.set_ylabel("number of kept ROIs at or below", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax = fig.add_subplot(gs[1, 2])
    t = np.arange(TRIAL[2]) / 60
    for k, run in enumerate(RUNS):
        roi, F, r = tr[run]
        row = D[(D.run == run) & (D.roi == roi)].iloc[0]
        ax.plot(t, F + 4 * k, color=COLORS[run], lw=0.5,
                label=f"{run} ROI {roi}: {row.npix} px, P(iscell) {row.p_iscell:.2f}"
                      + ("" if run == "production" else f", r = {r:.2f}"))
    ax.set_xlabel("time in magneto_0 (min)", fontsize=8)
    ax.set_ylabel("photons / pixel / frame (offset by 4 per run)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    for a in fig.axes:
        a.tick_params(labelsize=7)
    fig.savefig(_HERE / "fig_segmentation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        figure(pd.read_csv(_HERE / "results_segmentation.csv"))
    else:
        main()
