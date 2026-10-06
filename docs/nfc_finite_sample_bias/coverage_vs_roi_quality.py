"""How does activity coverage relate to the suite2p inclusion criteria (P(iscell), npix)?

    python docs/nfc_finite_sample_bias/coverage_vs_roi_quality.py     # ~2 min

Companion to activity_coverage.py. For EVERY ROI in each imaging recording of the Fig 2C
magnetic pool, plus the visual-only 2022 Q1 recordings (VISUAL) -- not just those passing
the production P(iscell)/npix cut -- computes coverage (activity_coverage.activity, on the
same first-N frames fit_Fourier analyses) and plots it against P(iscell) and npix.

One page per field of view (suite2p segmentation), grouped by set of recordings, so
different fish are never pooled. Zebrafish repeat trials share one segmentation; each ROI is
one dot, with its coverage averaged over those trials. ROIs below MIN_NPIX pixels are left
out of the plots (they are mostly tiny fragments and dominated the dot clouds); the CSV keeps
every ROI and recording. Three panels per row:

    npix vs coverage        colour = P(iscell)
    P(iscell) vs coverage   colour = npix (log)
    npix vs P(iscell)       colour = coverage

Top row: one dot per ROI trace. Coverage takes only ~20 discrete values (one step per 60 s
window) and npix is an integer, so both are jittered by less than half a step for display
only (the CSV holds the exact values). Bottom row: the same axes binned (hexagons), coloured
by the MEDIAN of the third variable over the ROIs in each bin (bins with >= 5 ROIs), which
stays readable where the dots overlap. Dashed lines mark the production thresholds (strict >,
as in _load_from_nwb).

Outputs (next to this script): results_coverage_vs_roi_quality.csv (11 MB, gitignored),
fig_coverage_vs_roi_quality.pdf.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, Normalize

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import slow_variation as sv  # noqa: E402
from slow_variation import BATCHES, schema  # noqa: E402
from activity_coverage import WIN_S, activity  # noqa: E402
from pipeline import nwb_io  # noqa: E402

OUT_CSV = _HERE / "results_coverage_vs_roi_quality.csv"
OUT_PDF = _HERE / "fig_coverage_vs_roi_quality.pdf"

# Visual-only positive control: the 2022 Q1 recordings with the sine grating alone (30 s
# moving / 30 s still, 1/60 Hz) and no magnet. Same suite2p segmentations as the 2022 Q1
# magnetic recordings (same session_path), so P(iscell)/npix/masks are identical; only the
# traces, and hence coverage, differ.
VISUAL = "zebrafish visual only, 1/60 Hz (2022 Q1)"
VISUAL_RECS = ["engert_20220221_visual", "engert_20220223_visual",
               "engert_20220301_visual_a", "engert_20220301_visual_b"]
SETS = BATCHES + [VISUAL]
MIN_NPIX = 10                   # ROIs with fewer pixels are left out of the plots


def segmentation_key(name):
    """Recordings sharing a suite2p segmentation share a key (engert: session_path)."""
    if name.startswith("medaka"):
        return name
    cfg = schema.load_experiment(str(sv._REPO / "experiments" / f"{name}.yml"))
    return cfg.session_path


def fields_of_view(df):
    """One row per ROI per field of view (ROIs with npix >= MIN_NPIX); coverage averaged over
    the repeat trials sharing that segmentation within a set of recordings."""
    df = df[df["npix"] >= MIN_NPIX].copy()
    keys = {n: segmentation_key(n) for n in df["experiment"].unique()}
    df["segmentation"] = df["experiment"].map(keys)
    return (df.groupby(["batch", "segmentation", "roi"], sort=False)
              .agg(p_iscell=("p_iscell", "first"), npix=("npix", "first"),
                   coverage=("coverage", "mean"), n_windows=("n_windows", "first"),
                   iscell_threshold=("iscell_threshold", "first"),
                   npix_threshold=("npix_threshold", "first"),
                   experiment=("experiment", "first"),
                   recordings=("experiment", lambda s: ", ".join(sorted(set(s)))))
              .reset_index())


def recordings():
    """(experiment name, set of recordings) for the Fig 2C magnetic pool plus the visual set."""
    recs, _ = sv.pool()
    return [(n, b) for n, b, _ in recs] + [(n, VISUAL) for n in VISUAL_RECS]


def load_all(name, batch):
    cfg = schema.load_experiment(str(sv._REPO / "experiments" / f"{name}.yml"))
    io_r, nwbfile = nwb_io.read_nwbfile(cfg.nwb_path())
    F, roi_df = nwb_io.read_roi_data(nwbfile, "suite2p")   # the traces this report describes
    io_r.close()
    N = min(int(120 * (F.shape[1] // 60)), F.shape[1])   # same frames as sv.load
    a = activity(F[:, :N].astype(float), cfg.sample_period)
    n_windows = N // int(round(WIN_S / cfg.sample_period))
    return pd.DataFrame(dict(experiment=name, batch=batch, roi=np.arange(len(F)),
                             n_windows=n_windows,
                             p_iscell=roi_df["p_iscell"].values, npix=roi_df["npix"].values,
                             iscell_threshold=cfg.iscell_threshold,
                             npix_threshold=cfg.npix_threshold, **a))


def jittered(d, col, rng):
    v = d[col].values.astype(float)
    if col == "coverage":
        return v + rng.uniform(-0.4, 0.4, len(v)) / d["n_windows"].values
    if col == "npix":
        return v + rng.uniform(-0.4, 0.4, len(v))
    return v


def panel(ax, d, x, y, c, norm, cmap, label_c, binned):
    rng = np.random.default_rng(0)
    order = rng.permutation(len(d))   # no colour drawn systematically on top
    xs, ys = jittered(d, x, rng)[order], jittered(d, y, rng)[order]
    if binned:
        sc = ax.hexbin(d[x].values, d[y].values, C=d[c].values, reduce_C_function=np.median,
                       mincnt=5, gridsize=(28, 20), norm=norm, cmap=cmap, linewidths=0.2,
                       xscale="log" if x == "npix" else "linear",
                       yscale="log" if y == "npix" else "linear")
    else:
        sc = ax.scatter(xs, ys, c=d[c].values[order], norm=norm, cmap=cmap, s=3,
                        linewidths=0, alpha=0.6, rasterized=True)
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(("median " if binned else "") + label_c, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    for col, line in (("iscell_threshold", "p_iscell"), ("npix_threshold", "npix")):
        for v in d[col].unique():
            if x == line:
                ax.axvline(v, color="k", ls="--", lw=0.8)
            if y == line:
                ax.axhline(v, color="k", ls="--", lw=0.8)
    if x == "npix":
        ax.set_xscale("log")
    if y == "npix":
        ax.set_yscale("log")
    names = dict(npix="npix (pixels, log)", p_iscell="P(iscell)", coverage="coverage")
    ax.set_xlabel(names[x], fontsize=8)
    ax.set_ylabel(names[y], fontsize=8)
    ax.tick_params(labelsize=7)


def figures(df):
    unit = Normalize(0, 1)
    fov = fields_of_view(df)
    with PdfPages(OUT_PDF) as pdf:
        for (b, _), d in sorted(fov.groupby(["batch", "segmentation"], sort=False),
                                key=lambda kv: SETS.index(kv[0][0])):
            npix_norm = LogNorm(d["npix"].min(), d["npix"].max())
            fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
            for r, binned in enumerate((False, True)):
                panel(axes[r, 0], d, "npix", "coverage", "p_iscell", unit, "viridis", "P(iscell)", binned)
                panel(axes[r, 1], d, "p_iscell", "coverage", "npix", npix_norm, "viridis",
                      "npix (log)", binned)
                panel(axes[r, 2], d, "npix", "p_iscell", "coverage", unit, "viridis", "coverage", binned)
            axes[0, 0].set_title("one dot per ROI trace (jittered)", fontsize=8, loc="left")
            axes[1, 0].set_title("binned: median of the colour variable, bins with >= 5 ROIs",
                                 fontsize=8, loc="left")
            thr = ", ".join(f"P(iscell) > {i:g}, npix > {n:g}" for i, n in
                            d[["iscell_threshold", "npix_threshold"]].drop_duplicates().values)
            kept = ((d["p_iscell"] > d["iscell_threshold"]) & (d["npix"] > d["npix_threshold"])).sum()
            recs = d["recordings"].iloc[0].replace("engert_", "")
            fig.suptitle(f"{b}: {recs}\n{len(d)} ROIs with npix >= {MIN_NPIX} (coverage averaged "
                         f"over repeat trials); {kept} pass production cut ({thr}; dashed)",
                         fontsize=10)
            pdf.savefig(fig, dpi=200)
            plt.close(fig)


def main():
    rows = []
    for name, batch in recordings():
        print(f"  {name}  [{batch}]", flush=True)
        rows.append(load_all(name, batch))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    figures(df)
    fov = fields_of_view(df)
    rho = fov.groupby(["batch", "segmentation"], sort=False).apply(lambda g: pd.Series(dict(
        n=len(g), cov_npix=g["coverage"].corr(g["npix"], method="spearman"),
        cov_iscell=g["coverage"].corr(g["p_iscell"], method="spearman"),
        iscell_npix=g["p_iscell"].corr(g["npix"], method="spearman"))))
    print(rho.round(2).to_string())
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
