"""Where are the ROIs in each tercile of P(iscell), coverage and npix?

    python docs/nfc_finite_sample_bias/roi_masks_by_tercile.py     # ~3 min

Companion to coverage_vs_roi_quality.py (reads its CSV; run that first). One page per
suite2p segmentation (field of view), grouped by set of recordings. Each page is 3 x 3:
rows are P(iscell), coverage and npix; columns are the low, middle and high tercile of that
variable. Each panel draws the pixel masks of the ROIs in that tercile, filled with their
value of the row variable, on top of every other ROI in faint gray.

- All suite2p ROIs are used, not just those passing the production P(iscell)/npix cut.
- Tercile edges are computed per set of recordings (pooling its fields of view), so the
  pages of one set share the same edges. A value equal to an edge goes to the lower
  tercile; when many ROIs tie (coverage 0 or 1) a tercile can hold far more or fewer than a
  third, or be empty. Panel titles give the range and count.
- Zebrafish repeat trials share one segmentation; a ROI's coverage on its page is the mean
  over those trials.

Output (next to this script): fig_roi_masks_by_tercile.pdf.
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
from slow_variation import schema  # noqa: E402
from coverage_vs_roi_quality import OUT_CSV, SETS  # noqa: E402
from pipeline import nwb_io  # noqa: E402

OUT_PDF = _HERE / "fig_roi_masks_by_tercile.pdf"
ROWS = [("p_iscell", "P(iscell)"), ("coverage", "coverage"), ("npix", "npix")]
TERCILES = ["low", "middle", "high"]
BACKGROUND = np.array([0.0, 0.0, 0.0])
OTHER = np.array([0.28, 0.28, 0.28])


def segmentation_key(name):
    """Recordings sharing a suite2p segmentation share a key (engert: session_path)."""
    if name.startswith("medaka"):
        return name
    cfg = schema.load_experiment(str(sv._REPO / "experiments" / f"{name}.yml"))
    return cfg.session_path


def read_masks(name):
    cfg = schema.load_experiment(str(sv._REPO / "experiments" / f"{name}.yml"))
    io_r, nwbfile = nwb_io.read_nwbfile(cfg.nwb_path())
    _, roi_df = nwb_io.read_roi_data(nwbfile)
    Ly, Lx = nwb_io.get_imaging_dims(nwbfile)
    io_r.close()
    masks = [(np.asarray(pm)["y"].astype(int), np.asarray(pm)["x"].astype(int))
             for pm in roi_df["pixel_mask"]]
    return masks, (Ly, Lx)


def tercile(v, edges):
    return np.where(v <= edges[0], 0, np.where(v <= edges[1], 1, 2))


def render(masks, dims, sel, values, norm, cmap):
    img = np.tile(BACKGROUND, (*dims, 1))
    for i, (yy, xx) in enumerate(masks):
        if not sel[i]:
            img[yy, xx] = OTHER
    colours = cmap(norm(values))[:, :3]
    for i in np.where(sel)[0]:
        yy, xx = masks[i]
        img[yy, xx] = colours[i]
    return img


def page(pdf, b, title, d, masks, dims, edges, norms):
    fig, axes = plt.subplots(3, 3, figsize=(12, 13), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for r, (col, label) in enumerate(ROWS):
        v = d[col].values
        t = tercile(v, edges[col])
        for c in range(3):
            ax = axes[r, c]
            sel = t == c
            ax.imshow(render(masks, dims, sel, v, norms[col], cmap), interpolation="nearest")
            lo = v.min() if c == 0 else edges[col][c - 1]
            hi = edges[col][c] if c < 2 else v.max()
            rng = f"{'[' if c == 0 else '('}{lo:.3g}, {hi:.3g}]"
            ax.set_title(f"{label}, {TERCILES[c]} tercile {rng}: {sel.sum()} ROIs"
                         + ("" if sel.any() else " (empty: ties)"), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        sm = plt.cm.ScalarMappable(norm=norms[col], cmap=cmap)
        cb = fig.colorbar(sm, ax=axes[r, :], shrink=0.8, pad=0.01)
        cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"{b}: {title}\n{len(d)} suite2p ROIs (all, unfiltered). Coloured: ROIs in "
                 f"the tercile; gray: every other ROI", fontsize=10)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def main():
    if not OUT_CSV.exists():
        sys.exit(f"{OUT_CSV.name} missing: run coverage_vs_roi_quality.py first")
    df = pd.read_csv(OUT_CSV)
    df["segmentation"] = [segmentation_key(n) for n in df["experiment"]]
    # One row per ROI per segmentation; coverage averaged over that set's repeat trials.
    rois = (df.groupby(["batch", "segmentation", "roi"], sort=False)
              .agg(p_iscell=("p_iscell", "first"), npix=("npix", "first"),
                   coverage=("coverage", "mean"), experiment=("experiment", "first"),
                   recordings=("experiment", lambda s: ", ".join(sorted(set(s)))))
              .reset_index())
    with PdfPages(OUT_PDF) as pdf:
        for b in SETS:
            sb = rois[rois["batch"] == b]
            if sb.empty:
                continue
            edges = {col: np.quantile(sb[col], [1 / 3, 2 / 3]) for col, _ in ROWS}
            norms = dict(p_iscell=Normalize(0, 1), coverage=Normalize(0, 1),
                         npix=LogNorm(max(sb["npix"].min(), 1), sb["npix"].max()))
            for seg, d in sb.groupby("segmentation", sort=False):
                d = d.sort_values("roi")
                masks, dims = read_masks(d["experiment"].iloc[0])
                assert len(masks) == len(d), (seg, len(masks), len(d))
                recs = d["recordings"].iloc[0].replace("engert_", "")
                print(f"  [{b}] {recs}", flush=True)
                page(pdf, b, recs, d, masks, dims, edges, norms)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
