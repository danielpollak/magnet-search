"""Can ROIs outside the fish be removed using the outline of the fish in the mean image?

    python docs/nfc_finite_sample_bias/fov_images.py   # once: copy suite2p summary images from the NAS
    python docs/nfc_finite_sample_bias/body_mask.py    # ~2 min

suite2p puts thousands of ROIs on the dark background around the fish. This script outlines
the fish in suite2p's time-averaged image (meanImg) and flags every ROI that lies outside.

Outline, per field of view:
  1. log(meanImg), smoothed with a Gaussian of SIGMA_PX pixels (brightness of tissue, not of
     single cells);
  2. threshold at background + FRAC x (tissue - background), where background and tissue are
     the 2nd and 98th percentiles of the smoothed image. A fixed fraction of the range, not
     Otsu: in the dim recordings Otsu splits bright from dim tissue instead of tissue from
     background;
  3. keep every connected region at least MIN_REGION of the largest (the two tectal lobes
     can be separate regions in a dim recording), fill holes, and widen it by MARGIN_PX pixels so
     cells on the edge of the brain are not cut.
A ROI is inside if at least half of its pixels are inside the outline. Nothing here uses
traces, P(iscell) or the stimulus.

Outputs (next to this script): results_body_mask.csv (one row per ROI per field of view,
ROIs with npix >= MIN_NPIX), fig_body_mask.pdf.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage as ndi

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import fov_images  # noqa: E402
from coverage_vs_roi_quality import MIN_NPIX, OUT_CSV, SETS, fields_of_view  # noqa: E402
from roi_masks_by_tercile import read_masks  # noqa: E402

OUT_ROIS = _HERE / "results_body_mask.csv"
OUT_PDF = _HERE / "fig_body_mask.pdf"
SIGMA_PX = 10
FRAC = 0.15
MARGIN_PX = 5
MIN_REGION = 0.1                # keep every region at least this fraction of the largest
C_IN, C_OUT, C_OUTLINE = "#2a78d6", "#eb6834", "#f2c400"


def log_mean(a):
    a = np.log1p(a - np.nanmin(a))
    return np.nan_to_num(a, nan=float(np.nanmin(a)))


def outline(mean_img):
    s = ndi.gaussian_filter(log_mean(mean_img), SIGMA_PX)
    bg, tissue = np.percentile(s, [2, 98])
    m = s > bg + FRAC * (tissue - bg)
    lab, n = ndi.label(m)
    if n > 1:
        area = ndi.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.where(area >= MIN_REGION * area.max())[0])
    m = ndi.binary_fill_holes(m)
    return ndi.binary_dilation(m, iterations=MARGIN_PX)


def inside_share(masks, body):
    return np.array([body[yy, xx].mean() for yy, xx in masks])


def page(pdf, b, recs, d, masks, mean_img, body):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    show = log_mean(mean_img)
    lo, hi = np.percentile(show, [1, 99.5])
    kept = d["kept"].values
    inside = d["inside"].values
    for ax, title, sel in ((axes[0], "all ROIs", np.ones(len(d), bool)),
                           (axes[1], "ROIs passing the production cut", kept)):
        ax.imshow(show, cmap="gray", vmin=lo, vmax=hi)
        rgba = np.zeros((*body.shape, 4))
        for i in np.where(sel)[0]:
            yy, xx = masks[i]
            rgba[yy, xx] = matplotlib.colors.to_rgba(C_IN if inside[i] else C_OUT, 0.85)
        ax.imshow(rgba, interpolation="nearest")
        ax.contour(body, levels=[0.5], colors=C_OUTLINE, linewidths=1)
        ax.set_title(f"{title}: {(sel & inside).sum()} inside (blue), "
                     f"{(sel & ~inside).sum()} outside (orange)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    # Inside vs outside: one strip per variable, one dot per ROI.
    ax = axes[2]
    rng = np.random.default_rng(0)
    for k, (col, label) in enumerate((("p_iscell", "P(iscell)"), ("coverage", "coverage"))):
        for j, (m, c) in enumerate(((inside, C_IN), (~inside, C_OUT))):
            x = 2 * k + j + rng.uniform(-0.3, 0.3, m.sum())
            ax.scatter(x, d.loc[m, col], s=2, c=c, alpha=0.4, linewidths=0, rasterized=True)
            if m.any():
                ax.plot([2 * k + j - 0.35, 2 * k + j + 0.35], [d.loc[m, col].median()] * 2,
                        color="k", lw=1.5)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["P(iscell)\ninside", "P(iscell)\noutside", "coverage\ninside",
                        "coverage\noutside"], fontsize=8)
    ax.axhline(d["iscell_threshold"].iloc[0], xmin=0, xmax=0.5, color="k", ls="--", lw=0.8)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("one dot per ROI; black bar: median; dashed: production P(iscell) cut",
                 fontsize=9)
    ax.tick_params(labelsize=7)
    fig.suptitle(f"{b}: {recs}\nyellow: outline of the fish from the mean image; "
                 f"{len(d)} ROIs with npix >= {MIN_NPIX}", fontsize=10)
    pdf.savefig(fig, dpi=110)
    plt.close(fig)


def main():
    if not fov_images.CACHE.exists():
        sys.exit(f"{fov_images.CACHE.name} missing: run fov_images.py first")
    images = fov_images.load()
    rois = fields_of_view(pd.read_csv(OUT_CSV))
    rois["kept"] = (rois["p_iscell"] > rois["iscell_threshold"]) & (rois["npix"] > rois["npix_threshold"])
    rows = []
    with PdfPages(OUT_PDF) as pdf:
        for b in SETS:
            for seg, d in rois[rois["batch"] == b].groupby("segmentation", sort=False):
                d = d.sort_values("roi").copy()
                masks, _ = read_masks(d["experiment"].iloc[0])
                masks = [masks[i] for i in d["roi"]]
                mean_img = images[seg]["meanImg"]
                body = outline(mean_img)
                d["inside_share"] = inside_share(masks, body)
                d["inside"] = d["inside_share"] >= 0.5
                recs = d["recordings"].iloc[0].replace("engert_", "")
                print(f"  [{b}] {recs}: {(~d['inside']).sum()} of {len(d)} outside; "
                      f"{(d['kept'] & ~d['inside']).sum()} of {d['kept'].sum()} kept outside",
                      flush=True)
                page(pdf, b, recs, d, masks, mean_img, body)
                rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out.drop(columns=["n_windows"]).to_csv(OUT_ROIS, index=False)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
