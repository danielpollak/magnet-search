"""Figures for docs/body_outline.md: the fish-outline method, one step at a time.

    python docs/body_outline/body_outline_figures.py     # ~1 min

Uses the same functions as docs/nfc_finite_sample_bias/body_mask.py (outline_steps), on two
example fields of view: a bright one (2022 Q1, 2022_02_21) and a dim, floor-clipped one
(0.3 Hz zebrafish, 2022_09_15-fish1). Needs fov_images.npz and
results_coverage_vs_roi_quality.csv in docs/nfc_finite_sample_bias/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "nfc_finite_sample_bias"))
import body_mask as bm  # noqa: E402
import fov_images  # noqa: E402
from coverage_vs_roi_quality import OUT_CSV, fields_of_view  # noqa: E402
from roi_masks_by_tercile import read_masks  # noqa: E402

EXAMPLES = [("2022_02_21", "bright: 2022 Q1 zebrafish, 2022_02_21"),
            ("2022_09_15-fish1", "dim: 0.3 Hz zebrafish, 2022_09_15 fish1")]
C_IN, C_OUT, C_LINE, C_RAW = "#2a78d6", "#eb6834", "#f2c400", "#22c3d6"


def load_examples():
    images = fov_images.load()
    rois = fields_of_view(pd.read_csv(OUT_CSV)).drop_duplicates(["segmentation", "roi"])
    out = []
    for tag, title in EXAMPLES:
        seg = next(s for s in images if s.replace("\\", "/").rstrip("/").split("/")[-1].startswith(tag))
        d = rois[rois["segmentation"] == seg].sort_values("roi")
        masks, _ = read_masks(d["experiment"].iloc[0])
        out.append(dict(title=title, img=images[seg]["meanImg"],
                        masks=[masks[i] for i in d["roi"]],
                        st=bm.outline_steps(images[seg]["meanImg"])))
    return out


def gray(ax, a, pct=(1, 99.5)):
    lo, hi = np.nanpercentile(a, pct)
    ax.imshow(a, cmap="gray", vmin=lo, vmax=hi)
    ax.set_xticks([])
    ax.set_yticks([])


def roi_overlay(ax, masks, shape, inside=None):
    rgba = np.zeros((*shape, 4))
    for i, (yy, xx) in enumerate(masks):
        c = C_IN if inside is None or inside[i] else C_OUT
        rgba[yy, xx] = matplotlib.colors.to_rgba(c, 0.85)
    ax.imshow(rgba, interpolation="nearest")


def grid(n_cols, ex, width=4.2):
    fig, axes = plt.subplots(len(ex), n_cols, figsize=(width * n_cols, width * len(ex) + 0.4),
                             constrained_layout=True, squeeze=False)
    for r, e in enumerate(ex):
        axes[r, 0].set_ylabel(e["title"], fontsize=9)
    return fig, axes


def fig_problem(ex, out):
    fig, axes = grid(2, ex)
    for r, e in enumerate(ex):
        gray(axes[r, 0], e["img"])
        axes[r, 0].set_title("suite2p mean image", fontsize=9)
        gray(axes[r, 1], e["img"])
        roi_overlay(axes[r, 1], e["masks"], e["img"].shape)
        axes[r, 1].set_title(f"+ all {len(e['masks'])} ROIs with npix >= 10", fontsize=9)
        axes[r, 0].set_ylabel(e["title"], fontsize=9)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_log(ex, out):
    fig, axes = grid(2, ex)
    for r, e in enumerate(ex):
        gray(axes[r, 0], e["img"])
        axes[r, 0].set_title("step 0: mean image (linear brightness)", fontsize=9)
        gray(axes[r, 1], e["st"]["log"])
        axes[r, 1].set_title("step 1: log brightness", fontsize=9)
        axes[r, 0].set_ylabel(e["title"], fontsize=9)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_smooth(ex, out):
    sigmas = [3, 10, 25]
    fig, axes = grid(len(sigmas), ex)
    for r, e in enumerate(ex):
        for c, s in enumerate(sigmas):
            gray(axes[r, c], bm.smooth(e["img"], s))
            axes[r, c].set_title(f"step 2: smoothed, {s} px" + (" (default)" if s == 10 else ""),
                                 fontsize=9)
        axes[r, 0].set_ylabel(e["title"], fontsize=9)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def otsu(x, nbins=256):
    h, edges = np.histogram(x.ravel(), nbins)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def fig_threshold(ex, out):
    fig, axes = plt.subplots(len(ex), 3, figsize=(13.5, 4.3 * len(ex)), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    for r, e in enumerate(ex):
        st = e["st"]
        s = st["smooth"]
        t_otsu = otsu(s)
        ax = axes[r, 0]
        ax.hist(s.ravel(), 200, color="0.5")
        ax.set_yscale("log")
        for v, lab, c, ls in ((st["background"], "background (2nd percentile)", "k", ":"),
                              (st["tissue"], "tissue (98th percentile)", "k", "--"),
                              (st["threshold"], "threshold = background + 0.15 x range", C_LINE, "-"),
                              (t_otsu, "Otsu threshold (not used)", C_OUT, "-")):
            ax.axvline(v, color=c, ls=ls, lw=1.5, label=lab)
        ax.set_xlabel("smoothed log brightness", fontsize=8)
        ax.set_ylabel(f"{e['title']}\npixels (log scale)", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        gray(axes[r, 1], s)
        axes[r, 1].contour(st["above"], levels=[0.5], colors=C_LINE, linewidths=1.2)
        axes[r, 1].set_title("step 3: above threshold (yellow)", fontsize=9)
        gray(axes[r, 2], s)
        axes[r, 2].contour(s > t_otsu, levels=[0.5], colors=C_OUT, linewidths=1.2)
        axes[r, 2].set_title("for comparison: above the Otsu threshold", fontsize=9)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_regions(ex, out):
    fig, axes = grid(2, ex)
    rng = np.random.default_rng(1)
    for r, e in enumerate(ex):
        st = e["st"]
        lab = st["labels"]
        n = lab.max()
        area = ndi.sum(st["above"], lab, range(1, n + 1))
        gray(axes[r, 0], st["smooth"])
        cols = rng.uniform(0.3, 1, (n + 1, 3))
        rgba = np.zeros((*lab.shape, 4))
        rgba[lab > 0, :3] = cols[lab[lab > 0]]
        rgba[lab > 0, 3] = 0.7
        axes[r, 0].imshow(rgba, interpolation="nearest")
        axes[r, 0].set_title(f"step 4a: {n} separate regions above threshold (one colour each)",
                             fontsize=9)
        gray(axes[r, 1], st["smooth"])
        dropped = (lab > 0) & ~st["regions"]
        ov = np.zeros((*lab.shape, 4))
        ov[st["regions"]] = matplotlib.colors.to_rgba(C_IN, 0.6)
        ov[dropped] = matplotlib.colors.to_rgba(C_OUT, 0.9)
        axes[r, 1].imshow(ov, interpolation="nearest")
        big = np.sort(area)[::-1]
        axes[r, 1].set_title(f"step 4b: kept {len(st['kept_labels'])} (blue), dropped "
                             f"{n - len(st['kept_labels'])} (orange: < 10% of the largest)\n"
                             f"region areas: {', '.join(str(int(a)) for a in big[:3])} px",
                             fontsize=8)
        axes[r, 0].set_ylabel(e["title"], fontsize=9)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_fill_margin(ex, out):
    fig, axes = grid(2, ex)
    for r, e in enumerate(ex):
        st = e["st"]
        gray(axes[r, 0], st["smooth"])
        ov = np.zeros((*st["filled"].shape, 4))
        ov[st["regions"]] = matplotlib.colors.to_rgba(C_IN, 0.5)
        ov[st["filled"] & ~st["regions"]] = matplotlib.colors.to_rgba(C_OUT, 0.95)
        axes[r, 0].imshow(ov, interpolation="nearest")
        axes[r, 0].set_title(f"step 5: holes filled (orange, "
                             f"{int((st['filled'] & ~st['regions']).sum())} px)", fontsize=9)
        gray(axes[r, 1], e["st"]["log"])
        axes[r, 1].contour(st["filled"], levels=[0.5], colors=C_RAW, linewidths=1)
        axes[r, 1].contour(st["body"], levels=[0.5], colors=C_LINE, linewidths=1.2)
        axes[r, 1].set_title("step 6: widened by 5 px (cyan: before, yellow: final outline)",
                             fontsize=9)
        axes[r, 0].set_ylabel(e["title"], fontsize=9)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_classify(ex, out):
    fig, axes = plt.subplots(len(ex), 2, figsize=(10, 4.4 * len(ex)), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1.1]))
    for r, e in enumerate(ex):
        body = e["st"]["body"]
        share = bm.inside_share(e["masks"], body)
        inside = share >= 0.5
        ax = axes[r, 0]
        ax.hist(share, np.linspace(0, 1, 21), color="0.5")
        ax.axvline(0.5, color="k", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("share of the ROI's pixels inside the outline", fontsize=8)
        ax.set_ylabel(f"{e['title']}\nROIs (log scale)", fontsize=8)
        ax.tick_params(labelsize=7)
        gray(axes[r, 1], e["st"]["log"])
        roi_overlay(axes[r, 1], e["masks"], body.shape, inside)
        axes[r, 1].contour(body, levels=[0.5], colors=C_LINE, linewidths=1.2)
        axes[r, 1].set_title(f"step 7: {inside.sum()} inside (blue), {(~inside).sum()} outside "
                             f"(orange)", fontsize=9)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_frac(ex, out):
    fracs = [0.05, 0.15, 0.30]
    e = ex[1]
    fig, axes = plt.subplots(1, len(fracs), figsize=(4.4 * len(fracs), 4.8), constrained_layout=True)
    for ax, f in zip(axes, fracs):
        body = bm.outline(e["img"], frac=f)
        inside = bm.inside_share(e["masks"], body) >= 0.5
        gray(ax, e["st"]["log"])
        roi_overlay(ax, e["masks"], body.shape, inside)
        ax.contour(body, levels=[0.5], colors=C_LINE, linewidths=1.2)
        ax.set_title(f"threshold {f:.2f}" + (" (default)" if f == 0.15 else "")
                     + f": {(~inside).sum()} of {len(inside)} outside", fontsize=9)
    fig.suptitle(e["title"], fontsize=10)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ex = load_examples()
    fig_problem(ex, _HERE / "fig_bo_problem.png")
    fig_log(ex, _HERE / "fig_bo_log.png")
    fig_smooth(ex, _HERE / "fig_bo_smooth.png")
    fig_threshold(ex, _HERE / "fig_bo_threshold.png")
    fig_regions(ex, _HERE / "fig_bo_regions.png")
    fig_fill_margin(ex, _HERE / "fig_bo_fill_margin.png")
    fig_classify(ex, _HERE / "fig_bo_classify.png")
    fig_frac(ex, _HERE / "fig_bo_frac.png")
    for e in ex:
        st = e["st"]
        print(e["title"], dict(background=round(st["background"], 3), tissue=round(st["tissue"], 3),
                               threshold=round(st["threshold"], 3), otsu=round(otsu(st["smooth"]), 3),
                               regions=int(st["labels"].max()), kept=len(st["kept_labels"])))


if __name__ == "__main__":
    main()
