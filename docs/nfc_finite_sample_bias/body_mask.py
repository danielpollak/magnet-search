"""Can ROIs outside the fish be removed using the outline of the fish in the mean image?

    python docs/nfc_finite_sample_bias/fov_images.py   # once: copy suite2p summary images from the NAS
    python docs/nfc_finite_sample_bias/body_mask.py    # ~2 min

suite2p puts thousands of ROIs on the dark background around the fish. This script outlines
the fish in suite2p's time-averaged image (meanImg) and flags every ROI that lies outside.

Outline, per field of view:
  1. log(meanImg - its minimum + LOG_OFFSET x its 98th percentile), smoothed with a
     Gaussian of `sigma` pixels (brightness of tissue, not of
     single cells);
  2. threshold at background + `frac` x (tissue - background), where background and tissue are
     the 2nd and 98th percentiles of the smoothed image. A fixed fraction of the range, not
     Otsu: in the dim recordings Otsu splits bright from dim tissue instead of tissue from
     background;
  3. keep every connected region at least `min_region` of the largest (the two tectal lobes
     can be separate regions in a dim recording), fill holes, and widen it by `margin` pixels so
     cells on the edge of the brain are not cut.
A ROI is inside if at least half of its pixels are inside the outline. Nothing here uses
traces, P(iscell) or the stimulus. The method is written up step by step in
docs/body_outline.md.

The four parameters can be set per field of view as a `body_outline:` block in the
experiment YAMLs (written by the slider GUI, body_outline_gui.ipynb, to every YAML that shares
the field of view); recordings without one use the defaults below. The block can instead hold
a hand-drawn `polygon`, which replaces the automatic outline.

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
EXPERIMENTS = _HERE.parents[1] / "experiments"
LOG_OFFSET = 0.05                # see log_mean
DEFAULTS = dict(sigma=10.0,      # smoothing, pixels
                frac=0.15,       # threshold, fraction of the way from background to tissue
                margin=5,        # widening of the outline, pixels
                min_region=0.1)  # keep every region at least this fraction of the largest
C_IN, C_OUT, C_OUTLINE = "#2a78d6", "#eb6834", "#f2c400"


def log_mean(a):
    """log(brightness above the image minimum + an offset). The offset is LOG_OFFSET x the
    image's 98th-percentile brightness above its minimum, so the transform behaves the same
    whether the image spans 50.0-50.6 (floor-clipped) or 5000-6300 (2022 Q1)."""
    a = np.nan_to_num(a, nan=float(np.nanmin(a)))
    a = a - a.min()
    return np.log(a + LOG_OFFSET * np.percentile(a, 98))


def smooth(mean_img, sigma):
    return ndi.gaussian_filter(log_mean(mean_img), sigma)


def outline_steps(mean_img, sigma=DEFAULTS["sigma"], frac=DEFAULTS["frac"],
                  margin=DEFAULTS["margin"], min_region=DEFAULTS["min_region"], smoothed=None):
    """Every intermediate of the outline, for the report and the GUI. `smoothed` lets the GUI
    reuse smooth(mean_img, sigma) when only the other sliders move."""
    st = dict(log=log_mean(mean_img))
    st["smooth"] = smooth(mean_img, sigma) if smoothed is None else smoothed
    st["background"], st["tissue"] = np.percentile(st["smooth"], [2, 98])
    st["threshold"] = st["background"] + frac * (st["tissue"] - st["background"])
    st["above"] = st["smooth"] > st["threshold"]
    lab, n = ndi.label(st["above"])
    area = ndi.sum(st["above"], lab, range(1, n + 1)) if n else np.zeros(0)
    keep = 1 + np.where(area >= min_region * area.max())[0] if n else np.zeros(0, int)
    st["labels"], st["kept_labels"] = lab, keep
    st["regions"] = np.isin(lab, keep)
    st["filled"] = ndi.binary_fill_holes(st["regions"])
    st["body"] = ndi.distance_transform_edt(~st["filled"]) <= margin
    return st


def outline(mean_img, **params):
    return outline_steps(mean_img, **params)["body"]


def polygon_mask(polygon, shape):
    """Pixels whose centres lie inside a hand-drawn polygon of [x (column), y (row)] vertices."""
    from matplotlib.path import Path as Polygon
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    inside = Polygon(polygon).contains_points(np.c_[xx.ravel(), yy.ravel()])
    return inside.reshape(shape)


def body(mean_img, params):
    """The outline for one field of view: a hand-drawn polygon if the parameters have one (it
    replaces the automatic outline, with no margin added), else the automatic outline."""
    if params.get("polygon"):
        return polygon_mask(params["polygon"], mean_img.shape)
    return outline(mean_img, **{k: params[k] for k in DEFAULTS})


_YAMLS = {}


def fov_yamls(seg):
    """Every experiment YAML whose recording uses this field of view (segmentation key, as in
    coverage_vs_roi_quality.segmentation_key): engert YAMLs sharing the session_path,
    including recordings not analysed here (e.g. nostim); medaka's single YAML."""
    if not _YAMLS:
        import yaml
        for path in sorted(EXPERIMENTS.glob("*.yml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if raw.get("paradigm") == "engert":
                _YAMLS.setdefault(raw["session_path"], []).append(path)
            elif raw.get("paradigm") == "medaka":
                _YAMLS.setdefault(raw["name"], []).append(path)
    return _YAMLS.get(seg, [])


def fov_params(seg):
    """(parameters, saved?) for one field of view: the YAMLs' body_outline block, else the
    defaults."""
    import yaml
    for path in fov_yamls(seg):
        block = yaml.safe_load(path.read_text(encoding="utf-8")).get("body_outline")
        if block:
            return {**DEFAULTS, **block}, True
    return dict(DEFAULTS), False


BLOCK_COMMENT = "# Outline of the fish in the mean image (docs/body_outline.md); set with body_outline_gui.ipynb"


def save_fov_params(seg, params):
    """Write `body_outline:` into every YAML of this field of view, replacing any existing
    block. Text-level edit, so the rest of each file (comments, order) is untouched. A
    hand-drawn polygon, if any, goes on one line as [[x, y], ...] in pixels."""
    block = [BLOCK_COMMENT, "body_outline:"] + [
        f"  {k}: {round(float(params[k]), 4) if k != 'margin' else int(params[k])}"
        for k in DEFAULTS]
    if params.get("polygon"):
        block.append("  polygon: [" + ", ".join(f"[{x:.1f}, {y:.1f}]"
                                               for x, y in params["polygon"]) + "]")
    for path in fov_yamls(seg):
        raw = path.read_bytes().decode("utf-8")
        eol = "\r\n" if "\r\n" in raw else "\n"           # keep the file's own line endings
        lines = raw.splitlines()
        out, skip = [], False
        for line in lines:
            if line == BLOCK_COMMENT:
                continue
            if line.startswith("body_outline:"):
                skip = True
                continue
            if skip and (line.startswith(" ") or not line.strip()):
                continue
            skip = False
            out.append(line)
        while out and not out[-1].strip():
            out.pop()
        path.write_bytes((eol.join(out + [""] + block) + eol).encode("utf-8"))
    return fov_yamls(seg)


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
                params = fov_params(seg)[0]
                body_ = body(mean_img, params)
                d["inside_share"] = inside_share(masks, body_)
                d["inside"] = d["inside_share"] >= 0.5
                recs = d["recordings"].iloc[0].replace("engert_", "")
                print(f"  [{b}] {recs}: {(~d['inside']).sum()} of {len(d)} outside; "
                      f"{(d['kept'] & ~d['inside']).sum()} of {d['kept'].sum()} kept outside",
                      flush=True)
                if params.get("polygon"):
                    recs += " (hand-drawn outline)"
                page(pdf, b, recs, d, masks, mean_img, body_)
                rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out.drop(columns=["n_windows"]).to_csv(OUT_ROIS, index=False)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
