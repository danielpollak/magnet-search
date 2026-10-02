"""Outline of the fish in suite2p's mean image, for flagging ROIs outside the fish.

suite2p puts many ROIs on the dark background around the fish. The outline is drawn on the
time-averaged image (ops["meanImg"], stored in the NWB file by the processing stage):
  1. log(meanImg - its minimum + LOG_OFFSET x its 98th percentile), smoothed with a Gaussian
     of `sigma` pixels (brightness of tissue, not of single cells);
  2. threshold at background + `frac` x (tissue - background), where background and tissue are
     the 2nd and 98th percentiles of the smoothed image;
  3. keep every connected region at least `min_region` of the largest, fill holes, and widen
     it by `margin` pixels so cells on the edge of the brain are not cut.
A ROI is inside if at least half of its pixels are inside the outline. The method is written up
step by step in docs/body_outline.md.

The parameters are set per field of view in the experiment YAML's `body_outline:` block
(written by docs/nfc_finite_sample_bias/body_outline_gui.ipynb); a hand-drawn `polygon` of
[x, y] pixel vertices in that block replaces the automatic outline. Missing keys use DEFAULTS.
"""
import numpy as np
from scipy import ndimage as ndi

LOG_OFFSET = 0.05                # see log_mean
DEFAULTS = dict(sigma=10.0,      # smoothing, pixels
                frac=0.15,       # threshold, fraction of the way from background to tissue
                margin=5,        # widening of the outline, pixels
                min_region=0.1)  # keep every region at least this fraction of the largest


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


def params_for(block):
    """Full parameter set from a YAML `body_outline:` block (cfg.body_outline); {} = defaults."""
    return {**DEFAULTS, **(block or {})}


def body(mean_img, params):
    """The outline for one field of view: a hand-drawn polygon if the parameters have one (it
    replaces the automatic outline, with no margin added), else the automatic outline."""
    if params.get("polygon"):
        return polygon_mask(params["polygon"], mean_img.shape)
    return outline(mean_img, **{k: params[k] for k in DEFAULTS})


def inside_share(masks, body_):
    """Share of each ROI's pixels inside the outline; masks are (ypix, xpix) pairs."""
    return np.array([body_[yy, xx].mean() for yy, xx in masks])
