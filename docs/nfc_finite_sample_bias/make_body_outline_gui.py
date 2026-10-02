"""Writes body_outline_gui.ipynb, the slider and polygon GUI for the fish outline (see body_mask.py).

    python docs/nfc_finite_sample_bias/make_body_outline_gui.py

The notebook is generated from this script so its code lives in a reviewable .py file. Open
the notebook in VS Code (magneto2 kernel) and run all cells.
"""
from pathlib import Path

import nbformat as nbf

_HERE = Path(__file__).resolve().parent

INTRO = """# Fish outline: slider and polygon GUI

Adjust the outline of the fish one field of view at a time, and save the settings. The method
is explained step by step in `docs/body_outline.md`.

**Run all cells.** Every field of view opens with its saved settings, or with the default
settings if none are saved, and shows the outline they produce. Then:
1. pick a field of view;
2. if the yellow outline doesn't follow the fish, either move the sliders, or press **Draw
   polygon** and click the corners of the fish in the left panel (click the first corner again
   to close it; Esc starts over). A polygon replaces the automatic outline exactly, with no
   margin added; **Clear polygon** goes back to the sliders. Orange ROIs are outside the
   outline and will be removed; blue ones are kept;
3. switch to the next field of view. If you changed anything, the settings are saved
   automatically, as a `body_outline:` block in **every experiment YAML that uses this field
   of view** (e.g. all repeat trials of a fish; for 2022 Q1 also the visual, visualmagnet and
   nostim recordings). **Save** saves without switching.

Fields of view you leave untouched are not written to; `body_mask.py` uses the defaults for
them. **Reset to defaults** puts the default sliders back and clears the polygon (on a field
of view with saved settings, switch away or press Save to store that).

The left panel is the mean image with the outline and the ROIs (npix >= 10). The right panel is
the smoothed image the threshold is applied to; the cyan line is the raw threshold before
small regions are dropped, holes filled and the margin added.

Needs `ipympl` in the kernel's environment, `results_coverage_vs_roi_quality.csv` (from
`coverage_vs_roi_quality.py`) and `fov_images.npz` (from `fov_images.py`)."""

SETUP = r'''import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import ipywidgets as w
from IPython.display import display

HERE = Path.cwd()
if not (HERE / "body_mask.py").exists():           # opened from the repo root instead
    HERE = HERE / "docs" / "nfc_finite_sample_bias"
sys.path.insert(0, str(HERE))
import body_mask as bm
import fov_images
from coverage_vs_roi_quality import OUT_CSV, fields_of_view
from roi_masks_by_tercile import read_masks
%matplotlib widget
# ^ after the imports: these scripts switch matplotlib to the non-interactive Agg backend

images = fov_images.load()
rois = fields_of_view(pd.read_csv(OUT_CSV)).drop_duplicates(["segmentation", "roi"])
rois["kept"] = (rois["p_iscell"] > rois["iscell_threshold"]) & (rois["npix"] > rois["npix_threshold"])
fovs = {}                                           # dropdown label -> segmentation key
for seg, d in rois.groupby("segmentation", sort=False):
    sets = sorted(set(d["batch"]))
    fovs[f"{seg.replace(chr(92), '/').split('/')[-1]}  ({'; '.join(sets)})"] = seg
print(f"{len(fovs)} fields of view")'''

STATE = r'''_masks, _smooth = {}, {}

def fov_data(seg):
    """ROI table, a pixel -> ROI label image, and the flat pixel indices of every ROI."""
    if seg not in _masks:
        d = rois[rois["segmentation"] == seg].sort_values("roi").reset_index(drop=True)
        masks, dims = read_masks(d["experiment"].iloc[0])
        masks = [masks[i] for i in d["roi"]]
        label = np.full(dims, -1)
        for i, (yy, xx) in enumerate(masks):
            label[yy, xx] = i
        flat = np.concatenate([yy * dims[1] + xx for yy, xx in masks])
        owner = np.repeat(np.arange(len(masks)), [len(yy) for yy, _ in masks])
        _masks[seg] = (d, label, flat, owner)
    return _masks[seg]

def smoothed(seg, sigma):
    if (seg, sigma) not in _smooth:
        _smooth[(seg, sigma)] = bm.smooth(images[seg]["meanImg"], sigma)
    return _smooth[(seg, sigma)]'''

GUI = r'''fov = w.Dropdown(options=list(fovs), description="field of view", layout=w.Layout(width="95%"),
                 style=dict(description_width="initial"))
style = dict(description_width="190px")
lay = w.Layout(width="560px")
frac = w.FloatSlider(min=0, max=0.6, step=0.01, description="threshold (frac of range)",
                     continuous_update=False, style=style, layout=lay)
sigma = w.FloatSlider(min=2, max=30, step=1, description="smoothing (px)",
                      continuous_update=False, style=style, layout=lay)
margin = w.IntSlider(min=0, max=30, step=1, description="margin (px)",
                     continuous_update=False, style=style, layout=lay)
min_region = w.FloatSlider(min=0, max=1, step=0.02, description="min region (frac of largest)",
                           continuous_update=False, style=style, layout=lay)
only_kept = w.Checkbox(description="only ROIs passing the production P(iscell)/npix cut")
save = w.Button(description="Save", button_style="success", icon="save")
reset = w.Button(description="Reset to defaults", icon="undo")
draw_poly = w.ToggleButton(description="Draw polygon", icon="pencil")
clear_poly = w.Button(description="Clear polygon", icon="times")
status = w.HTML()
progress = w.HTML()
log = w.HTML()
SLIDERS = dict(frac=frac, sigma=sigma, margin=margin, min_region=min_region)

with plt.ioff():
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.4), constrained_layout=True)
fig.canvas.header_visible = False

polygon = None                                      # hand-drawn outline of this field of view
selector = None
loaded = {}                                         # field of view -> settings shown on arrival

def current():
    p = {k: s.value for k, s in SLIDERS.items()}
    if polygon:
        p["polygon"] = polygon
    return p

def save_fov(label):
    seg = fovs[label]
    paths = bm.save_fov_params(seg, current())
    log.value = (f"last saved: {label.split('  ')[0]} to {len(paths)} YAML(s): "
                    + ", ".join(p.stem for p in paths))
    show_progress()

def show_progress():
    done = [label.split("  ")[0] for label, seg in fovs.items() if bm.fov_params(seg)[1]]
    progress.value = (f"<b>{len(done)} of {len(fovs)}</b> fields of view have saved settings"
                      + (f" ({', '.join(done)})" if done else "")
                      + "; the others use the defaults")

def on_fov(change):
    old = change["old"]
    if old in loaded and current() != loaded[old]:  # changed something: save on leaving
        save_fov(old)
    load_params()

def load_params(*_):
    global polygon
    p, saved = bm.fov_params(fovs[fov.value])
    polygon = [list(map(float, v)) for v in p["polygon"]] if p.get("polygon") else None
    draw_poly.value = False
    for k, s in SLIDERS.items():
        s.unobserve(draw, "value")
        s.value = p[k]
        s.observe(draw, "value")
    loaded[fov.value] = current()
    draw()
    n = len(bm.fov_yamls(fovs[fov.value]))
    status.value = (f"this field of view: settings from its {n} YAML(s)" if saved
                    else f"this field of view: <i>defaults</i>, not saved ({n} YAML(s))")

def draw(*_):
    global selector
    seg = fovs[fov.value]
    d, label, flat, owner = fov_data(seg)
    img = images[seg]["meanImg"]
    st = bm.outline_steps(img, sigma=sigma.value, frac=frac.value, margin=margin.value,
                          min_region=min_region.value, smoothed=smoothed(seg, sigma.value))
    body = bm.polygon_mask(polygon, img.shape) if polygon else st["body"]
    for s in SLIDERS.values():                      # the polygon replaces the sliders' outline
        s.disabled = bool(polygon)
    share = np.bincount(owner, body.ravel()[flat], len(d)) / np.bincount(owner, minlength=len(d))
    inside = share >= 0.5
    show = d["kept"].values if only_kept.value else np.ones(len(d), bool)
    colours = np.zeros((len(d) + 1, 4))
    colours[:-1][show & inside] = (0.16, 0.47, 0.84, 0.85)
    colours[:-1][show & ~inside] = (0.92, 0.41, 0.20, 0.9)
    overlay = colours[label]                       # label -1 -> last row, transparent
    if selector is not None:
        selector.disconnect_events()
        selector = None
    for a in ax:
        a.clear()
    lo, hi = np.percentile(st["log"], [1, 99.5])
    ax[0].imshow(st["log"], cmap="gray", vmin=lo, vmax=hi)
    ax[0].imshow(overlay, interpolation="nearest")
    ax[0].contour(body, levels=[0.5], colors="#f2c400", linewidths=1.2)
    kind = "hand-drawn polygon" if polygon else "automatic outline"
    ax[0].set_title(f"{kind}: {(show & inside).sum()} ROIs inside (blue), "
                    f"{(show & ~inside).sum()} outside (orange)", fontsize=10)
    ax[1].imshow(st["smooth"], cmap="gray")
    ax[1].contour(st["above"], levels=[0.5], colors="c", linewidths=0.8)
    ax[1].contour(body, levels=[0.5], colors="#f2c400", linewidths=1.2)
    ax[1].set_title("smoothed log image; cyan: above threshold; yellow: final outline",
                    fontsize=10)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    if draw_poly.value:
        ax[0].set_title("click the corners of the fish in this panel; click the first corner "
                        "again to close (Esc: start over)", fontsize=10, color="#d62728")
        selector = PolygonSelector(ax[0], on_polygon, useblit=False,
                                   props=dict(color="#d62728", linewidth=1.5))
    fig.canvas.draw_idle()

def on_polygon(verts):
    global polygon
    polygon = [[float(x), float(y)] for x, y in verts]
    draw_poly.value = False                         # redraws with the new outline

def on_clear(_):
    global polygon
    polygon = None
    draw_poly.value = False
    draw()

def on_save(_):
    save_fov(fov.value)
    loaded[fov.value] = current()
    status.value = f"this field of view: <b>saved</b> to its {len(bm.fov_yamls(fovs[fov.value]))} YAML(s)"

def on_reset(_):
    global polygon
    polygon = None
    draw_poly.value = False
    for k, s in SLIDERS.items():
        s.unobserve(draw, "value")
        s.value = bm.DEFAULTS[k]
        s.observe(draw, "value")
    draw()

fov.observe(on_fov, "value")
only_kept.observe(draw, "value")
draw_poly.observe(draw, "value")
save.on_click(on_save)
reset.on_click(on_reset)
clear_poly.on_click(on_clear)
display(w.VBox([fov, w.HBox([w.VBox([frac, sigma]), w.VBox([margin, min_region])]),
                w.HBox([only_kept, draw_poly, clear_poly]), w.HBox([save, reset]),
                status, progress, log, fig.canvas]))
load_params()
show_progress()'''


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(SETUP),
                nbf.v4.new_code_cell(STATE), nbf.v4.new_code_cell(GUI)]
    nb.metadata["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
    out = _HERE / "body_outline_gui.ipynb"
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
