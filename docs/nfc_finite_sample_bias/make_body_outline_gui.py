"""Writes body_outline_gui.ipynb, the slider GUI for the fish outline (see body_mask.py).

    python docs/nfc_finite_sample_bias/make_body_outline_gui.py

The notebook is generated from this script so its code lives in a reviewable .py file. Open
the notebook in VS Code (magneto2 kernel) and run all cells.
"""
from pathlib import Path

import nbformat as nbf

_HERE = Path(__file__).resolve().parent

INTRO = """# Fish outline: slider GUI

Adjust how the outline of the fish is drawn, one field of view at a time, and save the
settings. The method is explained step by step in `docs/body_outline.md`.

**Run all cells**, then:
1. pick a field of view;
2. move the sliders until the yellow outline follows the fish. Orange ROIs are outside it and
   will be removed; blue ones are kept;
3. switch to the next field of view. If you moved a slider, the settings are saved
   automatically, as a `body_outline:` block in **every experiment YAML that uses this field
   of view** (e.g. all repeat trials of a fish; for 2022 Q1 also the visual, visualmagnet and
   nostim recordings). To accept the settings without moving a slider (e.g. the defaults are
   fine), press **Save**.

The progress line lists the fields of view that still have no saved settings. `body_mask.py`
reads the YAML blocks; recordings without one use the defaults.

The left panel is the mean image with the outline and the ROIs (npix >= 10). The right panel is
the smoothed image the threshold is applied to; the cyan line is the raw threshold before
small regions are dropped, holes filled and the margin added.

Needs `results_coverage_vs_roi_quality.csv` (from `coverage_vs_roi_quality.py`) and
`fov_images.npz` (from `fov_images.py`)."""

SETUP = r'''%matplotlib inline
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as w
from IPython.display import display, clear_output

HERE = Path.cwd()
if not (HERE / "body_mask.py").exists():           # opened from the repo root instead
    HERE = HERE / "docs" / "nfc_finite_sample_bias"
sys.path.insert(0, str(HERE))
import body_mask as bm
import fov_images
from coverage_vs_roi_quality import OUT_CSV, fields_of_view
from roi_masks_by_tercile import read_masks

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
reset = w.Button(description="Reset to defaults")
status = w.HTML()
progress = w.HTML()
log = w.HTML()
out = w.Output()
SLIDERS = dict(frac=frac, sigma=sigma, margin=margin, min_region=min_region)

loaded = {}                                         # field of view -> settings shown on arrival

def current():
    return {k: s.value for k, s in SLIDERS.items()}

def save_fov(label):
    seg = fovs[label]
    paths = bm.save_fov_params(seg, current())
    log.value = (f"last saved: {label.split('  ')[0]} to {len(paths)} YAML(s): "
                    + ", ".join(p.stem for p in paths))
    show_progress()

def show_progress():
    todo = [label.split("  ")[0] for label, seg in fovs.items() if not bm.fov_params(seg)[1]]
    progress.value = (f"<b>{len(fovs) - len(todo)} of {len(fovs)}</b> fields of view saved"
                      + (f"; still to do: {', '.join(todo)}" if todo else ": all done"))

def on_fov(change):
    old = change["old"]
    if old in loaded and current() != loaded[old]:  # moved a slider: save on leaving
        save_fov(old)
    load_params()

def load_params(*_):
    p, saved = bm.fov_params(fovs[fov.value])
    for k, s in SLIDERS.items():
        s.unobserve(draw, "value")
        s.value = p[k]
        s.observe(draw, "value")
    loaded[fov.value] = current()
    draw()
    n = len(bm.fov_yamls(fovs[fov.value]))
    status.value = (f"this field of view: settings from its {n} YAML(s)" if saved
                    else f"this field of view: <i>defaults</i>, not saved yet ({n} YAML(s))")

def draw(*_):
    seg = fovs[fov.value]
    d, label, flat, owner = fov_data(seg)
    st = bm.outline_steps(images[seg]["meanImg"], sigma=sigma.value, frac=frac.value,
                          margin=margin.value, min_region=min_region.value,
                          smoothed=smoothed(seg, sigma.value))
    body = st["body"]
    share = np.bincount(owner, body.ravel()[flat], len(d)) / np.bincount(owner, minlength=len(d))
    inside = share >= 0.5
    show = d["kept"].values if only_kept.value else np.ones(len(d), bool)
    colours = np.zeros((len(d) + 1, 4))
    colours[:-1][show & inside] = (0.16, 0.47, 0.84, 0.85)
    colours[:-1][show & ~inside] = (0.92, 0.41, 0.20, 0.9)
    overlay = colours[label]                       # label -1 -> last row, transparent
    with out:
        clear_output(wait=True)
        fig, ax = plt.subplots(1, 2, figsize=(15, 7.3), constrained_layout=True)
        lo, hi = np.percentile(st["log"], [1, 99.5])
        ax[0].imshow(st["log"], cmap="gray", vmin=lo, vmax=hi)
        ax[0].imshow(overlay, interpolation="nearest")
        ax[0].contour(body, levels=[0.5], colors="#f2c400", linewidths=1.2)
        ax[0].set_title(f"{(show & inside).sum()} ROIs inside (blue), {(show & ~inside).sum()} "
                        f"outside (orange)", fontsize=10)
        ax[1].imshow(st["smooth"], cmap="gray")
        ax[1].contour(st["above"], levels=[0.5], colors="c", linewidths=0.8)
        ax[1].contour(body, levels=[0.5], colors="#f2c400", linewidths=1.2)
        ax[1].set_title("smoothed log image; cyan: above threshold; yellow: final outline",
                        fontsize=10)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        plt.show()

def on_save(_):
    save_fov(fov.value)
    loaded[fov.value] = current()
    status.value = f"this field of view: <b>saved</b> to its {len(bm.fov_yamls(fovs[fov.value]))} YAML(s)"

def on_reset(_):
    for k, s in SLIDERS.items():
        s.value = bm.DEFAULTS[k]

fov.observe(on_fov, "value")
only_kept.observe(draw, "value")
save.on_click(on_save)
reset.on_click(on_reset)
display(w.VBox([fov, w.HBox([w.VBox([frac, sigma]), w.VBox([margin, min_region])]),
                w.HBox([only_kept, save, reset]), status, progress, log, out]))
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
