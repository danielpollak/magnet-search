"""Methods appendix figure (figs/appendix/Fig_1_3.pdf in the Overleaf project,
label fig:Electromagnet) -- the electromagnetic stimulator.

(A) Solenoid on bobbin with iron rod core. (B-C) Finished package; B carries a
2.3 cm scale bar across the solenoid's width (the size given in methods.tex).

Rebuilt from the three photos in figs/paper/appendix/ (Fig_1_3_{A,B,C}.png),
replacing the old hand-assembled PDF. The old panel D (audio-plug wiring
diagram) and the in-figure caption are dropped; the caption lives in LaTeX.

Usage:
    python pipeline/manuscript/fig_electromagnet.py
"""
import argparse
from pathlib import Path

# Detect if running in Jupyter notebook (must do this before matplotlib.use)
try:
    from IPython import get_ipython
    in_notebook = get_ipython() is not None
except ImportError:
    in_notebook = False

import matplotlib
if not in_notebook:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import format_parameters as FP

# Photo B has a white strip along its left edge (columns 0-49); cut it off.
B_LEFT_CROP = 50
# The solenoid's left/right edges in photo B, in its ORIGINAL pixel columns:
# where the red channel crosses halfway between the solenoid body and the
# green background, measured on rows 300, 400 and 500 (identical on each).
# The solenoid is 2.3 cm wide (methods.tex).
B_SOLENOID_X = (195, 600)
B_SOLENOID_CM = 2.3
B_SCALEBAR_Y = 665  # px, in the green just below the solenoid's base (~620)

GAP_IN = 0.08  # horizontal gap between photos, inches


def plot_fig_electromagnet(photo_dir, out_dir, out_name="Fig_1_3.pdf"):
    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY}
    matplotlib.rc("font", **font)

    imgs = [plt.imread(Path(photo_dir) / f"Fig_1_3_{p}.png")[..., :3] for p in "ABC"]
    imgs[1] = imgs[1][:, B_LEFT_CROP:]
    heights = {img.shape[0] for img in imgs}
    if len(heights) != 1:
        raise ValueError(f"photos must share a height, got {sorted(heights)}")
    height_px = heights.pop()
    widths_px = [img.shape[1] for img in imgs]

    # Photos at a common height, side by side: the figure's height follows
    # from the width, so no photo is letterboxed.
    fig_w = FP.FIGSIZE_FIG1_COMPOSITE[0]
    scale = (fig_w - GAP_IN * (len(imgs) - 1)) / sum(widths_px)  # inches per px
    fig_h = height_px * scale
    fig = plt.figure(figsize=(fig_w, fig_h))

    axes = []
    x_in = 0.0
    for img, w_px in zip(imgs, widths_px):
        ax = fig.add_axes((x_in / fig_w, 0, w_px * scale / fig_w, 1))
        ax.imshow(img, interpolation="none")
        ax.set_axis_off()
        axes.append(ax)
        x_in += w_px * scale + GAP_IN

    x0, x1 = (x - B_LEFT_CROP for x in B_SOLENOID_X)
    axes[1].annotate("", xy=(x0, B_SCALEBAR_Y), xytext=(x1, B_SCALEBAR_Y),
                     arrowprops=dict(arrowstyle="<->", color="white", lw=FP.LW_TRACE,
                                     shrinkA=0, shrinkB=0))
    axes[1].text((x0 + x1) / 2, B_SCALEBAR_Y + 18, f"{B_SOLENOID_CM:g} cm", color="white",
                 fontsize=matplotlib.rcParams["xtick.labelsize"], ha="center", va="top")

    for ax, label in zip(axes, "ABC"):
        ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom",
                    fontfamily=FP.FONT_FAMILY, fontsize=FP.FS_PANEL, annotation_clip=False)

    out_path = Path(out_dir) / out_name
    fig.savefig(out_path, bbox_inches="tight", dpi=FP.DPI)
    print(f"Saved {out_path}")
    if not in_notebook:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate the electromagnet appendix figure")
    parser.add_argument("--photo-dir", default=FP.APPENDIX_DIR, help="Directory holding Fig_1_3_{A,B,C}.png")
    parser.add_argument("--out-dir", default=FP.APPENDIX_DIR, help="Output directory for the PDF")
    args = parser.parse_args([] if in_notebook else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_fig_electromagnet(args.photo_dir, out_dir)


if __name__ == "__main__":
    if in_notebook:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    main()
