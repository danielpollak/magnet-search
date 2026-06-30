"""Fig 1 — composite: NPIX raw + GCaMP anatomy/traces + spectra + p-value ECDFs.

Combines exemplar NPIX unit (electrical trace) with GCaMP anatomy and calcium trace,
along with spectral analysis and empirical CDFs of p-values (converted from c-hat).

Requires NAS access to:
  \\datanas\family\data_aggregated\20230413_firstsite
  \\datanas\family\data_aggregated\Engert\2022_02_21\...

Requires pipeline outputs:
  data/20230413_firstsite_processing.pickle
  data/20230413_firstsite_analysis.pickle

Usage:
    python pipeline/manuscript/fig1.py
    python pipeline/manuscript/fig1.py --out-dir figs/paper
"""
#%%
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecdfbounds import bootstrap_ecdf_band
from ephysio import openEphysIO
from magpyneto2 import statistics, engert_helpers
from magpyneto2.statistics import normalized_Fourier_CDF
from magpyneto2.utils import get_cluster_info
#%%
# ── NAS paths ────────────────────────────────────────────────────────────────
DATA_PATH = r"\\datanas\family\data_aggregated\20230413_firstsite"

ENGERT_MAG_SUITE2P = r"\\datanas\family\data_aggregated\Engert\2022_02_21\2022-02-21_17-54-26_magnet\rawdata\suite2p\plane0"
ENGERT_MAG_TIFF    = r"\\datanas\family\data_aggregated\Engert\2022_02_21\2022-02-21_17-54-26_magnet\rawdata\z_plane0000_trial000_imaging_roi00_green_channel.tif"
ENGERT_VIS_SUITE2P = r"\\datanas\family\data_aggregated\Engert\2022_02_21\2022-02-21_17-33-14_visual\rawdata\suite2p\plane0"
ENGERT_VIS_TIFF    = r"\\datanas\family\data_aggregated\Engert\2022_02_21\2022-02-21_17-33-14_visual\rawdata\z_plane0000_trial000_imaging_roi00_green_channel.tif"

# ── Exemplar unit / cell indices ──────────────────────────────────────────────
MAG_CONTINGENCY = "2023-04-13_15-15-40_W25R_Mag5"
VIS_CONTINGENCY = "2023-04-13_15-49-48_W25R_visual_3Hz"
MAG_CLUSTER_ID  = 186
VIS_CLUSTER_ID  = 2296
MAG_CELL_IND    = 36
VIS_CELL_IND    = 45

#%%

def load_data(data_dir: str):
    data_dir = Path(data_dir)
    with open(data_dir / "20230413_firstsite_processing.pickle", "rb") as f:
        modulation_df = pickle.load(f)
    fourier_df = pd.read_pickle(data_dir / "20230413_firstsite_analysis.pickle")
    udf = get_cluster_info(DATA_PATH)
    udf = udf.loc[udf.KSLabel == "good", :]
    return modulation_df, fourier_df, udf



def plot_fig1_composite(modulation_df, fourier_df, udf, out_dir: Path):
    mag_unitrow = udf.loc[udf.cluster_id == MAG_CLUSTER_ID].squeeze()
    vis_unitrow = udf.loc[udf.cluster_id == VIS_CLUSTER_ID].squeeze()

    mag_allspks, mag_spks, mag_exemplar_fourier, mag_contingency_path = \
        statistics.Fig1_NPIX_data(modulation_df, MAG_CONTINGENCY, mag_unitrow, 5)
    vis_allspks, vis_spks, vis_exemplar_fourier, vis_contingency_path = \
        statistics.Fig1_NPIX_data(modulation_df, VIS_CONTINGENCY, vis_unitrow, 3)

    # Load GCaMP data with cell filtering
    mag_tiff, mag_F, mag_stat = engert_helpers.load_GEVI(ENGERT_MAG_SUITE2P, ENGERT_MAG_TIFF, length=20)
    vis_tiff, vis_F, vis_stat = engert_helpers.load_GEVI(ENGERT_VIS_SUITE2P, ENGERT_VIS_TIFF, length=20)

    # Load iscell probabilities for filtering
    iscell_threshold = 0.6
    npix_threshold = 30
    mag_iscell = np.load(ENGERT_MAG_SUITE2P.replace("plane0", "plane0") + "/iscell.npy")

    mag_GECI_spectra = engert_helpers.fit_Fourier(mag_F, 1, f=0.4, Q=50)
    vis_GECI_spectra = engert_helpers.fit_Fourier(vis_F, 1, f=1/60, Q=6)

    font = {"family": "arial", "size": 6}
    matplotlib.rc("font", **font)

    fig = plt.figure(figsize=(9, 6), tight_layout=True)
    gs = gridspec.GridSpec(3, 6, left=0, bottom=0, right=1, top=1, wspace=0.35, hspace=0.4)

    # ── Row 1: Cartoon + Mag raw NPIX + Anatomy + Vis calcium ────────────────
    cartoon_ax = fig.add_subplot(gs[0, 0])
    mag_raw_ax = fig.add_subplot(gs[0, 1:3])
    anatomy_ax = fig.add_subplot(gs[0, 3])
    vis_calcium_ax = fig.add_subplot(gs[0, 4:])

    # ── Row 2: Mag spectrum + Mag distribution + Combined ECDF ───────────────
    mag_spectra_ax = fig.add_subplot(gs[1, 0])
    mag_dist_ax = fig.add_subplot(gs[1, 1])

    # ── Row 3: Vis spectrum + Vis distribution ─────────────────────────────
    vis_spectra_ax = fig.add_subplot(gs[2, 0])
    vis_dist_ax = fig.add_subplot(gs[2, 1])

    # ── Combined ECDF (spans rows 2-3, columns 2-5) ────────────────────────
    ecdf_ax = fig.add_subplot(gs[1:3, 2:])

    # ── Plot Row 1 ────────────────────────────────────────────────────────────
    # Cartoon placeholder
    cartoon_ax.add_patch(patches.Rectangle((0.05, 0.1), 0.9, 0.8, fill=False, edgecolor="black", linewidth=1))
    cartoon_ax.text(0.5, 0.5, "Neuropixel\nProbe", ha="center", va="center", fontsize=8)
    cartoon_ax.set_xlim(0, 1)
    cartoon_ax.set_ylim(0, 1)
    cartoon_ax.axis("off")

    # Mag raw NPIX
    mag_ldr = openEphysIO.Loader(mag_contingency_path.replace("\\", "/"), cntlbarcodes=True)
    statistics.raw_NPIX(mag_raw_ax, mag_ldr, mag_spks, mag_unitrow, (55.8, 56.3), 5, label=0.100, DX=300)

    # Anatomy (mag only) — direct visualization
    anatomy_img = np.mean(mag_tiff, axis=0)
    vmin = np.percentile(anatomy_img, 5)
    vmax = np.percentile(anatomy_img, 95)
    anatomy_ax.imshow(anatomy_img, cmap="gray", vmin=vmin, vmax=vmax)

    # Plot all qualifying cell masks
    h, w = anatomy_img.shape
    for i, cell in enumerate(mag_stat):
        # Only plot if cell passes filtering thresholds
        if mag_iscell[i, 0] < iscell_threshold or len(cell['xpix']) < npix_threshold:
            continue

        cell_y = cell['ypix']
        cell_x = cell['xpix']

        # Create binary mask for this cell
        mask = np.zeros((h, w), dtype=bool)
        mask[cell_y, cell_x] = True

        if i == MAG_CELL_IND:
            # Exemplar cell: draw filled region in red
            anatomy_ax.contourf(mask, levels=[0.5, 1.5], colors=['red'], alpha=0.5)
        else:
            # Other cells: draw outline in cyan
            anatomy_ax.contour(mask, levels=[0.5], colors=['cyan'], linewidths=0.5, alpha=0.6)

    anatomy_ax.axis("off")

    # Vis calcium trace
    statistics.raw_GECI(vis_calcium_ax, vis_F, VIS_CELL_IND)

    # ── Plot Row 2: Magnetic stimulation ──────────────────────────────────────
    (C, T, nn, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, exemplar_c_hat) = mag_exemplar_fourier
    statistics.plot_spectrum(mag_spectra_ax, fou_alt.flatten(), ff_alt, 5, fou0, legend=False)
    mag_spectra_ax.set_ylabel("Amplitude")
    mag_spectra_ax.set_xlabel("Frequency (Hz)")
    statistics.boundary_ticks(mag_spectra_ax)
    statistics.nestle_labels(mag_spectra_ax, y=True, x=True, x_offset=-0.05)

    statistics.draw_hist(fourier_df.loc[fourier_df.rec == MAG_CONTINGENCY, "rr"], mag_dist_ax, xlim=9, inset=True)
    mag_dist_ax.annotate("", (exemplar_c_hat, 0.6), xytext=(exemplar_c_hat, 0.8),
                         textcoords="data", arrowprops=dict(facecolor="black", arrowstyle="->"))
    statistics.boundary_ticks(mag_dist_ax, yprec=1)
    statistics.nestle_labels(mag_dist_ax, x_offset=-0.05, y_offset=-0.05)

    # ── Plot Row 3: Visual stimulation ────────────────────────────────────────
    (C, T, nn, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, exemplar_c_hat_vis) = vis_exemplar_fourier
    statistics.plot_spectrum(vis_spectra_ax, fou_alt.flatten(), ff_alt, 3, fou0, legend=False)
    vis_spectra_ax.set_ylabel("Amplitude")
    vis_spectra_ax.set_xlabel("Frequency (Hz)")
    statistics.boundary_ticks(vis_spectra_ax)
    statistics.nestle_labels(vis_spectra_ax, y=True, x=True, x_offset=-0.05)

    statistics.draw_hist(fourier_df.loc[fourier_df.rec == VIS_CONTINGENCY + "_90", "rr"],
                         vis_dist_ax, xlim=9, inset=True)
    vis_dist_ax.annotate("", (exemplar_c_hat_vis, 0.2), xytext=(exemplar_c_hat_vis, 0.4),
                         textcoords="data", arrowprops=dict(facecolor="black", arrowstyle="->"))
    statistics.boundary_ticks(vis_dist_ax, yprec=1)
    statistics.nestle_labels(vis_dist_ax, x_offset=-0.05, y_offset=-0.05)

    # ── Kolmogorov-Smirnov diagnostic plot ────────────────────────────────────
    mag_c_hat = fourier_df.loc[fourier_df.rec == MAG_CONTINGENCY, "rr"].values
    vis_c_hat = fourier_df.loc[fourier_df.rec == VIS_CONTINGENCY + "_90", "rr"].values

    # Convert c-hat to p-values
    mag_pvals = 1 - normalized_Fourier_CDF(mag_c_hat)
    vis_pvals = 1 - normalized_Fourier_CDF(vis_c_hat)

    # Plot mag K-S diagnostic: ECDF(x) - x with 95% CI
    mag_x, mag_lower, mag_upper = bootstrap_ecdf_band(mag_pvals)
    # Empirical ECDF at sorted points
    mag_ecdf = (np.arange(1, len(mag_pvals) + 1)) / len(mag_pvals)
    # K-S deviation: empirical ECDF minus theoretical (uniform) CDF
    mag_ks_dev = mag_ecdf - mag_x
    mag_ks_lower = mag_lower - mag_x
    mag_ks_upper = mag_upper - mag_x
    ecdf_ax.plot(mag_x, mag_ks_dev, color="steelblue", linewidth=1, label="5 Hz (mag)")
    ecdf_ax.fill_between(mag_x, mag_ks_lower, mag_ks_upper, color="steelblue", alpha=0.2)

    # Plot vis K-S diagnostic: ECDF(x) - x with 95% CI
    vis_x, vis_lower, vis_upper = bootstrap_ecdf_band(vis_pvals)
    # Empirical ECDF at sorted points
    vis_ecdf = (np.arange(1, len(vis_pvals) + 1)) / len(vis_pvals)
    # K-S deviation: empirical ECDF minus theoretical (uniform) CDF
    vis_ks_dev = vis_ecdf - vis_x
    vis_ks_lower = vis_lower - vis_x
    vis_ks_upper = vis_upper - vis_x
    ecdf_ax.plot(vis_x, vis_ks_dev, color="coral", linewidth=1, label="3 Hz (vis)")
    ecdf_ax.fill_between(vis_x, vis_ks_lower, vis_ks_upper, color="coral", alpha=0.2)

    # Plot null line (y=0, representing perfect agreement with uniform distribution)
    ecdf_ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ecdf_ax.set_xlabel("p-value")
    ecdf_ax.set_ylabel("ECDF deviation from null")
    ecdf_ax.set_xlim((0, 1))
    ecdf_ax.legend(loc="upper right", fontsize=7)
    statistics.boundary_ticks(ecdf_ax)
    statistics.nestle_labels(ecdf_ax, y=True, x_offset=-0.05)
    statistics.boundary_ticks(ecdf_ax)
    statistics.nestle_labels(ecdf_ax, y=True, x_offset=-0.05)

    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    fig.canvas.draw()

    # ── Panel labels ──────────────────────────────────────────────────────────
    # A-D: place at a shared figure-level y so they sit on the same horizontal
    # line despite the anatomy image forcing a smaller axis height.
    _row0 = [
        (cartoon_ax,     "A", -0.15),
        (mag_raw_ax,     "B", -0.05),
        (anatomy_ax,     "C", -0.15),
        (vis_calcium_ax, "D", -0.15),
    ]
    _label_y = max(ax.get_position().y1 for ax, _, _ in _row0) + 0.01
    for _ax, _lbl, _xoff in _row0:
        _pos = _ax.get_position()
        fig.text(_pos.x0 + _xoff * _pos.width, _label_y, _lbl,
                 fontfamily="arial", fontsize=11, weight="bold")

    mag_spectra_ax.annotate("E", xy=(-0.15, 1.1), xycoords="axes fraction", fontfamily="arial", fontsize=11, weight="bold")
    mag_dist_ax.annotate("F", xy=(-0.15, 1.1), xycoords="axes fraction", fontfamily="arial", fontsize=11, weight="bold")
    ecdf_ax.annotate("G", xy=(-0.03, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=11, weight="bold")

    vis_spectra_ax.annotate("H", xy=(-0.15, 1.1), xycoords="axes fraction", fontfamily="arial", fontsize=11, weight="bold")
    vis_dist_ax.annotate("I", xy=(-0.15, 1.1), xycoords="axes fraction", fontfamily="arial", fontsize=11, weight="bold")
    out_path = out_dir / "Fig1.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 1 (composite NPIX + GCaMP + ECDFs)")
    parser.add_argument("--out-dir", default="../../figs/paper", help="Output directory for PDFs")
    parser.add_argument("--data-dir", default="../../data", help="Directory containing pipeline output pickles")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading NPIX + GCaMP data...")
    modulation_df, fourier_df, udf = load_data(args.data_dir)
    print("Loading GCaMP data (requires NAS)...")
    plot_fig1_composite(modulation_df, fourier_df, udf, out_dir)


try:
    from IPython import get_ipython
    in_notebook = get_ipython() is not None
except ImportError:
    in_notebook = False

if __name__ == "__main__":
    if in_notebook:
        print("Running in Jupyter notebook.")
        print("Call plot_fig1_composite() with your own args, or use: main()")
    else:
        main()
