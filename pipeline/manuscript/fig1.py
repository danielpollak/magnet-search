"""Fig 1 — exemplar NPIX unit (top) and exemplar GCaMP cell (bottom).

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
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ephysio import openEphysIO
from magpyneto2 import statistics, engert_helpers
from magpyneto2.utils import get_cluster_info

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


def load_data(data_dir: str):
    data_dir = Path(data_dir)
    with open(data_dir / "20230413_firstsite_processing.pickle", "rb") as f:
        modulation_df = pickle.load(f)
    fourier_df = pd.read_pickle(data_dir / "20230413_firstsite_analysis.pickle")
    udf = get_cluster_info(DATA_PATH)
    udf = udf.loc[udf.KSLabel == "good", :]
    return modulation_df, fourier_df, udf


def plot_fig1_top(modulation_df, fourier_df, udf, out_dir: Path):
    mag_unitrow = udf.loc[udf.cluster_id == MAG_CLUSTER_ID].squeeze()
    vis_unitrow = udf.loc[udf.cluster_id == VIS_CLUSTER_ID].squeeze()

    mag_allspks, mag_spks, mag_exemplar_fourier, mag_contingency_path = \
        statistics.Fig1_NPIX_data(modulation_df, MAG_CONTINGENCY, mag_unitrow, 5)
    vis_allspks, vis_spks, vis_exemplar_fourier, vis_contingency_path = \
        statistics.Fig1_NPIX_data(modulation_df, VIS_CONTINGENCY, vis_unitrow, 3)

    font = {"family": "arial", "size": 6}
    matplotlib.rc("font", **font)

    fig = plt.figure(figsize=(6, 3), tight_layout=True)
    gs = gridspec.GridSpec(2, 5, left=0, bottom=0, right=1, top=1, wspace=0.4, hspace=0.5)

    mag_raw_ax      = fig.add_subplot(gs[0, :-2])
    vis_raw_ax      = fig.add_subplot(gs[1, :-2])
    mag_spectra_ax  = fig.add_subplot(gs[0, -2])
    vis_spectra_ax  = fig.add_subplot(gs[1, -2])
    mag_dist_ax     = fig.add_subplot(gs[0, -1])
    vis_dist_ax     = fig.add_subplot(gs[1, -1])

    mag_ldr = openEphysIO.Loader(mag_contingency_path.replace("\\", "/"), cntlbarcodes=True)
    statistics.raw_NPIX(mag_raw_ax, mag_ldr, mag_spks, mag_unitrow, (55.8, 56.3), 5, label=0.100, DX=300)

    vis_ldr = openEphysIO.Loader(vis_contingency_path.replace("\\", "/"), cntlbarcodes=True)
    statistics.raw_NPIX(vis_raw_ax, vis_ldr, vis_spks, vis_unitrow, (242, 242.5), 3, label=0.100, DX=300)

    (C, T, nn, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, exemplar_c_hat) = mag_exemplar_fourier
    statistics.plot_spectrum(mag_spectra_ax, fou_alt.flatten(), ff_alt, 5, fou0, legend=True)
    mag_spectra_ax.set_ylabel("Component\nAmplitudes")

    statistics.draw_hist(fourier_df.loc[fourier_df.rec == MAG_CONTINGENCY, "rr"], mag_dist_ax, xlim=9, inset=True)
    mag_dist_ax.annotate("", (exemplar_c_hat, 0.6), xytext=(exemplar_c_hat, 0.8),
                         textcoords="data", arrowprops=dict(facecolor="black", arrowstyle="->"))

    (C, T, nn, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, exemplar_c_hat) = vis_exemplar_fourier
    statistics.plot_spectrum(vis_spectra_ax, fou_alt.flatten(), ff_alt, 3, fou0, legend=False)
    vis_spectra_ax.set_ylabel("Component\nAmplitudes")
    vis_spectra_ax.set_xlabel("Frequency (Hz)")
    mag_spectra_ax.set_xlabel("Frequency (Hz)")

    statistics.boundary_ticks(vis_spectra_ax)
    statistics.nestle_labels(vis_spectra_ax, y=True, x=True, x_offset=-0.05)
    statistics.boundary_ticks(mag_spectra_ax)
    statistics.nestle_labels(mag_spectra_ax, y=True, x=True, x_offset=-0.05)

    statistics.draw_hist(fourier_df.loc[fourier_df.rec == VIS_CONTINGENCY + "_90", "rr"],
                         vis_dist_ax, xlim=9, inset=True)
    vis_dist_ax.annotate("", (exemplar_c_hat, 0.2), xytext=(exemplar_c_hat, 0.4),
                         textcoords="data", arrowprops=dict(facecolor="black", arrowstyle="->"))

    statistics.boundary_ticks(vis_dist_ax, yprec=1)
    statistics.nestle_labels(vis_dist_ax, x_offset=-0.05, y_offset=-0.05)
    statistics.boundary_ticks(mag_dist_ax, yprec=1)
    statistics.nestle_labels(mag_dist_ax, x_offset=-0.05, y_offset=-0.05)

    mag_raw_ax.annotate(    "A", xy=(-0.01, 1.1), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    mag_spectra_ax.annotate("B", xy=(-0.2,  1.1), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    mag_dist_ax.annotate(   "C", xy=(-0.2,  1.1), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    vis_raw_ax.annotate(    "D", xy=(-0.01, 1.1), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    vis_spectra_ax.annotate("E", xy=(-0.2,  1.1), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    vis_dist_ax.annotate(   "F", xy=(-0.2,  1.1), xycoords="axes fraction", fontfamily="arial", fontsize=12)

    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    out_path = out_dir / "Fig1_top.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_fig1_bottom(out_dir: Path):
    mag_tiff, mag_F, mag_stat = engert_helpers.load_GEVI(ENGERT_MAG_SUITE2P, ENGERT_MAG_TIFF, length=20)
    vis_tiff, vis_F, vis_stat = engert_helpers.load_GEVI(ENGERT_VIS_SUITE2P, ENGERT_VIS_TIFF, length=20)

    mag_GECI_spectra = engert_helpers.fit_Fourier(mag_F, 1, f=0.4, Q=50)
    vis_GECI_spectra = engert_helpers.fit_Fourier(vis_F, 1, f=1/60, Q=6)

    font = {"family": "arial", "size": 6}
    matplotlib.rc("font", **font)

    fig = plt.figure(figsize=(6, 2), tight_layout=True)
    gs = gridspec.GridSpec(2, 5, left=0, bottom=0, right=1, top=1, wspace=0.4, hspace=0.5)

    anatomy_ax      = fig.add_subplot(gs[:, 0])
    mag_raw_ax      = fig.add_subplot(gs[0, 1:-2])
    mag_dist_ax     = fig.add_subplot(gs[0, -1])
    mag_spectra_ax  = fig.add_subplot(gs[0, -2])
    vis_raw_ax      = fig.add_subplot(gs[1, 1:-2])
    vis_dist_ax     = fig.add_subplot(gs[1, -1])
    vis_spectra_ax  = fig.add_subplot(gs[1, -2])

    engert_helpers.var_projection_GCaMP(
        anatomy_ax, mag_tiff, vis_tiff, mag_stat,
        cell_inds=[MAG_CELL_IND, VIS_CELL_IND])

    statistics.raw_GECI(mag_raw_ax, mag_F, MAG_CELL_IND)
    statistics.raw_GECI(vis_raw_ax, vis_F, VIS_CELL_IND)

    chat_l, onfreq_pow_l, offfreq_pow_l, freq_win = mag_GECI_spectra
    statistics.plot_spectrum(mag_spectra_ax, offfreq_pow_l[MAG_CELL_IND], freq_win,
                             0.4, np.array([onfreq_pow_l[MAG_CELL_IND]]))
    statistics.draw_hist(np.array(chat_l), mag_dist_ax, xlim=9, inset=True)
    mag_dist_ax.annotate("", (chat_l[MAG_CELL_IND], 0.4), xytext=(chat_l[MAG_CELL_IND], 0.6),
                         textcoords="data", arrowprops=dict(facecolor="black", arrowstyle="->"))

    chat_l, onfreq_pow_l, offfreq_pow_l, freq_win = vis_GECI_spectra
    statistics.plot_spectrum(vis_spectra_ax, offfreq_pow_l[VIS_CELL_IND], freq_win,
                             1/60, np.array([onfreq_pow_l[VIS_CELL_IND]]))
    statistics.draw_hist(np.array(chat_l), vis_dist_ax, xlim=9, inset=True)
    vis_dist_ax.annotate("", (chat_l[VIS_CELL_IND], 0.2), xytext=(chat_l[VIS_CELL_IND], 0.4),
                         textcoords="data", arrowprops=dict(facecolor="black", arrowstyle="->"))

    mag_spectra_ax.set_xlabel("Frequency (Hz)")
    vis_spectra_ax.set_xlabel("Frequency (Hz)")
    statistics.boundary_ticks(vis_spectra_ax, xprec=3)
    statistics.nestle_labels(vis_spectra_ax, y=True, x_offset=-0.05)
    statistics.boundary_ticks(mag_spectra_ax, xprec=3)
    statistics.nestle_labels(mag_spectra_ax, y=True, x_offset=-0.05)
    statistics.boundary_ticks(vis_dist_ax, yprec=1)
    statistics.nestle_labels(vis_dist_ax, x_offset=-0.05, y_offset=-0.05)
    statistics.boundary_ticks(mag_dist_ax, yprec=1)
    statistics.nestle_labels(mag_dist_ax, x_offset=-0.05, y_offset=-0.05)

    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

    anatomy_ax.annotate(         "G", xy=(-0.07, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    mag_raw_ax.annotate(         "H", xy=(-0.02,  1.1), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    mag_spectra_ax.annotate(     "I", xy=(-0.2,   1.1), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    mag_dist_ax.annotate(        "J", xy=(-0.01,  1.1), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    anatomy_ax.annotate(         "K", xy=(-0.07, 0.45), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    vis_raw_ax.annotate(         "L", xy=(-0.02,  1.1), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    vis_dist_ax.annotate(        "M", xy=(-0.2,   1.1), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    vis_spectra_ax.annotate(     "N", xy=(-0.2,   1.1), xycoords="axes fraction", fontfamily="arial", fontsize=12)

    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    out_path = out_dir / "Fig1_bottom.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=1200)
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 1 (NPIX + GCaMP)")
    parser.add_argument("--out-dir", default="../../figs/paper", help="Output directory for PDFs")
    parser.add_argument("--data-dir", default="../../data", help="Directory containing pipeline output pickles")
    parser.add_argument("--top-only", action="store_true", help="Only generate Fig 1 top (NPIX)")
    parser.add_argument("--bottom-only", action="store_true", help="Only generate Fig 1 bottom (GCaMP)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.bottom_only:
        print("Loading NPIX data...")
        modulation_df, fourier_df, udf = load_data(args.data_dir)
        plot_fig1_top(modulation_df, fourier_df, udf, out_dir)

    if not args.top_only:
        print("Loading GCaMP data (requires NAS)...")
        plot_fig1_bottom(out_dir)


if __name__ == "__main__":
    main()
