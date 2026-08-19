"""Fig 1 (auditory variant) — composite: NPIX null result + NPIX positive result + spectra + p-value ECDFs.

This is a previous version of Fig 1, kept alongside the current primary
version (pipeline/manuscript/fig1.py, which instead compares magnetic
stimulation to 2/3Hz visual gratings, cluster 540/experiment
20230413_secondsite) for reference/comparison -- see that file's docstring
for why the visual comparison was ultimately preferred (mainly: a much
stronger population-level "positive control" story in panels D-I).

Both the null (magnetic stimulus) and positive (white noise) panels come from
the SAME exemplar unit (cluster 106, experiment 20230415, Pigeon W1R,
hippocampus) -- it's non-significant across all 6 of its own magnetic-stimulus
trials and both visual-grating conditions, yet strongly significant to white
noise (NFC=12.26). Using one neuron/session/probe for both panels is a
stronger story than two different units from two different sessions: it
preempts the "maybe that channel just wasn't picking up signal" critique,
since the same channel clearly does pick up signal (for WN) but just doesn't
respond to the magnet.

Phasor arrows in panels B/C are colored by phase (cyclic `hsv` colormap, 0 to
one stimulus period) -- see the colorwheel legend inset in panel A. (This
coloring comes from magpyneto2.statistics.raw_NPIX, shared with fig1.py --
added after this file was originally written, so this file was updated to
pass the same `hsv` colormap explicitly rather than silently picking up
raw_NPIX's default, which has a visibility bug -- see fig1.py's docstring.)

Requires pipeline outputs:
  data/20230415.nwb

Requires pre-extracted raw-voltage snippets (see
fig1_auditory_extract_raw_snippets.py, which is the only piece of this
figure that needs NAS access):
  data/fig1_raw_auditory/mag_trace.npy
  data/fig1_raw_auditory/wn_trace.npy
  data/fig1_raw_auditory/wn_waveforms.npy

Usage:
    python pipeline/manuscript/fig1_auditory.py
    python pipeline/manuscript/fig1_auditory.py --out-dir figs/paper
"""
#%%
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
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecdfbounds import bootstrap_ecdf_band
from magpyneto2 import statistics
from magpyneto2.statistics import normalized_Fourier_CDF
from pipeline import nwb_io
from pipeline.schema import load_experiment

import format_parameters as FP
#%%
# ── NAS paths (only used by fig1_auditory_extract_raw_snippets.py, not this script) ──
RAW_DATA_ROOT = r"\\datanas\family\data_raw\20230415"
AGGREGATED_PATH = r"\\datanas\family\data_aggregated\20230415"

# ── Experiment / exemplar unit ────────────────────────────────────────────────
EXPERIMENT = "20230415"
CLUSTER_ID = 106  # single unit, used for both the null and positive panels

MAG_CONTINGENCY = "2023-04-15_15-56-12_W1R_mag2"
MAG_FREQ = 2
# 2Hz, not 8Hz: this unit's session has mag trials at 2/3/8Hz (plus inclined
# variants) -- 2Hz sits much closer to the WN frequency (0.8Hz) than 8Hz did,
# making the null-vs-positive comparison less confounded by a stark frequency
# mismatch. Confirmed non-significant here too (NFC=1.13, p=0.53); the other
# 5 mag trials for this cluster (3Hz, 8Hz, and their "inclined" variants) are
# all also non-significant if a different frequency is ever needed instead.
MAG_WINDOW = (63.158, 64.158)  # 1s window = 2 cycles at 2Hz (period 0.5s)

WN_CONTINGENCY = "2023-04-15_16-37-23_W1R_3D_WN_Samechan"
WN_FREQ = 0.8
# 2.5s = 2 cycles at 0.8Hz (period 1.25s) -- same "2 stimulus periods shown"
# convention as the mag panel above.
WN_WINDOW = (144.813, 147.313)

# Sampling rates for the pre-extracted raw-voltage snippets (see
# fig1_auditory_extract_raw_snippets.py, which prints these when it runs).
MAG_TRACE_SR = 30000
WN_TRACE_SR = 30000

# Spike-raster tick linewidth, shared by both raw_NPIX panels (default is 2,
# which reads as visually heavy/blocky once many ticks are packed into a
# wider window).
RASTER_LW = 0.5

# Phasor-arrow half-length (samples), shared since both raw_NPIX panels are
# now sized proportional to their own window duration (see row0_width_ratios
# in plot_fig1_composite) -- spikes-per-pixel is comparable between the two
# panels, so there's no need to shrink DX or subsample phasors for either.
MAG_DX = 300
WN_DX = 300

# Cyclic colormap for phase-coloring the phasor arrows in both raw_NPIX
# panels -- paired with a colorwheel legend (statistics.plot_phase_colorwheel)
# drawn as an inset in the cartoon panel (A). "twilight" (matplotlib's usual
# cyclic recommendation, and raw_NPIX's default) has a built-in near-white
# anchor at phase=0, which makes those arrows nearly invisible against this
# figure's white background -- "hsv" has no light/white point anywhere in its
# cycle, staying visible throughout, at the cost of being less perceptually
# uniform. (See fig1.py, the current primary Fig1, where this was chosen.)
PHASE_CMAP = "hsv"

# ── Average spike-waveform panel ──────────────────────────────────────────────
# 100 randomly-sampled raw waveforms (seeded for reproducibility) for this
# unit, drawn from the WN recording (largest spike count of the two).
N_WAVEFORMS = 100
WAVEFORM_HALFWIDTH_MS = 1.0
WAVEFORM_SEED = 0

#%%

def load_data(data_dir: str):
    cfg = load_experiment(Path(__file__).parent.parent.parent / "experiments" / f"{EXPERIMENT}.yml")
    io_r, nwbfile = nwb_io.read_nwbfile(str(Path(data_dir) / f"{EXPERIMENT}.nwb"))
    modulation_df = nwb_io.build_modulation_frame(nwbfile, good_only=cfg.good)
    fourier_df = nwb_io.read_fourier_results_as_full_fourier_df(nwbfile)
    io_r.close()
    return modulation_df, fourier_df


def plot_fig1_composite(modulation_df, fourier_df, out_dir: Path):
    # Fig1_NPIX_data only reads unitrow.cluster_id (not .ch) -- no NAS-backed
    # cluster_info.tsv lookup needed here, just the cluster id itself.
    unitrow = pd.Series({"cluster_id": CLUSTER_ID})

    mag_allspks, mag_spks, mag_exemplar_fourier, _ = \
        statistics.Fig1_NPIX_data(modulation_df, MAG_CONTINGENCY, unitrow, MAG_FREQ)
    wn_allspks, wn_spks, wn_exemplar_fourier, _ = \
        statistics.Fig1_NPIX_data(modulation_df, WN_CONTINGENCY, unitrow, WN_FREQ)

    # Pre-extracted raw-voltage snippets (see
    # fig1_auditory_extract_raw_snippets.py) -- the only NAS-derived inputs to
    # this figure, cached locally so this script itself needs no NAS access.
    raw_dir = Path(FP.DATA_DIR) / "fig1_raw_auditory"
    mag_trace = np.load(raw_dir / "mag_trace.npy")
    wn_trace = np.load(raw_dir / "wn_trace.npy")
    waveforms = np.load(raw_dir / "wn_waveforms.npy")  # (N_WAVEFORMS, n_samples)

    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY}
    matplotlib.rc("font", **font)

    fig = plt.figure(figsize=FP.FIGSIZE_FIG1, tight_layout=True)
    outer_gs = gridspec.GridSpec(3, 1, left=0, bottom=0, right=1, top=1, hspace=0.4)

    # ── Row 1: Cartoon + Mag raw NPIX (null) + WN raw NPIX (positive) ───────
    # The two raw-trace panels' column widths are proportional to their own
    # displayed window duration -- otherwise a much-wider WN window gets
    # squeezed into the same column width as the narrower mag window and
    # looks compressed/dense. The cartoon isn't a time-domain panel, so its
    # width is a small fixed ratio (independent of the window durations,
    # deliberately squished) rather than matching either panel's width, so
    # the two raw-trace panels get most of the row.
    CARTOON_WIDTH_RATIO = 0.5
    mag_dur = MAG_WINDOW[1] - MAG_WINDOW[0]
    wn_dur = WN_WINDOW[1] - WN_WINDOW[0]
    row0_gs = outer_gs[0].subgridspec(1, 3, width_ratios=[CARTOON_WIDTH_RATIO, mag_dur, wn_dur], wspace=0.15)
    cartoon_ax = fig.add_subplot(row0_gs[0, 0])
    mag_raw_ax = fig.add_subplot(row0_gs[0, 1])
    wn_raw_ax = fig.add_subplot(row0_gs[0, 2])

    # ── Rows 2-3: Mag/WN spectra + distributions + combined ECDF ────────────
    rows12_gs = outer_gs[1:3].subgridspec(2, 6, wspace=0.35, hspace=0.4)
    mag_spectra_ax = fig.add_subplot(rows12_gs[0, 0])
    mag_dist_ax = fig.add_subplot(rows12_gs[0, 1])
    wn_spectra_ax = fig.add_subplot(rows12_gs[1, 0])
    wn_dist_ax = fig.add_subplot(rows12_gs[1, 1])
    # Combined ECDF (spans both rows, columns 2-end)
    ecdf_ax = fig.add_subplot(rows12_gs[0:2, 2:])

    # ── Plot Row 1 ────────────────────────────────────────────────────────────
    # Cartoon placeholder
    cartoon_ax.add_patch(patches.Rectangle((0.05, 0.1), 0.9, 0.8, fill=False, edgecolor="black", linewidth=1))
    cartoon_ax.text(0.5, 0.5, "Neuropixel\nProbe", ha="center", va="center", fontsize=8)
    cartoon_ax.set_xlim(0, 1)
    cartoon_ax.set_ylim(0, 1)
    cartoon_ax.axis("off")

    # Mag raw NPIX (null result) -- scale bar shows one stimulus period (1/freq)
    statistics.raw_NPIX(mag_raw_ax, None, mag_spks, None, MAG_WINDOW, MAG_FREQ,
                         label=1 / MAG_FREQ, DX=MAG_DX, trace=mag_trace, spike_sr=MAG_TRACE_SR,
                         raster_lw=RASTER_LW, phase_cmap=PHASE_CMAP)

    # WN raw NPIX (positive result) -- same unit, same primitive as the mag panel
    statistics.raw_NPIX(wn_raw_ax, None, wn_spks, None, WN_WINDOW, WN_FREQ,
                         label=1 / WN_FREQ, DX=WN_DX, trace=wn_trace, spike_sr=WN_TRACE_SR,
                         raster_lw=RASTER_LW, phase_cmap=PHASE_CMAP)

    # Phase colorwheel legend for the phasor arrows above -- tucked into the
    # cartoon panel's lower-right quadrant, clear of the centered probe label.
    wheel_ax = cartoon_ax.inset_axes([0.55, 0.15, 0.33, 0.3])
    statistics.plot_phase_colorwheel(wheel_ax, cmap=PHASE_CMAP)

    # ── Plot Row 2: Magnetic stimulation (null) ──────────────────────────────
    (C, T, spk_count, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, exemplar_NFC) = mag_exemplar_fourier
    statistics.plot_spectrum(mag_spectra_ax, fou_alt.flatten(), ff_alt, MAG_FREQ, fou0, legend=False)
    mag_spectra_ax.set_ylabel("Amplitude")
    mag_spectra_ax.set_xlabel("Frequency (Hz)")
    statistics.boundary_ticks(mag_spectra_ax)
    statistics.nestle_labels(mag_spectra_ax, y=True, x=True, x_offset=-0.05)

    statistics.draw_hist(fourier_df.loc[fourier_df.rec == MAG_CONTINGENCY, "NFC"], mag_dist_ax, xlim=9, inset=True)
    mag_dist_ax.annotate("", (exemplar_NFC, 0.6), xytext=(exemplar_NFC, 0.8),
                         textcoords="data", arrowprops=dict(facecolor="black", arrowstyle="->"))
    statistics.boundary_ticks(mag_dist_ax, yprec=1)
    statistics.nestle_labels(mag_dist_ax, x_offset=-0.05, y_offset=-0.05)

    # ── Plot Row 3: White noise (positive) ──────────────────────────────────
    (C, T, spk_count, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, exemplar_NFC_wn) = wn_exemplar_fourier
    statistics.plot_spectrum(wn_spectra_ax, fou_alt.flatten(), ff_alt, WN_FREQ, fou0, legend=False)
    wn_spectra_ax.set_ylabel("Amplitude")
    wn_spectra_ax.set_xlabel("Frequency (Hz)")
    statistics.boundary_ticks(wn_spectra_ax)
    statistics.nestle_labels(wn_spectra_ax, y=True, x=True, x_offset=-0.05)

    statistics.draw_hist(fourier_df.loc[fourier_df.rec == WN_CONTINGENCY, "NFC"],
                         wn_dist_ax, xlim=16, inset=True)
    # exemplar_NFC_wn (~13.3) sits far out in the histogram's near-empty right
    # tail, unlike the mag panel's arrow -- point straight down at the axis
    # floor at the exemplar's actual x position rather than reusing the mag
    # panel's mid-height arrow coordinates.
    wn_dist_ax.annotate("", (exemplar_NFC_wn, 0.03), xytext=(exemplar_NFC_wn, 0.25),
                         textcoords="data", arrowprops=dict(facecolor="black", arrowstyle="->"))
    statistics.boundary_ticks(wn_dist_ax, yprec=1)
    statistics.nestle_labels(wn_dist_ax, x_offset=-0.05, y_offset=-0.05)

    # ── Kolmogorov-Smirnov diagnostic plot ────────────────────────────────────
    mag_NFC = fourier_df.loc[fourier_df.rec == MAG_CONTINGENCY, "NFC"].values
    wn_NFC = fourier_df.loc[fourier_df.rec == WN_CONTINGENCY, "NFC"].values

    # Convert NFC to p-values
    mag_pvals = 1 - normalized_Fourier_CDF(mag_NFC)
    wn_pvals = 1 - normalized_Fourier_CDF(wn_NFC)

    # Plot mag K-S diagnostic: ECDF(x) - x with 95% CI
    mag_x, mag_lower, mag_upper = bootstrap_ecdf_band(mag_pvals)
    # Empirical ECDF at sorted points
    mag_ecdf = (np.arange(1, len(mag_pvals) + 1)) / len(mag_pvals)
    # K-S deviation: empirical ECDF minus theoretical (uniform) CDF
    mag_ks_dev = mag_ecdf - mag_x
    mag_ks_lower = mag_lower - mag_x
    mag_ks_upper = mag_upper - mag_x
    ecdf_ax.plot(mag_x, mag_ks_dev, color=FP.COLOR_MAG, linewidth=FP.LW_TRACE, label=f"{MAG_FREQ:g} Hz (mag)")
    ecdf_ax.fill_between(mag_x, mag_ks_lower, mag_ks_upper, color=FP.COLOR_MAG, alpha=FP.ALPHA_CONFIDENCE)

    # Plot WN K-S diagnostic: ECDF(x) - x with 95% CI
    wn_x, wn_lower, wn_upper = bootstrap_ecdf_band(wn_pvals)
    # Empirical ECDF at sorted points
    wn_ecdf = (np.arange(1, len(wn_pvals) + 1)) / len(wn_pvals)
    # K-S deviation: empirical ECDF minus theoretical (uniform) CDF
    wn_ks_dev = wn_ecdf - wn_x
    wn_ks_lower = wn_lower - wn_x
    wn_ks_upper = wn_upper - wn_x
    ecdf_ax.plot(wn_x, wn_ks_dev, color=FP.COLOR_VIS, linewidth=FP.LW_TRACE, label=f"{WN_FREQ:g} Hz (WN)")
    ecdf_ax.fill_between(wn_x, wn_ks_lower, wn_ks_upper, color=FP.COLOR_VIS, alpha=FP.ALPHA_CONFIDENCE)

    # Plot null line (y=0, representing perfect agreement with uniform distribution)
    ecdf_ax.axhline(0, color=FP.COLOR_NULL, linestyle="--", linewidth=FP.LW_REFERENCE, alpha=0.6)

    ecdf_ax.set_xlabel("p-value")
    ecdf_ax.set_ylabel("ECDF deviation from null")
    ecdf_ax.set_xlim((0, 1))
    ecdf_ax.legend(loc="upper right", fontsize=7)
    statistics.boundary_ticks(ecdf_ax)
    statistics.nestle_labels(ecdf_ax, y=True, x_offset=-0.05)
    statistics.boundary_ticks(ecdf_ax)
    statistics.nestle_labels(ecdf_ax, y=True, x_offset=-0.05)

    # ── Average spike-waveform inset (100 individual raw draws, grey/thin/low
    # alpha, + mean in black) -- placed inside the ECDF panel to save the
    # horizontal room a standalone top-row panel would need. Tucked into the
    # p->1 corner: by construction, ECDF(p)-p is pinned back toward 0 at both
    # p=0 and p=1 regardless of the underlying data (a boundary condition of
    # this Brownian-bridge-style deviation plot), so that corner is the least
    # informative part of the curves to sit an opaque inset on top of.
    waveform_ax = ecdf_ax.inset_axes([0.78, 0.03, 0.21, 0.27])
    n_samples = waveforms.shape[1]
    half_width_samples = n_samples // 2
    t_ms = (np.arange(n_samples) - half_width_samples) / WN_TRACE_SR * 1000
    for wf in waveforms:
        waveform_ax.plot(t_ms, wf, color="grey", linewidth=0.3, alpha=0.15)
    waveform_ax.plot(t_ms, waveforms.mean(axis=0), color="k", linewidth=1)
    waveform_ax.set_xlabel("Time (ms)", fontsize=7, labelpad=1)
    waveform_ax.set_ylabel("Amplitude", fontsize=7, labelpad=1)
    waveform_ax.tick_params(labelsize=6, pad=1)
    statistics.boundary_ticks(waveform_ax, xprec=1)

    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    fig.canvas.draw()

    # ── Panel labels ──────────────────────────────────────────────────────────
    # A-C: place at a shared figure-level y so they sit on the same horizontal
    # line despite the row's axes having independently-fit heights.
    _row0 = [
        (cartoon_ax, "A", -0.15),
        (mag_raw_ax, "B", -0.05),
        (wn_raw_ax,  "C", -0.05),
    ]
    _label_y = max(ax.get_position().y1 for ax, _, _ in _row0) + 0.01
    for _ax, _lbl, _xoff in _row0:
        _pos = _ax.get_position()
        fig.text(_pos.x0 + _xoff * _pos.width, _label_y, _lbl,
                 fontfamily="arial", fontsize=11, weight="bold")

    mag_spectra_ax.annotate("D", xy=(-0.15, 1.1), xycoords="axes fraction", fontfamily="arial", fontsize=11, weight="bold")
    mag_dist_ax.annotate("E", xy=(-0.15, 1.1), xycoords="axes fraction", fontfamily="arial", fontsize=11, weight="bold")
    ecdf_ax.annotate("F", xy=(-0.03, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=11, weight="bold")

    wn_spectra_ax.annotate("G", xy=(-0.15, 1.1), xycoords="axes fraction", fontfamily="arial", fontsize=11, weight="bold")
    wn_dist_ax.annotate("H", xy=(-0.15, 1.1), xycoords="axes fraction", fontfamily="arial", fontsize=11, weight="bold")
    out_path = out_dir / "Fig1_auditory.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=FP.DPI)
    print(f"Saved {out_path}")
    if not in_notebook:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 1 (composite NPIX null + positive + ECDFs)")
    parser.add_argument("--out-dir", default=FP.OUT_DIR, help="Output directory for PDFs")
    parser.add_argument("--data-dir", default=FP.DATA_DIR, help="Directory containing pipeline output pickles")
    args = parser.parse_args([] if in_notebook else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading NPIX data...")
    modulation_df, fourier_df = load_data(args.data_dir)
    plot_fig1_composite(modulation_df, fourier_df, out_dir)


if __name__ == "__main__":
    if in_notebook:
        # Programmatic equivalent of the `%config` IPython magic -- same
        # effect (retina figure format for inline display), but doesn't
        # require IPython's magic-command preprocessing, so this file stays
        # runnable via plain `python fig1.py` too, not just `%run` inside a
        # live IPython/Jupyter session.
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    main()
