"""
Oddball-paradigm-specific diagnostics.

plot_oddball_raw_ttl — raw aux/photodiode TTL trace, short-window overview +
single-pulse zoom, with Schmitt-trigger thresholds and detected up/down
crossings marked (down-crossings colour-coded by whether they were kept or
dropped by _handle_oddball's max_interval_s filter).

Why this exists: _handle_oddball (pipeline/paradigms/openephys_multistim.py)
-- and the original legacy script it's a faithful port of
(MagnetSearch/code/processing/20230221_processing.py:201-203, whose own
in-line comment reads "I believe this is just looking at all switches in
general") -- does not distinguish a standard tone/flash from a rare
oddball/deviant one anywhere. It only keeps closely-spaced HIGH/LOW
transitions (gap < max_interval_s) and treats every retained transition
identically; there is no duty-cycle or baseline-period concept either. This
plot is a first diagnostic step toward checking, directly against the raw
signal, whether standard vs. deviant pulses are even distinguishable here
(e.g. a different pulse width/shape/amplitude) -- if they are, that could
become the basis for a real oddball/deviant split later; if they aren't
(i.e. the deviant identity was only ever coded through the video/audio
content voltage never resolves), this plot documents that limit concretely
instead of leaving it implicit.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def plot_oddball_raw_ttl(cfg, recname, viz_trace, fs, thr_on, thr_off,
                          iup, idown, kept_mask, save_dir,
                          overview_s=30.0, zoom_pulses=3):
    """Save a 2-page PDF of the raw oddball aux-channel TTL trace.

    Parameters
    ----------
    cfg        : ExperimentConfig
    recname    : str -- aux recording name (titles/filename)
    viz_trace  : 1-D array, raw aux-channel samples, NATIVE aux sample rate
                 (same array/indexing schmitt() was run on -- not shifted to
                 NPIX time)
    fs         : float -- native sampling rate of viz_trace (Hz)
    thr_on, thr_off : float -- Schmitt trigger thresholds
    iup, idown : 1-D int arrays -- up/down crossing sample indices into
                 viz_trace, as returned by schmitt()
    kept_mask  : 1-D bool array, same length as idown -- True for a
                 down-crossing that ended up inside a retained
                 (< max_interval_s) interval pair; False for one dropped for
                 too long a gap to its neighbour (i.e. excluded from the
                 oddball Fourier analysis entirely)
    save_dir   : Path-like, destination directory (must already exist)
    overview_s : float -- length of the first (overview) window, seconds
    zoom_pulses: int -- number of retained pulses to span in the zoom page
    """
    save_dir = Path(save_dir)
    t = np.arange(len(viz_trace)) / fs
    pdf_path = save_dir / f"{cfg.name}_oddball_ttl.pdf"

    with PdfPages(pdf_path) as pdf:
        # Page 1: short overview window
        fig, ax = plt.subplots(figsize=(11, 4))
        n_overview = min(int(overview_s * fs), len(viz_trace))
        ax.plot(t[:n_overview], viz_trace[:n_overview], color="black", lw=0.6,
                rasterized=True)
        ax.axhline(thr_on, color="green", ls="--", lw=0.8, label=f"thr_on={thr_on:g}")
        ax.axhline(thr_off, color="firebrick", ls="--", lw=0.8, label=f"thr_off={thr_off:g}")

        iup_in = iup[iup < n_overview]
        in_window = idown < n_overview
        idown_in = idown[in_window]
        kept_in = (kept_mask[in_window] if kept_mask is not None
                   else np.ones(len(idown_in), dtype=bool))

        ax.scatter(iup_in / fs, np.full(len(iup_in), thr_on), marker="^",
                   color="green", s=18, zorder=5, label="up-crossing")
        if kept_in.any():
            ax.scatter(idown_in[kept_in] / fs, np.full(kept_in.sum(), thr_off),
                       marker="v", color="firebrick", s=22, zorder=5,
                       label="down-crossing (retained)")
        if (~kept_in).any():
            ax.scatter(idown_in[~kept_in] / fs, np.full((~kept_in).sum(), thr_off),
                       marker="v", color="lightgray", edgecolor="dimgray",
                       linewidth=0.5, s=22, zorder=5,
                       label="down-crossing (excluded: gap too long)")

        ax.set_xlabel("Time (s), native aux-channel clock")
        ax.set_ylabel("Aux channel signal (a.u.)")
        ax.set_title(
            f"{cfg.name}  |  {recname}\nraw TTL overview, first {overview_s:g}s "
            f"— green=up, red=down (retained), gray=down (excluded)", fontsize=9)
        ax.legend(fontsize=6, loc="upper right", ncol=2)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: zoom on the first few RETAINED pulses -- close enough to
        # compare individual pulse widths/shapes, since that (not timing
        # alone) is the only place a standard-vs-deviant signature could
        # possibly show up in this single-channel TTL trace.
        kept_samples = idown[kept_mask] if kept_mask is not None else idown
        if len(kept_samples) >= 2:
            span_end_idx = min(zoom_pulses, len(kept_samples) - 1)
            zoom_start = max(0, int(kept_samples[0] - 0.2 * fs))
            zoom_end = min(len(viz_trace), int(kept_samples[span_end_idx] + 0.2 * fs))
            fig2, ax2 = plt.subplots(figsize=(11, 4))
            ax2.plot(t[zoom_start:zoom_end], viz_trace[zoom_start:zoom_end],
                     color="black", lw=1.0, rasterized=True)
            ax2.axhline(thr_on, color="green", ls="--", lw=0.8)
            ax2.axhline(thr_off, color="firebrick", ls="--", lw=0.8)
            zoom_up = iup[(iup >= zoom_start) & (iup < zoom_end)]
            zoom_down = idown[(idown >= zoom_start) & (idown < zoom_end)]
            ax2.scatter(zoom_up / fs, np.full(len(zoom_up), thr_on), marker="^",
                        color="green", s=30, zorder=5)
            ax2.scatter(zoom_down / fs, np.full(len(zoom_down), thr_off), marker="v",
                        color="firebrick", s=30, zorder=5)
            ax2.set_xlabel("Time (s), native aux-channel clock")
            ax2.set_ylabel("Aux channel signal (a.u.)")
            ax2.set_title(
                f"{cfg.name}  |  {recname}\nzoomed pulse shape, first "
                f"{zoom_pulses} retained pulses — compare "
                f"widths/shapes for a possible standard-vs-deviant signature",
                fontsize=9)
            pdf.savefig(fig2)
            plt.close(fig2)

    print(f"  [oddball] raw TTL diagnostic -> {pdf_path}")
