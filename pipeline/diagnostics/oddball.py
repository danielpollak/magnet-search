"""
Oddball-paradigm-specific diagnostics.

plot_oddball_raw_ttl — raw aux/photodiode TTL trace, short-window overview +
single-pulse zoom, with Schmitt-trigger thresholds and detected up/down
crossings marked (down-crossings colour-coded by which _handle_oddball
category they ended up in: standard, long_on, long_off, long_both, or
dropped [ambiguous on either axis]).

plot_oddball_trial_diagnostics — stimulus/baseline spans (against the real,
padded, onset-anchored windows that feed the analysis), dropped-trial
markers, and the spike raster together, across all four categories.

Why this exists: _handle_oddball (pipeline/paradigms/openephys_multistim.py)
classifies each candidate trial by its own on-duration (stimulus pulse
width) and off-duration (preceding silence) SEPARATELY -- confirmed on real
data that the combined down-to-down gap conflates two distinct
manipulations (a long stimulus-ON period vs. a long stimulus-OFF/silence
period before an otherwise-normal pulse), plus a rare third case where both
are elevated at once. See MagnetSearch/code/notebooks/
20230221_concatenated.ipynb's ball_d/oddball_d split, which the original
single-gap version of this ported -- this generalizes it along the axis the
notebook never separated. Neither this plot nor _handle_oddball uses pulse
shape/amplitude to classify -- that's carried entirely by on/off timing, not
by anything visible in a single pulse's own waveform. The raw-TTL plot's
zoom page is a sanity check on that absence, comparing pulse shapes directly
against the raw signal in case a duty-cycle/amplitude signature turns out to
exist after all.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# category codes/hues shared by both plots in this module (see _handle_oddball)
_DROPPED = 0
_CATEGORY_CODE = {"standard": 1, "long_on": 2, "long_off": 3, "long_both": 4}
_CATEGORY_HUE = {
    "standard": "teal",
    "long_on": "crimson",
    "long_off": "darkorange",
    "long_both": "mediumorchid",
}
_CATEGORY_STYLE = {
    _CATEGORY_CODE[cat]: dict(color=hue, label=f"down-crossing ({cat})")
    for cat, hue in _CATEGORY_HUE.items()
}
_CATEGORY_STYLE[_DROPPED] = dict(
    color="lightgray", edgecolor="dimgray", linewidth=0.5,
    label="down-crossing (dropped: ambiguous on/off)")


def plot_oddball_raw_ttl(cfg, recname, viz_trace, fs, thr_on, thr_off,
                          iup, idown, category, save_dir,
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
    category   : 1-D int array, same length as idown -- 0 (dropped) or one
                 of _CATEGORY_CODE's values per down-crossing (see
                 _handle_oddball). A crossing ends one candidate trial AND
                 starts the next, so it can only carry one label here --
                 this is a rough visual aid, not authoritative.
    save_dir   : Path-like, destination directory (must already exist)
    overview_s : float -- length of the first (overview) window, seconds
    zoom_pulses: int -- number of retained (non-dropped) pulses to span in
                 the zoom page
    """
    save_dir = Path(save_dir)
    t = np.arange(len(viz_trace)) / fs
    pdf_path = save_dir / f"{cfg.name}_oddball_ttl.pdf"
    category = np.asarray(category)

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
        category_in = category[in_window]

        ax.scatter(iup_in / fs, np.full(len(iup_in), thr_on), marker="^",
                   color="green", s=18, zorder=5, label="up-crossing")
        for code, style in _CATEGORY_STYLE.items():
            sel = category_in == code
            if sel.any():
                ax.scatter(idown_in[sel] / fs, np.full(sel.sum(), thr_off),
                           marker="v", s=22, zorder=5, **style)

        ax.set_xlabel("Time (s), native aux-channel clock")
        ax.set_ylabel("Aux channel signal (a.u.)")
        ax.set_title(
            f"{cfg.name}  |  {recname}\nraw TTL overview, first {overview_s:g}s "
            f"— green=up; down: teal=standard, crimson=long_on, "
            f"orange=long_off, purple=long_both, gray=dropped", fontsize=9)
        ax.legend(fontsize=6, loc="upper right", ncol=2)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: zoom on the first few retained (non-dropped) pulses --
        # close enough to compare individual pulse widths/shapes across
        # categories, since that (not timing alone) is the only place a
        # standard-vs-long-on/off signature could possibly show up in this
        # single-channel TTL trace.
        retained = idown[category != _DROPPED]
        if len(retained) >= 2:
            span_end_idx = min(zoom_pulses, len(retained) - 1)
            zoom_start = max(0, int(retained[0] - 0.2 * fs))
            zoom_end = min(len(viz_trace), int(retained[span_end_idx] + 0.2 * fs))
            fig2, ax2 = plt.subplots(figsize=(11, 4))
            ax2.plot(t[zoom_start:zoom_end], viz_trace[zoom_start:zoom_end],
                     color="black", lw=1.0, rasterized=True)
            ax2.axhline(thr_on, color="green", ls="--", lw=0.8)
            ax2.axhline(thr_off, color="firebrick", ls="--", lw=0.8)
            zoom_up = iup[(iup >= zoom_start) & (iup < zoom_end)]
            zoom_down_mask = (idown >= zoom_start) & (idown < zoom_end)
            ax2.scatter(zoom_up / fs, np.full(len(zoom_up), thr_on), marker="^",
                        color="green", s=30, zorder=5)
            for code, style in _CATEGORY_STYLE.items():
                sel = zoom_down_mask & (category == code)
                if sel.any():
                    ax2.scatter(idown[sel] / fs, np.full(sel.sum(), thr_off),
                                marker="v", s=30, zorder=5,
                                color=style["color"])
            ax2.set_xlabel("Time (s), native aux-channel clock")
            ax2.set_ylabel("Aux channel signal (a.u.)")
            ax2.set_title(
                f"{cfg.name}  |  {recname}\nzoomed pulse shape, first "
                f"{zoom_pulses} retained pulses — compare "
                f"widths/shapes for a possible category signature",
                fontsize=9)
            pdf.savefig(fig2)
            plt.close(fig2)

    print(f"  [oddball] raw TTL diagnostic -> {pdf_path}")


_TRIAL_STYLE = {
    cat: dict(hue=hue, stim_alpha=0.38, baseline_alpha=0.12)
    for cat, hue in _CATEGORY_HUE.items()
}


def _trial_spans(intervals, down_idx, windows_by_di, t_up):
    """Per-trial (baseline_pre, stim_on, baseline_post) time spans, in the
    same (recording-local) sample domain as t_up/t_down.

    stim_on is this trial's own pulse (onset -> offset, i.e. t_up[down_i] ->
    curr). baseline_pre/baseline_post are what padding actually bought --
    the padded window's start up to onset, and offset up to the padded
    window's end -- ANCHORED AT ONSET, matching _pad_oddball_windows (not at
    the padded window's own start, which is only the spike-inclusion bound,
    not the phase reference).
    """
    spans = []
    for (curr, prev), di in zip(intervals, down_idx):
        onset = t_up[di]
        w_start, w_stop = windows_by_di[di]
        spans.append(((w_start, onset), (onset, curr), (curr, w_stop)))
    return spans


def plot_oddball_trial_diagnostics(cfg, recname, st_d, t_up, t_down, ap_sr,
                                    buckets, windows_by_di, dropped_down_idx,
                                    save_dir, zoom_trials=10):
    """Save a 2-page PDF combining, for one oddball aux recording, everything
    needed to sanity-check _handle_oddball's trial construction directly
    against the real spike data:

      1) where each stimulus pulse actually is  -- solid shading, onset->offset
      2) where each trial's baseline sits        -- light shading, the padded
                                                      window minus the pulse itself
      3) which candidate trials were dropped      -- sparse gray ticks (ambiguous
                                                      on either on/off axis)
      4) a spike raster across every unit          -- eventplot, rasterized
      5) standard / long_on / long_off / long_both -- teal / crimson / orange /
                                                      purple, matching
                                                      diagnostics/processing.py's
                                                      _KIND_STYLE

    Parameters
    ----------
    cfg, recname : as elsewhere
    st_d      : {unit_id: spike_samples} -- recording-local sample domain,
                same as t_up/t_down (pre get_MM_offset; see _handle_oddball)
    t_up, t_down : shifttime()'d crossing sample arrays, recording-local
    ap_sr     : sampling rate
    buckets   : {"standard"|"long_on"|"long_off"|"long_both": (intervals,
                down_indices)}, straight from _classify_oddball_trials
    windows_by_di : {down_i: (padded_start, padded_stop)}, straight from
                _pad_oddball_windows -- the actual spike-inclusion bounds
                used by the analysis; keyed by down_i so it covers every
                category (and the dropped ones) uniformly
    dropped_down_idx : down_i list from _classify_oddball_trials -- ambiguous
                on either axis, excluded from every category
    save_dir  : Path-like, destination directory (must already exist)
    zoom_trials : number of consecutive candidate trials (by down_i) to
                show on the zoom page
    """
    save_dir = Path(save_dir)
    pdf_path = save_dir / f"{cfg.name}_oddball_trials.pdf"

    unit_ids = sorted(st_d.keys())
    spike_trains_s = [np.asarray(st_d[u]) / ap_sr for u in unit_ids]
    n_units = len(unit_ids)

    dropped_mid_s = [(t_down[di - 1] + t_down[di]) / 2 / ap_sr for di in dropped_down_idx]

    def _draw(ax, di_range=None):
        """di_range: optional (lo, hi) down_i bound to restrict which trials
        get drawn (for the zoom page) -- None draws everything."""
        for cat, (intervals, down_idx) in buckets.items():
            if not down_idx:
                continue
            style = _TRIAL_STYLE[cat]
            for (base_pre, stim_on, base_post), di in zip(
                    _trial_spans(intervals, down_idx, windows_by_di, t_up), down_idx):
                if di_range is not None and not (di_range[0] <= di <= di_range[1]):
                    continue
                for span in (base_pre, base_post):
                    ax.axvspan(span[0] / ap_sr, span[1] / ap_sr,
                               facecolor=style["hue"], edgecolor="none",
                               alpha=style["baseline_alpha"], zorder=0)
                ax.axvspan(stim_on[0] / ap_sr, stim_on[1] / ap_sr,
                           facecolor=style["hue"], edgecolor="none",
                           alpha=style["stim_alpha"], zorder=0)

        if di_range is None:
            for mid_s in dropped_mid_s:
                ax.plot(mid_s, n_units + 0.5, marker="x", color="dimgray",
                        markersize=5, zorder=5)
        # di_range != None (zoom page): the caller draws its own dropped
        # markers separately, filtered to (lo, hi), instead of reusing this.

        ax.eventplot(spike_trains_s, lineoffsets=np.arange(n_units), linelengths=0.8,
                     linewidths=0.4, colors=["black"] * n_units, alpha=0.4, zorder=2)
        ax.set_ylabel("unit")
        ax.set_ylim(-1, n_units + 1.5)

    legend_handles = []
    for cat, hue in _CATEGORY_HUE.items():
        legend_handles.append(Patch(facecolor=hue, alpha=0.6, label=f"{cat}: stimulus on"))
        legend_handles.append(Patch(facecolor=hue, alpha=0.15, label=f"{cat}: baseline/padding"))
    legend_handles.append(Line2D([0], [0], marker="x", color="dimgray", linestyle="",
                                  markersize=6, label="dropped candidate trial (ambiguous on/off)"))

    with PdfPages(pdf_path) as pdf:
        # Page 1: full-session overview
        all_window_stops = [w[1] for w in windows_by_di.values()]
        total_dur_s = (max([t_down[-1]] + all_window_stops) / ap_sr) if len(t_down) else 1.0
        fig_w = float(np.clip(total_dur_s / 5, 10, 60))
        fig, ax = plt.subplots(figsize=(fig_w, max(4, n_units * 0.05 + 1)))
        _draw(ax)
        ax.set_xlabel("Time (s), recording-local NPIX clock")
        ax.set_title(f"{cfg.name}  |  {recname}\noddball trial diagnostics -- full session", fontsize=9)
        ax.legend(handles=legend_handles, fontsize=6, loc="upper right", ncol=2)
        for a in ax.collections:
            a.set_rasterized(True)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: zoom on the first zoom_trials candidate trials (by down_i)
        # so individual baseline/stim-on/padding spans are actually visible
        # at pixel resolution, rather than compressed into the full overview.
        all_di_sorted = sorted(windows_by_di.keys())
        if all_di_sorted:
            zoom_di = all_di_sorted[:zoom_trials]
            lo, hi = min(zoom_di), max(zoom_di)
            fig2, ax2 = plt.subplots(figsize=(11, max(4, n_units * 0.05 + 1)))
            _draw(ax2, di_range=(lo, hi))
            for di in dropped_down_idx:
                if lo <= di <= hi:
                    mid_s = (t_down[di - 1] + t_down[di]) / 2 / ap_sr
                    ax2.plot(mid_s, n_units + 0.5, marker="x", color="dimgray",
                              markersize=6, zorder=5)
            zoom_start_s = t_down[lo - 1] / ap_sr
            zoom_stop_s = t_down[hi] / ap_sr
            pad_s = 0.05 * (zoom_stop_s - zoom_start_s)
            ax2.set_xlim(zoom_start_s - pad_s, zoom_stop_s + pad_s)
            ax2.set_xlabel("Time (s), recording-local NPIX clock")
            ax2.set_title(
                f"{cfg.name}  |  {recname}\nzoom: first {len(zoom_di)} candidate trials",
                fontsize=9)
            ax2.legend(handles=legend_handles, fontsize=6, loc="upper right", ncol=2)
            pdf.savefig(fig2)
            plt.close(fig2)

    print(f"  [oddball] trial diagnostics -> {pdf_path}")
