"""Fig 3 — p-value and q-value uniformity across neuron populations.

Rows 0-1: negative-result (magnet-experiment) population, expected to be
uniform under the null. Rows 2-3: positive-control population (audio/visual/
oddball/WN/3D stimulus presentations with a real, expected evoked effect),
shown for contrast — non-uniform, left-skewed p-values.

Requires:
  data/manuscript/all_fourier_df.parquet  (run python pipeline/aggregate.py first)

Fig3.pdf shows each neuron once (its recording closest to 3 Hz, see
`one_recording_per_neuron`); Fig3_by_modality.pdf splits its positive
control into white noise and every visual stimulus (spikes and imaging); Fig3S2_all_recordings.pdf
shows every recording, split into occurrence waves (page 1) and pooled
(page 2) -- see plot_fig3's `buckets`.

Usage:
    python pipeline/manuscript/fig3.py
    python pipeline/manuscript/fig3.py --out-dir figs/paper
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from magpyneto2 import statistics

import format_parameters as FP

# Sorted-q-value axes range, and the FDR threshold drawn on it (dashed line,
# plus the responder count where each panel's curve crosses it). q-values
# below Q_YLIM[0] (including the float64-underflow zeros of the most extreme
# NFCs) fall off the bottom of the axes; the responder count still includes
# them.
Q_YLIM = (1e-4, 1)
FDR = 0.05

# Fig3.pdf's single-contribution selector: each neuron contributes the one
# recording whose stimulus frequency is closest to this (see
# `one_recording_per_neuron`).
TARGET_FREQ = 3.0

# Positive-control ties (the 8 grating orientations of one pigeon recording,
# repeat trials) go to a pseudo-random recording rather than the first one,
# which was always orientation 0 (2026-11-02). The choice hashes each row's
# neuron key and rec name with this seed, so a neuron gets the same recording
# in every panel and in Fig 2.
POS_CONTROL_TIE_SEED = 0


def cmap_to_black(cmap_name, floor=0.25):
    """`cmap_name`'s light tint -> saturated hue -> pure black, so the darkest
    shade in every Fig 3-family plot is black for maximum dynamic range. Shared
    by fig3s4_bucketings / fig3s3_reproducibility so every Fig 3 PDF uses the same ramp."""
    base = matplotlib.colormaps[cmap_name]
    return LinearSegmentedColormap.from_list(
        f"{cmap_name}_to_black", [base(floor), base(0.7), "black"])


def shades_to_black(cmap_name, n):
    """`n` discrete shades from `cmap_to_black`, lightest first; a lone shade
    is black."""
    if n == 0:
        return []
    ramp = cmap_to_black(cmap_name)
    return list(ramp(np.linspace(0.0, 1.0, n) if n > 1 else [1.0]))


def _split_into_waves(df, wave_groups=("species", "ID", "date", "id")):
    """Split a dataframe into a list of "wave" dataframes -- one per occurrence
    number of repeated rows for the same neuron (same `wave_groups` key). Wave 0
    holds each neuron's first row, wave 1 its second, etc.
    """
    df = df.copy()
    df["occurrence"] = df.groupby(list(wave_groups)).cumcount()
    return [g.reset_index(drop=True) for _, g in df.groupby("occurrence")]


def one_recording_per_neuron(waves, pval_col="p_value", target_freq=TARGET_FREQ,
                             wave_groups=("species", "ID", "date", "id"), tie_seed=None):
    """One row per neuron: the recording whose `freq` is closest to
    `target_freq`, among the neuron's rows with a finite `pval_col` (so a
    neuron isn't lost to a NaN 2F p-value when another recording has one).
    Ties -- several recordings at the closest frequency, e.g. repeat trials --
    go to the earliest occurrence, i.e. the row that `_split_into_waves` puts
    in wave 0 when it is among them -- or, given `tie_seed`, to a pseudo-random
    one of them (a hash of the row's neuron key and rec name, so the choice
    doesn't depend on which other rows are in `waves`). The choice never looks
    at p-values.

    Neurons recorded at one frequency only (all the fish, quail) keep their
    first recording; for the fish, recorded below 1 Hz, "closest to 3 Hz" is
    their highest frequency.
    """
    df = pd.concat(waves, ignore_index=True)  # wave order = occurrence order
    df = df.loc[np.isfinite(pd.to_numeric(df[pval_col], errors="coerce"))]
    dist = (df["freq"] - target_freq).abs().to_numpy()
    if tie_seed is None:
        df = df.iloc[np.argsort(dist, kind="stable")]
    else:
        tie = pd.util.hash_pandas_object(df[list(wave_groups) + ["rec"]], index=False,
                                         hash_key=f"{tie_seed:016d}").to_numpy()
        df = df.iloc[np.lexsort((tie, dist))]
    return df.drop_duplicates(subset=list(wave_groups), keep="first").reset_index(drop=True)


def is_visual(df):
    """Rows of the positive control that are visual stimuli, spikes and
    imaging alike: everything but the white noise (`*WN*` recs) -- the
    pigeon hippocampus 2 Hz and 3 Hz gratings, the pigeon cerebellum oddball
    light flashes (standard trials), the zebra finch wulst bars and the
    zebrafish/medaka 1/60 Hz gratings. The "Vis" curve of the by-modality
    figures (2026-11-18; 3 Hz gratings only until then)."""
    return ~df.rec.str.contains("WN")


def split_into_occurrence_waves(all_fourier_df, pval_col="p_value"):
    """`pval_col`: which column to treat as the p-value for the Storey q-value
    computation -- "p_value" (1F, default) or "2f_p_value" (see fig3s1_second_harmonic.py,
    which reuses this same function for the 2F-harmonic version of this
    figure). Occurrence-wave splitting itself (which row is a given neuron's
    1st/2nd/... magnetic trial) is structural and doesn't depend on this
    choice, only the q-value computation within each wave does.
    """
    all_neg_res, all_pos_control, all_unique_pos_control = \
        statistics.get_poscontrols_negresults(all_fourier_df)

    waves = _split_into_waves(all_neg_res)
    pos_control_waves = _split_into_waves(all_pos_control)

    wave_df_l = []
    for wave_df in waves:
        pval = wave_df[pval_col].values
        qval, pi0 = statistics.storey_qvalues(pval, lambda_=0.5)
        wave_df = wave_df.copy()
        wave_df["pval"] = pval
        wave_df["qval"] = qval
        wave_df["pi0"] = pi0
        wave_df_l.append(wave_df)

    return waves, wave_df_l, pos_control_waves


def draw_fdr_threshold(ax, sorted_qvals=None, color=None, fdr=FDR, label=None, row=0, label_x=None):
    """Dashed line at q = `fdr` on a sorted-q-value axes; given the panel's
    one sorted q-value curve, also a faint grey vertical line (behind the
    data) from the x-axis up to where that curve crosses it, labelled at its
    foot in `color` with the responder count (units at
    q <= `fdr`). The crossing lies between the last unit at or below `fdr`
    (x = n - 1) and the first one above it (x = n). With no responders only
    the label is drawn. Call after the axes' y-limits are final.

    `label` prefixes the count (e.g. the curve's bucket name); `row` stacks
    the labels of several curves on one axes, one text line per row;
    `label_x`, if given, is the x (data units) the label is left-justified
    at instead of this curve's own crossing -- the caller passes the
    rightmost crossing of all the axes' curves, so stacked labels share one
    left edge clear of every curve's data (2026-11-12)."""
    if row == 0:
        ax.axhline(fdr, color=FP.COLOR_NULL, linestyle="--", linewidth=FP.LW_REFERENCE)
    if sorted_qvals is None:
        return
    n = int(np.sum(sorted_qvals <= fdr))
    x_cross = n - 0.5
    if n > 0:
        # Behind the data (lines default to zorder 2), grey and faint: it
        # marks the crossing, it isn't data.
        ax.plot([x_cross, x_cross], [ax.get_ylim()[0], fdr], color=FP.COLOR_NULL,
                alpha=FP.ALPHA_TRACE, linewidth=FP.LW_REFERENCE, zorder=1)
    # At the foot of the line, on the x-axis, rather than at the crossing,
    # where it sat on top of the curve.
    # The q threshold is stated in the legend, not on every panel (2026-11-09).
    text = f"r = {n}" if label is None else f"{label}: r = {n}"
    fontsize = matplotlib.font_manager.FontProperties(
        size=matplotlib.rcParams["xtick.labelsize"]).get_size_in_points()
    if label_x is None:
        label_x = x_cross
    return ax.annotate(text, (max(label_x, 0), ax.get_ylim()[0]), xytext=(3, 3 + 1.3 * fontsize * row),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=matplotlib.rcParams["xtick.labelsize"], color=color)


def plot_uniform_p(waves, axes, percentile=None, colors=None, pval_col="p_value", sens_col="sens",
                    q_ylim=Q_YLIM, fdr=FDR, labels=None):
    """`colors`, if given, must have one entry per entry of `waves` (same order,
    before the internal reversal) -- assigns an explicit color per occurrence-wave
    instead of relying on matplotlib's automatic per-Axes color cycle. This matters
    because occurrence-wave index is NOT comparable across different populations
    (see plot_fig3's neg-result vs. pos-control blocks): without explicit colors,
    two unrelated populations plotted on separate Axes would each restart the same
    default color cycle, so a shared color between them would be pure coincidence
    that could be misread as "the same neurons."

    `pval_col`/`sens_col`: which columns to treat as the p-value / sensitivity
    for this call -- "p_value"/"sens" (1F, default) or "2f_p_value"/"sens_2f"
    (see fig3s1_second_harmonic.py's 2F-harmonic version of this figure).

    `q_ylim`: (bottom, top) of the sorted-q-value axes.

    `fdr`: if not None, the FDR threshold line, plus -- when the panel has a
    single curve -- the responder count where it crosses (see
    `draw_fdr_threshold`). With several curves (occurrence waves, variant
    bucketings) only the threshold line is drawn; one count per bucket would
    bury the panel -- unless `labels` (one per entry of `waves`) is given, in
    which case every curve gets its own count, prefixed by its label.

    `axes[0]` (the sorted-p-value axes) may be None, in which case only the
    sorted-q-value axes is drawn. fig3s4_bucketings.py's ECDF-deviation page
    replaces the sorted-p-value panel with a stack of per-bucket
    ECDF-deviation mini-axes but keeps this exact q-value panel below it, so
    it needs the q half of this function without the p half.
    """
    if colors is None:
        colors = [None] * len(waves)
    count_labels = labels
    if labels is None:
        labels = [None] * len(waves)
    # Widest bucket, which is what the shared x-axis actually autoscales to.
    # This used to track the LAST bucket plotted instead, which happens to be
    # the same thing for occurrence waves (wave 0 is plotted last and is
    # always the largest, since every neuron has a 1st recording) but is wrong
    # for any other bucketing -- fig3s4_bucketings.py buckets by frequency and by
    # brain region, orderings under which the last-plotted bucket is routinely
    # not the widest, and the right-hand tick then landed mid-axes carrying a
    # label that disagreed with the axis limit it was supposed to mark.
    max_n = 0
    wave_sorted_qvals, wave_colors, wave_labels = [], [], []
    for wave_df, color, label in zip(waves[::-1], colors[::-1], labels[::-1]):
        if len(wave_df) == 0:
            continue
        if percentile is not None:
            pct = np.percentile(wave_df[sens_col], percentile)
            wave_df = wave_df.loc[wave_df[sens_col] > pct]
        if len(wave_df) == 0:
            continue
        # Units with no p-value (the owl's precomputed rows; for 2F, recordings
        # with no paired 2F fit) have no place on a sorted p/q axis, and would
        # otherwise stretch the "Unit" axis past the last plotted point.
        pval = wave_df[pval_col].values
        pval = pval[np.isfinite(pval)]
        if len(pval) == 0:
            continue
        qval, pi0 = statistics.storey_qvalues(pval, lambda_=0.5)
        max_n = max(max_n, len(qval))
        sq = np.sort(qval)
        wave_sorted_qvals.append(sq)
        wave_colors.append(color)
        wave_labels.append(label)

        if axes[0] is not None:
            axes[0].plot(np.sort(pval), ".", color=color, alpha=FP.ALPHA_TRACE, markersize=FP.MS_DATA, markeredgewidth=0, rasterized=True)
        axes[1].plot(sq, ".", color=color, alpha=FP.ALPHA_TRACE, markersize=FP.MS_DATA, markeredgewidth=0, rasterized=True)

    if axes[0] is not None:
        statistics.nestle_labels(axes[0], x_offset=-0.05, y=False)
    statistics.nestle_labels(axes[1], x_offset=-0.05, y=False)
    if max_n > 0:
        for ax in axes:
            if ax is None:
                continue
            ax.set_xticks([0, max_n])
            ax.set_xticklabels([0, max_n])
    axes[1].set_yscale("log")
    axes[1].set_ylim(*q_ylim)
    if fdr is not None:
        if count_labels is not None:
            # Back in the caller's order, the first curve on the bottom line.
            # Every label left-justified at the rightmost crossing, so no
            # label sits on top of another curve's data (2026-11-12).
            label_x = max(int(np.sum(sq <= fdr)) - 0.5 for sq in wave_sorted_qvals)
            anns = [draw_fdr_threshold(axes[1], sq, color, fdr=fdr, label=label, row=row,
                                       label_x=label_x)
                    for row, (sq, color, label) in enumerate(zip(wave_sorted_qvals[::-1], wave_colors[::-1],
                                                                 wave_labels[::-1]))]
            # A crossing far to the right would push the labels past the
            # axes' right edge (into the next panel's y-label): slide them
            # all left just enough to end at the edge (2026-11-18).
            fig = axes[1].figure
            renderer = fig.canvas.get_renderer()
            overflow_px = max(a.get_window_extent(renderer).x1 for a in anns) \
                - axes[1].get_window_extent(renderer).x1
            if overflow_px > 0:
                for a in anns:
                    a.xyann = (a.xyann[0] - overflow_px * 72 / fig.dpi, a.xyann[1])
        elif len(wave_sorted_qvals) == 1:
            draw_fdr_threshold(axes[1], wave_sorted_qvals[0], wave_colors[0], fdr=fdr)
        else:
            draw_fdr_threshold(axes[1], fdr=fdr)


def plot_fig3(all_fourier_df, out_dir: Path, pval_col="p_value", sens_col="sens",
              out_name="Fig3.pdf", suptitle=None, buckets="one_per_neuron", area="HP"):
    """`pval_col`/`sens_col`/`out_name`/`suptitle`: overridden by fig3s1_second_harmonic.py to
    reuse this exact function for the 2F-harmonic version of this figure
    ("2f_p_value"/"sens_2f"/"Fig3S1_second_harmonic.pdf").

    `buckets`: which recordings each panel shows (2026-10-18).
      "one_per_neuron" -- Fig3.pdf: each neuron counted once, by its
          recording closest to TARGET_FREQ (`one_recording_per_neuron`,
          2026-10-27; until then its first recording, i.e. the largest
          occurrence wave).
      "waves"   -- every occurrence wave, one shade each (Fig3.pdf before
          2026-10-18; now page 1 of Fig3S2_all_recordings.pdf).
      "pooled"  -- every recording of every neuron in one bucket, each NFC
          treated as an independent sample.
      "pooled_modality" -- page 2 of Fig3S2_all_recordings.pdf (2026-11-18):
          as "pooled", but the positive control split into white noise and
          visual like "modality" (every recording, no per-neuron pick).
      "modality" -- Fig3_by_modality.pdf (2026-11-02): as "one_per_neuron",
          but the positive control split into two curves, white noise
          (auditory) and every visual stimulus (oddball included),
          spikes and imaging alike (`is_visual`; 3 Hz gratings only until 2026-11-18), each neuron
          once per curve.
      "wn_oddball" -- Fig3_WN_oddball.pdf (2026-11-02): as "modality", with
          the oddball recordings (light flashes, standard trials only) in
          place of visual.
    In "one_per_neuron", "modality" and "wn_oddball" positive-control ties go to a
    pseudo-random recording (`POS_CONTROL_TIE_SEED`).

    `area`: the pigeon brain area of panels C and D, "HP" (hippocampus) or
    "CB" (cerebellum; Fig3_by_modality_CB.pdf, 2026-11-02).
    """
    area_name = {"HP": "hippocampus", "CB": "cerebellum"}[area]
    waves, wave_df_l, pos_control_waves = split_into_occurrence_waves(all_fourier_df, pval_col=pval_col)

    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY}
    matplotlib.rc("font", **font)

    if buckets == "waves":
        # Explicit, non-overlapping color families per population -- occurrence-wave
        # index means "Nth magnet trial" for neg-result vs. "Nth stimulus condition"
        # for pos-control, so the two are not comparable and must never share a color
        # (see plot_uniform_p's docstring). Shade still varies by occurrence within
        # each population, echoing FP.COLOR_MAG ("steelblue") / FP.COLOR_VIS ("coral").
        neg_colors = shades_to_black("Blues", len(waves))
        pos_colors = shades_to_black("Oranges", len(pos_control_waves))
    elif buckets == "pooled":
        waves = [pd.concat(waves, ignore_index=True)]
        pos_control_waves = [pd.concat(pos_control_waves, ignore_index=True)]
        neg_colors, pos_colors = [FP.COLOR_MAG], [FP.COLOR_VIS]
    elif buckets == "pooled_modality":
        waves = [pd.concat(waves, ignore_index=True)]
        pos_control_waves = [pd.concat(pos_control_waves, ignore_index=True)]
        neg_colors, pos_colors = [FP.COLOR_MAG], [FP.COLOR_AUD, FP.COLOR_VIS]
    elif buckets == "one_per_neuron":
        neg_colors, pos_colors = [FP.COLOR_MAG], [FP.COLOR_VIS]
    elif buckets == "modality":
        neg_colors, pos_colors = [FP.COLOR_MAG], [FP.COLOR_AUD, FP.COLOR_VIS]
    elif buckets == "wn_oddball":
        neg_colors, pos_colors = [FP.COLOR_MAG], [FP.COLOR_AUD, FP.COLOR_ODDBALL]
    else:
        raise ValueError(f"unknown buckets={buckets!r}")

    # Each of the 4 conditions gets its own 2x2 quadrant: columns = negative-
    # result ("Magnetic") vs. positive-control ("Visual/Audio"), rows = p-values
    # vs. q-values. y is shared within a quadrant's row, so magnetic and
    # visual/audio q-values (or p-values) land on the same scale and are directly
    # comparable; x is shared within a quadrant's column, since the p- and
    # q-value plots for a given population come from the same neuron subset.
    # Sharing is scoped to each quadrant only -- the 4 conditions have very
    # different neuron counts and shouldn't force each other's axis ranges
    # (same reasoning as the neg-result/pos-control x-sharing fix above).
    conditions = [
        ("All neurons across all species", None, None),
        ("All pigeon neurons", lambda df: df.species == "Pigeon", None),
        (f"Pigeon {area_name} neurons", lambda df: (df.area == area) & (df.species == "Pigeon"), None),
        (f"Top 10% most sensitive pigeon {area} neurons",
         lambda df: (df.area == area) & (df.species == "Pigeon"), 90),
    ]
    quadrant_slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    panel_letters = "ABCD"

    fig = plt.figure(figsize=FP.FIGSIZE_FIG3_MAIN)
    # Spacing re-tuned 2026-11-09 for the 7.5 in shared main-figure width
    # (was 11.2 in): wider inner gaps so neighbouring axes' end ticks
    # ("11709" | "0") and the p/q panels' end labels don't touch, more room
    # between quadrant rows for the quadrant titles.
    outer = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.55,
                              left=0.1, right=0.98, top=0.88, bottom=0.07)

    for (title, cond_filter, percentile), (orow, ocol), letter in \
            zip(conditions, quadrant_slots, panel_letters):
        inner = outer[orow, ocol].subgridspec(2, 2, wspace=0.25, hspace=0.2)
        ax_p_neg = fig.add_subplot(inner[0, 0])
        ax_p_pos = fig.add_subplot(inner[0, 1])
        ax_q_neg = fig.add_subplot(inner[1, 0])
        ax_q_pos = fig.add_subplot(inner[1, 1])

        ax_p_neg.sharey(ax_p_pos)
        ax_q_neg.sharey(ax_q_pos)
        ax_p_neg.sharex(ax_q_neg)
        ax_p_pos.sharex(ax_q_pos)
        ax_p_neg.tick_params(labelbottom=False)
        ax_p_pos.tick_params(labelbottom=False, labelleft=False)
        ax_q_pos.tick_params(labelleft=False)
        for ax in (ax_p_neg, ax_p_pos, ax_q_neg, ax_q_pos):
            ax.spines[["top", "right"]].set_visible(False)

        neg_subset = waves if cond_filter is None else [w.loc[cond_filter(w)] for w in waves]
        pos_subset = pos_control_waves if cond_filter is None else \
            [w.loc[cond_filter(w)] for w in pos_control_waves]
        pos_labels = None
        if buckets in ("one_per_neuron", "modality", "wn_oddball"):
            neg_subset = [one_recording_per_neuron(neg_subset, pval_col=pval_col)]
        if buckets == "one_per_neuron":
            pos_subset = [one_recording_per_neuron(pos_subset, pval_col=pval_col,
                                                   tie_seed=POS_CONTROL_TIE_SEED)]
        elif buckets in ("modality", "wn_oddball"):
            wn = [w.loc[w.rec.str.contains("WN")] for w in pos_subset]
            if buckets == "modality":
                other = [w.loc[is_visual(w)] for w in pos_subset]
                pos_labels = ["Aud", "Vis"]  # was ["WN", "3 Hz"] (2026-11-10)
            else:
                other = [w.loc[w.rec.str.contains("oddball")] for w in pos_subset]
                pos_labels = ["WN", "oddball"]
            pos_subset = [one_recording_per_neuron(b, pval_col=pval_col, tie_seed=POS_CONTROL_TIE_SEED)
                          for b in (wn, other)]
        elif buckets == "pooled_modality":
            # Already one pooled bucket; split it, keeping every recording.
            pooled = pos_subset[0]
            pos_subset = [pooled.loc[pooled.rec.str.contains("WN")],
                          pooled.loc[is_visual(pooled)]]
            pos_labels = ["Aud", "Vis"]

        plot_uniform_p(neg_subset, [ax_p_neg, ax_q_neg], percentile=percentile, colors=neg_colors,
                       pval_col=pval_col, sens_col=sens_col)
        plot_uniform_p(pos_subset, [ax_p_pos, ax_q_pos], percentile=percentile, colors=pos_colors,
                       pval_col=pval_col, sens_col=sens_col, labels=pos_labels)

        ax_p_neg.set_title("Magnetic", fontsize=FP.FS_TITLE)
        if buckets in ("modality", "pooled_modality"):
            ax_p_pos.set_title("Auditory / Visual", fontsize=FP.FS_TITLE)
        elif buckets == "wn_oddball":
            ax_p_pos.set_title("Auditory (WN) / visual (oddball flashes)", fontsize=FP.FS_TITLE)
        else:
            ax_p_pos.set_title("Visual/Audio", fontsize=FP.FS_TITLE)
        ax_p_neg.set_ylabel("Sorted p-values")
        ax_q_neg.set_ylabel("Sorted q-values")
        ax_q_neg.set_xlabel("Unit")
        ax_q_pos.set_xlabel("Unit")

        quad_pos = outer[orow, ocol].get_position(fig)
        # Quadrant title a fixed 0.35 in above the quadrant, clearing the
        # axis titles ("Magnetic", ...) below it at any figure height; the
        # letter sits out in the y-label margin so a long title can't run
        # into it.
        title_y = quad_pos.y1 + 0.35 / fig.get_figheight()
        fig.text((quad_pos.x0 + quad_pos.x1) / 2, title_y, title,
                  ha="center", va="bottom", fontsize=FP.FS_BODY + 1)
        fig.text(quad_pos.x0 - 0.07, title_y, letter,
                  ha="left", va="bottom", fontfamily="arial", fontsize=12)

    if suptitle:
        # Above the quadrant titles (which sit at top + 0.04 = 0.94); at the
        # old y=0.98 it ran into the A/B titles.
        fig.suptitle(suptitle, fontsize=FP.FS_BODY + 2, fontweight="bold", y=1.0, va="bottom")

    if out_name is None:
        # Caller wants the Figure itself, not a file -- fig3s4_bucketings.py
        # renders page 1 of its multipage PDF through this exact function so
        # that page can't drift from the canonical figure, and writes it to
        # its own PdfPages instead of a standalone Fig3.pdf.
        return fig

    out_path = out_dir / out_name
    FP.save_main_figure(fig, out_path)
    print(f"Saved {out_path}")
    if not in_notebook:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 3 (p/q-value uniformity)")
    parser.add_argument("--out-dir", default=FP.OUT_DIR, help="Output directory for PDFs")
    parser.add_argument("--parquet", default=FP.PARQUET_PATH,
                        help=f"Path to all_fourier_df.parquet (default: {FP.PARQUET_PATH})")
    args = parser.parse_args([] if in_notebook else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.parquet} ...")
    all_fourier_df = pd.read_parquet(args.parquet)
    plot_fig3(all_fourier_df, out_dir)
    plot_fig3(all_fourier_df, out_dir, out_name="Fig3_by_modality.pdf", buckets="modality")
    plot_fig3(all_fourier_df, out_dir, out_name="Fig3_by_modality_CB.pdf", buckets="modality", area="CB")
    # White noise and oddball, panels C/D hippocampus (page 1) and cerebellum (page 2).
    wn_oddball_path = out_dir / "Fig3_WN_oddball.pdf"
    with PdfPages(wn_oddball_path) as pdf:
        for area in ("HP", "CB"):
            fig = plot_fig3(all_fourier_df, out_dir, out_name=None, buckets="wn_oddball", area=area)
            pdf.savefig(fig, bbox_inches="tight", dpi=FP.DPI)
            plt.close(fig)
    print(f"Saved {wn_oddball_path}")

    # Supplement: the occurrence-wave version (Fig3.pdf before 2026-10-18) and
    # the all-recordings-pooled version (positive control split into
    # auditory / visual since 2026-11-18), one page each.
    supp_dir = Path(FP.SUPPLEMENT_DIR)
    supp_dir.mkdir(parents=True, exist_ok=True)
    supp_path = supp_dir / "Fig3S2_all_recordings.pdf"
    with PdfPages(supp_path) as pdf:
        for buckets, suptitle in (
                ("waves", "Each neuron's 1st, 2nd, ... recording as a separate bucket"),
                ("pooled_modality", "Every recording of every neuron pooled (each NFC an independent sample)")):
            fig = plot_fig3(all_fourier_df, out_dir, out_name=None, suptitle=suptitle, buckets=buckets)
            pdf.savefig(fig, bbox_inches="tight", dpi=FP.DPI)
            plt.close(fig)
    print(f"Saved {supp_path}")


if __name__ == "__main__":
    if in_notebook:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    main()
