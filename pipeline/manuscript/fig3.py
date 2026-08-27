"""Fig 3 — p-value and q-value uniformity across neuron populations.

Rows 0-1: negative-result (magnet-experiment) population, expected to be
uniform under the null. Rows 2-3: positive-control population (audio/visual/
oddball/WN/3D stimulus presentations with a real, expected evoked effect),
shown for contrast — non-uniform, left-skewed p-values.

Requires:
  data/manuscript/all_fourier_df.parquet  (run python pipeline/aggregate.py first)
  ecdfbounds library

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
import numpy as np
import pandas as pd
from ecdfbounds import ecdf, bootstrap_ecdf_band

from magpyneto2 import statistics

import format_parameters as FP



def _split_into_waves(df, wave_groups=("species", "ID", "date", "id")):
    """Split a dataframe into a list of "wave" dataframes -- one per occurrence
    number of repeated rows for the same neuron (same `wave_groups` key). Wave 0
    holds each neuron's first row, wave 1 its second, etc.
    """
    df = df.copy()
    df["occurrence"] = df.groupby(list(wave_groups)).cumcount()
    return [g.reset_index(drop=True) for _, g in df.groupby("occurrence")]


def split_into_occurrence_waves(all_fourier_df, pval_col="p_value"):
    """`pval_col`: which column to treat as the p-value for the Storey q-value
    computation -- "p_value" (1F, default) or "2f_p_value" (see fig3_supp.py,
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


def _add_qval_inset(ax, wave_sorted_qvals, wave_colors, n_inset=50, x0_frac=0.35, margin=0.7,
                     ylim_bottom=None):
    """Add an inset to `ax` (a log-scale sorted-q-value axes) magnifying each
    wave's first `n_inset` units -- the steep early rise is otherwise
    compressed into a sliver of pixels against the full (often thousands-
    long) neuron count. The inset gets its own independent y-limits (plain
    autoscale on just the first `n_inset` points of each wave) rather than
    matching the parent axes, so the actual vertical spread at that zoom
    level is visible rather than stretched across the parent's full range.

    Placement exploits every wave's curve being sorted (monotonic
    non-decreasing): past some x0, a wave's value only ever increases, so its
    minimum value for x >= x0 is exactly its value AT x0. A box whose top
    edge sits at `margin * min-over-waves(value at x0)`, for x >= x0, is
    therefore *guaranteed* not to intersect any wave's plotted curve -- this
    is computed from the actual data being plotted, not a placement guess.

    A wave's q-values can include NaNs (from NaN p-values -- e.g. `2f_p_value`
    rows with no paired 2F harmonic, see fig3_supp.py), which `np.sort` pushes
    to the tail of the sorted array. That tail isn't real, monotonic data, so
    the "value at x0" guarantee above only holds within a wave's finite
    (non-NaN) prefix -- `len(sq)` is not the right length to compare `x0`
    against here, only the count of finite entries is.

    `ylim_bottom`, if given, floors the inset's own (autoscaled) log-scale
    y-axis -- used for the neg-result ("blue"/magnetic) inset, which
    otherwise autoscales down to whatever its smallest plotted q-value
    happens to be.
    """
    if not wave_sorted_qvals:
        return
    finite_lens = [np.sum(np.isfinite(sq)) for sq in wave_sorted_qvals]
    max_n = max(finite_lens)
    x0 = int(x0_frac * max_n)
    if x0 <= n_inset:
        # Panel too small for a clean x0 past the zoomed-in region -- skip
        # rather than risk the inset box colliding with its own source data.
        return
    ceilings = [sq[x0] for sq, flen in zip(wave_sorted_qvals, finite_lens) if flen > x0]
    if not ceilings:
        return
    y1 = min(ceilings) * margin
    if not np.isfinite(y1) or y1 <= 0:
        return

    parent_ylim_bottom, parent_ylim_top = ax.get_ylim()
    log_lo, log_hi = np.log10(parent_ylim_bottom), np.log10(parent_ylim_top)
    y1_frac = np.clip((np.log10(y1) - log_lo) / (log_hi - log_lo), 0.15, 0.85)

    # Bottom offset (0.18, vs. 0.06 for the box's other edges) leaves the
    # inset's own x-tick labels room to sit above the parent axes' bottom
    # border -- too little clearance here lets them collide with the parent's
    # "Neuron" x-axis label just below it.
    inset_bottom = 0.18
    inset_top = y1_frac - 0.04
    if inset_top - inset_bottom < 0.10:
        return
    axins = ax.inset_axes([x0_frac + 0.01, inset_bottom, 0.97 - (x0_frac + 0.01), inset_top - inset_bottom])
    for sq, color in zip(wave_sorted_qvals, wave_colors):
        n = min(n_inset, len(sq))
        axins.plot(np.arange(n), sq[:n], ".", color=color, alpha=FP.ALPHA_TRACE, markersize=FP.MS_DATA, rasterized=True)
    axins.set_yscale("log")
    if ylim_bottom is not None:
        axins.set_ylim(bottom=ylim_bottom)
    # Pad the left edge so points sitting right at x=0 aren't squashed against
    # the inset's own y-axis spine.
    axins.set_xlim(-0.05 * n_inset, n_inset)
    # Inset x-tick labels at ~90% of the main axes' tick label size (FP.FS_BODY,
    # set globally via matplotlib.rc("font", ...) in plot_fig3) -- readable
    # against the inset's cramped width without visually competing with the
    # parent axes' own labels.
    axins.tick_params(axis="x", labelsize=FP.FS_BODY * 0.9)
    axins.tick_params(axis="y", labelsize=FP.FS_BODY - 4)


def plot_uniform_p(waves, axes, percentile=None, colors=None, pval_col="p_value", sens_col="sens",
                    inset_ylim_bottom=None):
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
    (see fig3_supp.py's 2F-harmonic version of this figure).

    `inset_ylim_bottom`: forwarded to `_add_qval_inset`'s `ylim_bottom` --
    plot_fig3 passes 1e-2 for the neg-result (blue) call only.
    """
    if colors is None:
        colors = [None] * len(waves)
    last_n = 0
    wave_sorted_qvals, wave_colors = [], []
    for wave_df, color in zip(waves[::-1], colors[::-1]):
        if len(wave_df) == 0:
            continue
        if percentile is not None:
            pct = np.percentile(wave_df[sens_col], percentile)
            wave_df = wave_df.loc[wave_df[sens_col] > pct]
        if len(wave_df) == 0:
            continue
        pval = wave_df[pval_col].values
        if len(pval) == 0:
            continue
        qval, pi0 = statistics.storey_qvalues(pval, lambda_=0.5)
        last_n = len(qval)
        sq = np.sort(qval)
        wave_sorted_qvals.append(sq)
        wave_colors.append(color)

        axes[0].plot(np.sort(pval), ".", color=color, alpha=FP.ALPHA_TRACE, markersize=FP.MS_DATA, rasterized=True)
        x, lower, upper = bootstrap_ecdf_band(pval)
        axes[1].plot(sq, ".", color=color, alpha=FP.ALPHA_TRACE, markersize=FP.MS_DATA, rasterized=True)

    statistics.nestle_labels(axes[0], x_offset=-0.05, y=False)
    statistics.nestle_labels(axes[1], x_offset=-0.05, y=False)
    if last_n > 0:
        for ax in axes:
            ax.set_xticks([0, last_n])
            ax.set_xticklabels([0, last_n])
    axes[1].set_yscale("log")
    # Floor the q-value axis at 1e-8 -- points below this are dominated by
    # float64 precision-floor artifacts (1-CDF underflowing to exactly 0 for
    # the most extreme NFC, see the dynamic null-distribution-support fix),
    # not meaningfully distinguishable q-values, so let the axis actually
    # show the interesting range instead of stretching to whatever the
    # smallest underflowed point happens to be.
    axes[1].set_ylim(bottom=1e-8)

    _add_qval_inset(axes[1], wave_sorted_qvals, wave_colors, ylim_bottom=inset_ylim_bottom)


def plot_fig3(all_fourier_df, out_dir: Path, pval_col="p_value", sens_col="sens",
              out_name="Fig3.pdf", suptitle=None):
    """`pval_col`/`sens_col`/`out_name`/`suptitle`: overridden by fig3_supp.py to
    reuse this exact function for the 2F-harmonic version of this figure
    ("2f_p_value"/"sens_2f"/"Fig3_supp.pdf") -- defaults reproduce the original
    1F Fig3.pdf byte-for-byte, so `main()` below is unaffected.
    """
    waves, wave_df_l, pos_control_waves = split_into_occurrence_waves(all_fourier_df, pval_col=pval_col)

    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY}
    matplotlib.rc("font", **font)

    # Explicit, non-overlapping color families per population -- occurrence-wave
    # index means "Nth magnet trial" for neg-result vs. "Nth stimulus condition"
    # for pos-control, so the two are not comparable and must never share a color
    # (see plot_uniform_p's docstring). Shade still varies by occurrence within
    # each population, echoing FP.COLOR_MAG ("steelblue") / FP.COLOR_VIS ("coral").
    neg_colors = list(matplotlib.colormaps["Blues"](np.linspace(0.35, 0.9, len(waves))))
    pos_colors = list(matplotlib.colormaps["Oranges"](np.linspace(0.35, 0.9, len(pos_control_waves))))

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
        ("Pigeon hippocampus neurons", lambda df: (df.area == "HP") & (df.species == "Pigeon"), None),
        ("Top 10% most sensitive neurons in pigeon HP",
         lambda df: (df.area == "HP") & (df.species == "Pigeon"), 90),
    ]
    quadrant_slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    panel_letters = "ABCD"

    fig = plt.figure(figsize=(FP.FIGSIZE_FIG3[0] * 1.4, FP.FIGSIZE_FIG3[1] * 1.8))
    outer = fig.add_gridspec(2, 2, wspace=0.45, hspace=0.35,
                              left=0.08, right=0.98, top=0.90, bottom=0.07)

    for (title, cond_filter, percentile), (orow, ocol), letter in \
            zip(conditions, quadrant_slots, panel_letters):
        inner = outer[orow, ocol].subgridspec(2, 2, wspace=0.08, hspace=0.12)
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

        neg_subset = waves if cond_filter is None else [w.loc[cond_filter(w)] for w in waves]
        pos_subset = pos_control_waves if cond_filter is None else \
            [w.loc[cond_filter(w)] for w in pos_control_waves]

        plot_uniform_p(neg_subset, [ax_p_neg, ax_q_neg], percentile=percentile, colors=neg_colors,
                       pval_col=pval_col, sens_col=sens_col, inset_ylim_bottom=1e-2)
        plot_uniform_p(pos_subset, [ax_p_pos, ax_q_pos], percentile=percentile, colors=pos_colors,
                       pval_col=pval_col, sens_col=sens_col)

        ax_p_neg.set_title("Magnetic", fontsize=FP.FS_BODY)
        ax_p_pos.set_title("Visual/Audio", fontsize=FP.FS_BODY)
        ax_p_neg.set_ylabel("Sorted p-values")
        ax_q_neg.set_ylabel("Sorted q-values")
        ax_q_neg.set_xlabel("Neuron")
        ax_q_pos.set_xlabel("Neuron")

        quad_pos = outer[orow, ocol].get_position(fig)
        fig.text((quad_pos.x0 + quad_pos.x1) / 2, quad_pos.y1 + 0.04, title,
                  ha="center", va="bottom", fontsize=FP.FS_BODY + 1, fontweight="bold")
        fig.text(quad_pos.x0, quad_pos.y1 + 0.04, letter,
                  ha="left", va="bottom", fontfamily="arial", fontsize=12, fontweight="bold")

    if suptitle:
        fig.suptitle(suptitle, fontsize=FP.FS_BODY + 2, fontweight="bold", y=0.98)

    out_path = out_dir / out_name
    fig.savefig(out_path, bbox_inches="tight", dpi=FP.DPI)
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


if __name__ == "__main__":
    if in_notebook:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    main()
