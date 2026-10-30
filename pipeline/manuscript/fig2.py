"""Fig 2 â€” excess-count barplots and NFC distributions for all neurons.

Requires:
  data/manuscript/all_fourier_df.parquet  (run python pipeline/aggregate.py first)

Usage:
    python pipeline/manuscript/fig2.py
    python pipeline/manuscript/fig2.py --out-dir figs/paper
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
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from magpyneto2 import statistics

import format_parameters as FP
import tex_numbers
from fig3 import one_recording_per_neuron



def _fix_excess_legend(ax, ncol=3, loc=None, bbox_to_anchor=None):
    handles, labels = ax.get_legend_handles_labels()
    # Dedup by DISPLAY label, keeping only the first handle for each --
    # rounding alone (as before) just gave two near-identical-looking rows
    # the same text without actually merging them; panel B's oddball recs
    # (0.977875286666966 / 0.9788924940846944) and engert/medaka's visual_f
    # (0.016666666666666666 / 0.016667) are the same experiment type,
    # differing only by floating-point rounding, and should be one row.
    deduped = {}
    for h, lbl in zip(handles, labels):
        try:
            val = float(lbl)
            if val < 0.02:          # 1/60 Hz â†’ 0.016
                new_lbl = "0.016"
            else:                   # round to 2 decimals, drop trailing .0
                new_lbl = f"{round(val, 2):g}"
        except ValueError:
            new_lbl = lbl
        deduped.setdefault(new_lbl, h)
    legend_kw = dict(title="freq (Hz)", ncol=ncol,
                     fontsize=FP.FS_LEGEND, title_fontsize=FP.FS_LEGEND)
    if loc is not None:
        legend_kw["loc"] = loc
    if bbox_to_anchor is not None:
        legend_kw["bbox_to_anchor"] = bbox_to_anchor
    ax.legend(list(deduped.values()), list(deduped.keys()), **legend_kw)


def _plot_pvalue_ecdf_deviation(ax, df, color):
    """Deviation of a pooled p-value ECDF from uniform, with the 95% binomial
    band expected under the null (gray). Returns the number of p-values.

    Mirrors the ECDF insets in Fig1 E/F (see fig1.py), and replaces what used
    to be a pooled NFC histogram with a null-PDF overlay.

    Plotting p-values rather than NFC is what makes pooling across experiments
    legitimate. `fit_fourier_sig` already maps each unit's NFC through its OWN
    eps-corrected null, so every `p_value` is Uniform(0,1) under the null
    regardless of that recording's Q -- and Q runs from 8 to 370 across this
    pool (65 distinct values, eps 0.018-0.125), so there is no single NFC null
    curve that would be correct for all of it. The old overlay implicitly
    assumed one.

    `df` holds one row per neuron, its recording closest to 3 Hz -- the same
    bucket as Fig 3A (`fig3.one_recording_per_neuron`, 2026-10-27; from
    2026-10-19 its first recording, before that every recording, with an
    occurrence bootstrap over which one each neuron contributes). Rows without a p-value
    (owl, precomputed) are left out, as in Fig 3.
    """
    p = np.sort(df["p_value"].to_numpy(dtype=float))
    p = p[np.isfinite(p)]
    x = statistics.ECDF_P_GRID
    deviation = np.searchsorted(p, x, side="right") / len(p) - x

    statistics.plot_binomial_ecdf_band(ax, len(p))
    ax.plot(x, deviation, color=color, linewidth=FP.LW_TRACE)
    ax.axhline(0, color=FP.COLOR_NULL, linestyle="--", linewidth=FP.LW_REFERENCE, alpha=0.6)

    ax.set_xlabel("p-value")
    ax.set_ylabel("ECDF - uniform")
    ax.set_xlim((0, 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return len(p)


def flag_sessions_with_excess_suspects(df, freq_harmonic=1):
    """Flag sessions with excess suspects, optionally filtered by frequency harmonic.

    Args:
        df: DataFrame with 'freq' column
        freq_harmonic: 1 for fundamental frequency (F), 2 for second harmonic (2F), None for all
    """
    diff_list = []
    for (_), recdf in df.groupby(["species", "area", "rec"]):
        if freq_harmonic==1:
            vals= recdf.NFC.values
            Q_col = "Q"
        elif freq_harmonic==2:
            vals= recdf['2f_NFC'].values
            Q_col = "Q_2f"
        else:
            raise ValueError("freq_harmonic must be 1 or 2")

        # eps corrects the null distribution for the same finite-Q
        # dependent-sampling effect already baked into the per-unit p_value/
        # 2f_p_value p-values (see fit_fourier_sig) -- without this, "excess
        # suspects" here would be flagged against an uncorrected Rayleigh
        # null while the p-value uniformity panels use the corrected one.
        eps = (statistics.eps_from_Q(recdf[Q_col].iloc[0])
               if Q_col in recdf.columns else 0.0)
        # bound_percentile=0.95 matches the bar drawn by plot_excess_counts:
        # a one-tailed 95th-percentile bound on the null binomial, so under
        # the null a session clears it about 5% of the time.
        n_empirical, f_expected, f_lo, f_hi = statistics.suspect_count_significance(
            vals, 0.99, bound_percentile=0.95, eps=eps)

        diff_list.append(1 if n_empirical > f_hi else 0)
    return diff_list


def plot_fig2(all_fourier_df, out_dir: Path):
    print(f"\nDEBUG - Columns in all_fourier_df: {all_fourier_df.columns.tolist()}")
    print(f"DEBUG - Unique frequencies: {sorted(all_fourier_df['freq'].unique())}")

    all_neg_res, all_pos_control, all_unique_pos_control = \
        statistics.get_poscontrols_negresults(all_fourier_df)

    print(f"DEBUG - Frequencies in all_neg_res: {sorted(all_neg_res['freq'].unique())}")
    print(f"DEBUG - Frequencies in all_pos_control: {sorted(all_pos_control['freq'].unique())}")

    # One recording per neuron for C/D, chosen exactly as in Fig 3: the one
    # closest to 3 Hz, ties to the first. Its neuron key includes "ID"
    # (subject), not just "date" -- see get_poscontrols_negresults's own
    # dedup for why "date"+"id" alone collides different animals recorded on
    # the same calendar date.
    all_fourier_df_unique_neg_res = one_recording_per_neuron([all_neg_res])
    all_unique_pos_control = one_recording_per_neuron([all_pos_control])

    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY_LG}
    matplotlib.rc("font", **font)

    fig = plt.figure(figsize=FP.FIGSIZE_FIG2, tight_layout=True)
    gs = gridspec.GridSpec(3, 4, left=0, bottom=0, right=1, top=1, wspace=FP.WSPACE_DEFAULT, hspace=FP.HSPACE_DEFAULT)

    ax_A = fig.add_subplot(gs[0, :])
    ax_B = fig.add_subplot(gs[1, :])
    ax_C = fig.add_subplot(gs[2, :2])
    # Shared y with C: the two deviation curves are the same statistic on the
    # same scale, and the whole comparison is how far each departs from zero.
    # On independent axes C's few-thousandths wiggle would be magnified to
    # look like D's, which is exactly the wrong impression -- shared, C reads
    # as the flat line it is against D's excursion.
    ax_D = fig.add_subplot(gs[2, 2:], sharey=ax_C)
    ax_D.tick_params(labelleft=False)

    # One palette shared by A and B, defined over the union of both panels'
    # frequencies, so 2 Hz and 3 Hz (which occur in both) get the same colour in
    # each -- see statistics.freq_color_map.
    freq_levels = np.concatenate([all_neg_res.freq.unique(), all_pos_control.freq.unique()])

    statistics.plot_excess_counts(ax_A, all_neg_res, ylim=(-15, 25), freq_levels=freq_levels)
    _fix_excess_legend(ax_A, ncol=7, loc="upper left")
    num_exp_A = all_neg_res.groupby(["species", "area", "rec"]).ngroups
    print(f"Subfig A (magnetic): {num_exp_A} experiments")

    # Staggered area labels, as in A: since the coil-on fish recordings left
    # this panel (2026-10-18), medaka has two bars and its "whole brain" ran
    # into zebrafish's.
    # Overflow counts (arrow + number) run above the panel, beside its
    # title, so ylim only has to fit the bars themselves.
    statistics.plot_excess_counts(ax_B, all_pos_control, ylim=(-15, 30),
                                  freq_levels=freq_levels)
    _fix_excess_legend(ax_B, ncol=2, loc="upper center", bbox_to_anchor=(0.7, 1))
    num_exp_B = all_pos_control.groupby(["species", "area", "rec"]).ngroups
    print(f"Subfig B (visual & auditory): {num_exp_B} experiments")

    # Each neuron's recording closest to 3 Hz, i.e. Fig 3A's bucket; N
    # counts the neurons with a p-value.
    n_C = _plot_pvalue_ecdf_deviation(ax_C, all_fourier_df_unique_neg_res, FP.COLOR_MAG)
    n_D = _plot_pvalue_ecdf_deviation(ax_D, all_unique_pos_control, FP.COLOR_VIS)
    # D shares C's y-axis, so it carries neither the tick labels nor the label.
    ax_D.set_ylabel("")

    ax_A.set_title("Magnetic stimulation", fontsize=FP.FS_TITLE)
    ax_B.set_title("Visual & auditory stimulation", fontsize=FP.FS_TITLE)
    ax_C.set_title(f"Magnetic (N={n_C})", fontsize=FP.FS_TITLE)
    ax_D.set_title(f"Visual & auditory (N={n_D})", fontsize=FP.FS_TITLE)

    ax_A.annotate("A", xy=(-0.05, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    ax_B.annotate("B", xy=(-0.05, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    # -0.10 (was -0.20): C/D are double-width now, and these offsets are in
    # axes fraction, so keeping -0.20 would have doubled the letters' actual
    # distance from the panels relative to A/B's.
    ax_C.annotate("C", xy=(-0.10, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    ax_D.annotate("D", xy=(-0.10, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)

    # yprec=3: the deviation axis spans a few thousandths for the magnetic
    # panel, so 1 decimal (the old NFC-histogram precision) collapsed both
    # bounds to "0.0".
    statistics.boundarize_and_nestle(ax_C, x_offset=-0.07, y_offset=-0.03, xprec=1, yprec=3)
    statistics.boundarize_and_nestle(ax_D, x_offset=-0.07, y_offset=-0.03, xprec=1, yprec=3)

    out_path = out_dir / "Fig2.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=FP.DPI)
    print(f"Saved {out_path}")
    if not in_notebook:
        plt.close(fig)

    # Stats summary printed to stdout
    # Separate F and 2F analyses
    neg_res_F = flag_sessions_with_excess_suspects(all_neg_res, freq_harmonic=1)
    neg_res_2F = flag_sessions_with_excess_suspects(all_neg_res, freq_harmonic=2)
    pos_control_F = flag_sessions_with_excess_suspects(all_pos_control, freq_harmonic=1)
    pos_control_2F = flag_sessions_with_excess_suspects(all_pos_control, freq_harmonic=2)

    # Also exported to the manuscript's numbers.tex (see tex_numbers.py), e.g.
    # \importnumber{pct_flagged_magnetic_F} for the share of Fig 2A
    # experiments above the 95% bound.
    numbers = {}
    for harmonic, flags_mag, flags_pos in [("F", neg_res_F, pos_control_F),
                                           ("2F", neg_res_2F, pos_control_2F)]:
        print(f"\n% sessions with excess suspects ({harmonic}):")
        for pop, label, flags in [("magnetic", "Magnetic:         ", flags_mag),
                                  ("poscontrol", "Visual & auditory:", flags_pos)]:
            if len(flags) == 0:
                print(f"  {label} no data")
                continue
            pct = sum(flags) / len(flags) * 100
            print(f"  {label} {pct:.1f}% ({sum(flags)} of {len(flags)} sessions)")
            numbers[f"n_flagged_{pop}_{harmonic}"] = sum(flags)
            numbers[f"pct_flagged_{pop}_{harmonic}"] = f"{pct:.1f}"
        numbers[f"n_experiments_magnetic_{harmonic}"] = len(flags_mag)
        numbers[f"n_experiments_poscontrol_{harmonic}"] = len(flags_pos)
    tex_numbers.write_numbers("fig2", numbers)
    print(f"Unique dates: {all_neg_res['date'].nunique()}  "
          f"areas: {all_neg_res['area'].nunique()}  "
          f"species: {all_neg_res['species'].nunique()}")


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 2 (excess-count barplots + distributions)")
    parser.add_argument("--out-dir", default=FP.OUT_DIR, help="Output directory for PDFs")
    parser.add_argument("--parquet", default=FP.PARQUET_PATH,
                        help=f"Path to all_fourier_df.parquet (default: {FP.PARQUET_PATH})")
    args = parser.parse_args([] if in_notebook else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.parquet} ...")
    all_fourier_df = pd.read_parquet(args.parquet)
    plot_fig2(all_fourier_df, out_dir)


if __name__ == "__main__":
    if in_notebook:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    main()
