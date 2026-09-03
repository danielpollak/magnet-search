"""Fig 2 — excess-count barplots and NFC distributions for all neurons.

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



def _fix_excess_legend(ax, ncol=3, loc=None):
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
            if val < 0.02:          # 1/60 Hz → 0.016
                new_lbl = "0.016"
            else:                   # round to 2 decimals, drop trailing .0
                new_lbl = f"{round(val, 2):g}"
        except ValueError:
            new_lbl = lbl
        deduped.setdefault(new_lbl, h)
    legend_kw = dict(title="freq (Hz)", ncol=ncol,
                     fontsize=FP.FS_LEGEND, title_fontsize=FP.FS_LEGEND, frameon=False)
    if loc is not None:
        legend_kw["loc"] = loc
    ax.legend(list(deduped.values()), list(deduped.keys()), **legend_kw)


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
        n_empirical, f_expected, f_lo, f_hi = statistics.suspect_count_significance(
            vals, 0.99, conf_int_α=0.05, eps=eps)

        diff_list.append(1 if n_empirical > f_hi else 0)
    return diff_list


def plot_fig2(all_fourier_df, out_dir: Path):
    print(f"\nDEBUG - Columns in all_fourier_df: {all_fourier_df.columns.tolist()}")
    print(f"DEBUG - Unique frequencies: {sorted(all_fourier_df['freq'].unique())}")

    all_neg_res, all_pos_control, all_unique_pos_control = \
        statistics.get_poscontrols_negresults(all_fourier_df)

    print(f"DEBUG - Frequencies in all_neg_res: {sorted(all_neg_res['freq'].unique())}")
    print(f"DEBUG - Frequencies in all_pos_control: {sorted(all_pos_control['freq'].unique())}")

    # Dedup key includes "ID" (subject), not just "date" -- see
    # get_poscontrols_negresults's own dedup for why "date"+"id" alone
    # collides different animals recorded on the same calendar date.
    all_fourier_df_unique_neg_res = all_neg_res.drop_duplicates(
        subset=["species", "ID", "date", "id"], keep="first")
    all_unique_pos_control = all_pos_control.drop_duplicates(
        subset=["species", "ID", "date", "id"], keep="first")

    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY_LG}
    matplotlib.rc("font", **font)

    fig = plt.figure(figsize=FP.FIGSIZE_FIG2, tight_layout=True)
    gs = gridspec.GridSpec(3, 4, left=0, bottom=0, right=1, top=1, wspace=FP.WSPACE_DEFAULT, hspace=FP.HSPACE_DEFAULT)

    ax_A = fig.add_subplot(gs[0, :])
    ax_B = fig.add_subplot(gs[1, :])
    ax_C = fig.add_subplot(gs[2, :2])
    ax_D = fig.add_subplot(gs[2, 2:])

    # One palette shared by A and B, defined over the union of both panels'
    # frequencies, so 2 Hz and 3 Hz (which occur in both) get the same colour in
    # each -- see statistics.freq_color_map.
    freq_levels = np.concatenate([all_neg_res.freq.unique(), all_pos_control.freq.unique()])

    statistics.plot_excess_counts(ax_A, all_neg_res, ylim=(-15, 35), freq_levels=freq_levels)
    _fix_excess_legend(ax_A, ncol=7)
    num_exp_A = all_neg_res.groupby(["species", "area", "rec"]).ngroups
    print(f"Subfig A (magnetic): {num_exp_A} experiments")

    # stagger_area_labels=False: unlike panel A, B's areas (wulst/CB/HP/WB)
    # are spaced widely enough that alternating labels above/below the area
    # line isn't needed, so every label just goes below it.
    # ylim[1]=45 (was 35): gives combined 1F+2F overflow labels (e.g.
    # zebrafish's "699/150"/"672/308") enough headroom that the
    # keep-inside-frame clamp in plot_excess_counts never has to engage at
    # all for them.
    statistics.plot_excess_counts(ax_B, all_pos_control, ylim=(-15, 45),
                                  stagger_area_labels=False, freq_levels=freq_levels)
    _fix_excess_legend(ax_B, ncol=3, loc="upper center")
    num_exp_B = all_pos_control.groupby(["species", "area", "rec"]).ngroups
    print(f"Subfig B (visual & auditory): {num_exp_B} experiments")

    vals_neg_res, bins_neg_res = statistics.draw_hist(
        all_fourier_df_unique_neg_res.NFC, ax_C, inset=False, bar_color=FP.COLOR_MAG,
        legend_fontsize=FP.FS_LEGEND)
    vals_pos_con, bins_pos_con = statistics.draw_hist(
        all_unique_pos_control.NFC, ax_D, inset=False, bar_color=FP.COLOR_VIS,
        legend_fontsize=FP.FS_LEGEND)

    axins_C = statistics.inset_hist(ax_C, vals_neg_res, bins_neg_res, bar_color=FP.COLOR_MAG)
    axins_D = statistics.inset_hist(ax_D, vals_pos_con, bins_pos_con, bar_color=FP.COLOR_VIS)

    ax_A.set_title("Magnetic stimulation", fontsize=FP.FS_TITLE)
    ax_B.set_title("Visual & auditory stimulation", fontsize=FP.FS_TITLE)
    ax_C.set_title(f"Magnetic (N={len(all_fourier_df_unique_neg_res)})", fontsize=FP.FS_TITLE)
    ax_D.set_title(f"Visual & auditory (N={len(all_unique_pos_control)})", fontsize=FP.FS_TITLE)

    ax_A.annotate("A", xy=(-0.05, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    ax_B.annotate("B", xy=(-0.05, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    # -0.10 (was -0.20): C/D are double-width now, and these offsets are in
    # axes fraction, so keeping -0.20 would have doubled the letters' actual
    # distance from the panels relative to A/B's.
    ax_C.annotate("C", xy=(-0.10, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    ax_D.annotate("D", xy=(-0.10, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)

    statistics.boundarize_and_nestle(ax_C, x_offset=-0.07, y_offset=-0.03, xprec=1, yprec=1)
    statistics.boundarize_and_nestle(ax_D, x_offset=-0.07, y_offset=-0.03, xprec=1, yprec=1)
    statistics.boundary_ticks(axins_C, yprec=2, x=False)
    statistics.boundary_ticks(axins_D, yprec=2, x=False)

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

    print(f"\n% sessions with excess suspects (F):")
    if len(neg_res_F) > 0:
        print(f"  Magnetic:          {sum(neg_res_F)/len(neg_res_F)*100:.1f}% ({len(neg_res_F)} sessions)")
    else:
        print(f"  Magnetic:          no data")
    if len(pos_control_F) > 0:
        print(f"  Visual & auditory: {sum(pos_control_F)/len(pos_control_F)*100:.1f}% ({len(pos_control_F)} sessions)")
    else:
        print(f"  Visual & auditory: no data")

    print(f"\n% sessions with excess suspects (2F):")
    if len(neg_res_2F) > 0:
        print(f"  Magnetic:          {sum(neg_res_2F)/len(neg_res_2F)*100:.1f}% ({len(neg_res_2F)} sessions)")
    else:
        print(f"  Magnetic:          no data")
    if len(pos_control_2F) > 0:
        print(f"  Visual & auditory: {sum(pos_control_2F)/len(pos_control_2F)*100:.1f}% ({len(pos_control_2F)} sessions)")
    else:
        print(f"  Visual & auditory: no data")
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
