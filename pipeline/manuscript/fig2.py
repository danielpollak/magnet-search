"""Fig 2 — excess-count barplots and c-hat distributions for all neurons.

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

DEFAULT_PARQUET = r"../../data\manuscript\all_fourier_df.parquet"


def _fix_excess_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for lbl in labels:
        try:
            val = float(lbl)
            if val < 0.02:          # 1/60 Hz → 0.016
                new_labels.append("0.016")
            else:                   # ≥ 1 Hz: drop trailing .0
                new_labels.append(f"{val:g}")
        except ValueError:
            new_labels.append(lbl)
    ax.legend(handles, new_labels, title="freq (Hz)", ncol=3,
              fontsize=5, title_fontsize=5)


def get_num_positives(df, freq_harmonic=1):
    """Count sessions with excess suspects, optionally filtered by frequency harmonic.

    Args:
        df: DataFrame with 'freq' column
        freq_harmonic: 1 for fundamental frequency (F), 2 for second harmonic (2F), None for all
    """
    diff_list = []
    for (_), recdf in df.groupby(["species", "area", "rec"]):
        if freq_harmonic==1:
            vals= recdf.rr.values 
        elif freq_harmonic==2:
            vals= recdf['2f_rr'].values
        else:
            raise ValueError("freq_harmonic must be 1 or 2")
        
        n_empirical, f_expected, f_lo, f_hi = statistics.get_suspect_stats(
            vals, 0.99, conf_int_α=0.05)

        diff_list.append(1 if n_empirical > f_hi else 0)
    return diff_list


def plot_fig2(all_fourier_df, out_dir: Path):
    print(f"\nDEBUG - Columns in all_fourier_df: {all_fourier_df.columns.tolist()}")
    print(f"DEBUG - Unique frequencies: {sorted(all_fourier_df['freq'].unique())}")

    all_neg_res, all_pos_control, all_unique_pos_control = \
        statistics.get_poscontrols_negresults(all_fourier_df)

    print(f"DEBUG - Frequencies in all_neg_res: {sorted(all_neg_res['freq'].unique())}")
    print(f"DEBUG - Frequencies in all_pos_control: {sorted(all_pos_control['freq'].unique())}")

    all_fourier_df_unique_neg_res = all_neg_res.drop_duplicates(
        subset=["species", "date", "id"], keep="first")
    all_unique_pos_control = all_pos_control.drop_duplicates(
        subset=["species", "date", "id"], keep="first")

    font = {"family": "arial", "size": 8}
    matplotlib.rc("font", **font)

    fig = plt.figure(figsize=(6.5, 6.5), tight_layout=True)
    gs = gridspec.GridSpec(3, 4, left=0, bottom=0, right=1, top=1, wspace=0.3, hspace=0.3)

    ax_A = fig.add_subplot(gs[0, :])
    ax_B = fig.add_subplot(gs[1, :])
    ax_C = fig.add_subplot(gs[2, 0])
    ax_D = fig.add_subplot(gs[2, 1])
    ax_E = fig.add_subplot(gs[2, 2])
    ax_F = fig.add_subplot(gs[2, 3])

    statistics.plot_excess_counts(ax_A, all_neg_res, ylim=(-12, 30))
    _fix_excess_legend(ax_A)
    num_exp_A = all_neg_res.groupby(["species", "area", "rec"]).ngroups
    print(f"Subfig A (magnetic): {num_exp_A} experiments")

    statistics.plot_excess_counts(ax_B, all_pos_control, ylim=(-12, 30))
    _fix_excess_legend(ax_B)
    num_exp_B = all_pos_control.groupby(["species", "area", "rec"]).ngroups
    print(f"Subfig B (visual & auditory): {num_exp_B} experiments")

    vals_neg_res, bins_neg_res = statistics.draw_hist(
        all_fourier_df_unique_neg_res.rr, ax_C, inset=False)
    vals_pos_con, bins_pos_con = statistics.draw_hist(
        all_unique_pos_control.rr, ax_D, inset=False)

    axins_C = statistics.inset_hist(ax_C, vals_neg_res, bins_neg_res)
    axins_D = statistics.inset_hist(ax_D, vals_pos_con, bins_pos_con)

    statistics.plot_combo_scatterplot(
        all_fourier_df.loc[
            (all_fourier_df.date == "20230413_firstsite") &
            np.array([("mag" in rec.lower()) & ("inclined" not in rec)
                      for rec in all_fourier_df.rec])
        ], ax=ax_E)

    statistics.plot_combo_scatterplot(
        all_fourier_df.loc[
            np.array([("visual" in rec) and ("2023-04-13" in rec) and
                      rec.endswith("_45") for rec in all_fourier_df.rec])
        ], ax=ax_F)

    ax_A.set_title("Magnetic stimulation")
    ax_B.set_title("Visual & auditory stimulation")
    ax_C.set_title(f"Magnetic (N={len(all_fourier_df_unique_neg_res)})")
    ax_D.set_title(f"Visual & auditory\n(N={len(all_unique_pos_control)})")
    ax_E.set_title("Magnetic")
    ax_F.set_title("Visual")

    ax_A.annotate("A", xy=(-0.05, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    ax_B.annotate("B", xy=(-0.05, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    ax_C.annotate("C", xy=(-0.20, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    ax_D.annotate("D", xy=(-0.20, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    ax_E.annotate("E", xy=(-0.2,  1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    ax_F.annotate("F", xy=(-0.2,  1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)

    statistics.boundarize_and_nestle(ax_C, x_offset=-0.05, y_offset=-0.05, xprec=1, yprec=1)
    statistics.boundarize_and_nestle(ax_D, x_offset=-0.05, y_offset=-0.05, xprec=1, yprec=1)
    statistics.boundarize_and_nestle(ax_E, x_offset=-0.05, y_offset=-0.05, xprec=1, yprec=1)
    statistics.boundarize_and_nestle(ax_F, x_offset=-0.05, y_offset=-0.05, xprec=1, yprec=1)
    statistics.boundary_ticks(axins_C, yprec=2, x=False)
    statistics.boundary_ticks(axins_D, yprec=2, x=False)

    out_path = out_dir / "Fig2.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)

    # Stats summary printed to stdout
    # Separate F and 2F analyses
    neg_res_F = get_num_positives(all_neg_res, freq_harmonic=1)
    neg_res_2F = get_num_positives(all_neg_res, freq_harmonic=2)
    pos_control_F = get_num_positives(all_pos_control, freq_harmonic=1)
    pos_control_2F = get_num_positives(all_pos_control, freq_harmonic=2)

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
    parser.add_argument("--out-dir", default="../../figs/paper", help="Output directory for PDFs")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET,
                        help=f"Path to all_fourier_df.parquet (default: {DEFAULT_PARQUET})")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.parquet} ...")
    all_fourier_df = pd.read_parquet(args.parquet)
    plot_fig2(all_fourier_df, out_dir)


if __name__ == "__main__":
    if in_notebook:
        print("Running in Jupyter notebook.")
        print("Call plot_fig2() with your own args, or use: main()")
    else:
        main()
