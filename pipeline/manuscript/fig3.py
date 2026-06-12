"""Fig 3 — excess-count barplots and c-hat distributions for all neurons.

Requires:
  data/manuscript/all_fourier_df.parquet  (run python pipeline/aggregate.py first)

Usage:
    python pipeline/manuscript/fig3.py
    python pipeline/manuscript/fig3.py --out-dir figs/paper
"""
import argparse
from pathlib import Path

import matplotlib
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


def get_num_positives(df):
    diff_list = []
    for (species, area, rec), recdf in df.groupby(["species", "area", "rec"]):
        n_empirical, f_expected, f_lo, f_hi = statistics.get_suspect_stats(
            recdf.rr.values, 0.99, conf_int_α=0.05)
        diff_list.append(1 if n_empirical > f_hi else 0)
    return diff_list


def plot_fig3(all_fourier_df, out_dir: Path):
    all_neg_res, all_pos_control, all_unique_pos_control = \
        statistics.get_poscontrols_negresults(all_fourier_df)

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
    statistics.plot_excess_counts(ax_B, all_pos_control, ylim=(-12, 30))
    _fix_excess_legend(ax_B)

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

    out_path = out_dir / "Fig3.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)

    # Stats summary printed to stdout
    neg_res_diff_list     = get_num_positives(all_neg_res)
    pos_control_diff_list = get_num_positives(all_pos_control)
    print(f"% sessions with excess suspects — neg result: "
          f"{sum(neg_res_diff_list)/len(neg_res_diff_list)*100:.1f}%  "
          f"pos control: {sum(pos_control_diff_list)/len(pos_control_diff_list)*100:.1f}%")
    print(f"N sessions — neg result: {len(neg_res_diff_list)}  pos control: {len(pos_control_diff_list)}")
    print(f"Unique dates: {all_neg_res['date'].nunique()}  "
          f"areas: {all_neg_res['area'].nunique()}  "
          f"species: {all_neg_res['species'].nunique()}")


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 3 (excess-count barplots + distributions)")
    parser.add_argument("--out-dir", default="../../figs/paper", help="Output directory for PDFs")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET,
                        help=f"Path to all_fourier_df.parquet (default: {DEFAULT_PARQUET})")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.parquet} ...")
    all_fourier_df = pd.read_parquet(args.parquet)
    plot_fig3(all_fourier_df, out_dir)


if __name__ == "__main__":
    main()
