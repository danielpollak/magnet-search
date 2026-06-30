"""Fig 3 — p-value and q-value uniformity across neuron populations.

Requires:
  data/manuscript/all_fourier_df.parquet  (run python pipeline/aggregate.py first)
  ecdfbounds library

Usage:
    python pipeline/manuscript/fig3.py
    python pipeline/manuscript/fig3.py --out-dir figs/paper
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ecdfbounds import ecdf, bootstrap_ecdf_band

from magpyneto2 import statistics

DEFAULT_PARQUET = "../../data/manuscript/all_fourier_df.parquet"


def compute_waves(all_fourier_df):
    all_neg_res, all_pos_control, all_unique_pos_control = \
        statistics.get_poscontrols_negresults(all_fourier_df)

    wave_groups = ["species", "ID", "date", "id"]
    all_neg_res["occurrence"] = all_neg_res.groupby(wave_groups).cumcount()
    waves = [g.reset_index(drop=True) for _, g in all_neg_res.groupby("occurrence")]

    u = 0
    wave_df_l = []
    for wave_df in waves:
        c_hat = wave_df.rr.values
        pval = statistics.compute_p_value(c_hat, u)
        qval, pi0 = statistics.storey_qvalues(pval, lambda_=0.5)
        wave_df = wave_df.copy()
        wave_df["pval"] = pval
        wave_df["qval"] = qval
        wave_df["pi0"] = pi0
        wave_df_l.append(wave_df)

    return waves, wave_df_l


def plot_uniform_p(waves, axes, u=0, percentile=None):
    last_n = 0
    for wave_df in waves[::-1]:
        if len(wave_df) == 0:
            continue
        if percentile is not None:
            pct = np.percentile(wave_df.sens, percentile)
            wave_df = wave_df.loc[wave_df.sens > pct]
        if len(wave_df) == 0:
            continue
        c_hat = wave_df["rr"].values
        pval = statistics.compute_p_value(c_hat, u)
        if len(pval) == 0:
            continue
        qval, pi0 = statistics.storey_qvalues(pval, lambda_=0.5)
        last_n = len(qval)

        axes[0].plot(np.sort(pval), ".", alpha=0.5, markersize=4, rasterized=True)
        x, lower, upper = bootstrap_ecdf_band(pval)
        axes[1].plot(np.sort(qval), ".", alpha=0.5, markersize=4, rasterized=True)

    statistics.nestle_labels(axes[0], x_offset=-0.05, y=False)
    statistics.nestle_labels(axes[1], x_offset=-0.05, y=False)
    if last_n > 0:
        for ax in axes:
            ax.set_xticks([0, last_n])
            ax.set_xticklabels([0, last_n])
    axes[1].set_yscale("log")


def plot_fig3(all_fourier_df, out_dir: Path):
    waves, wave_df_l = compute_waves(all_fourier_df)

    font = {"family": "arial", "size": 6}
    matplotlib.rc("font", **font)

    fig, axes = plt.subplots(2, 4, figsize=(6, 3), sharey="row", sharex="col")

    plot_uniform_p(waves, axes[:, 0])
    plot_uniform_p(
        [w.loc[w.species == "Pigeon"] for w in waves], axes[:, 1])
    plot_uniform_p(
        [w.loc[(w.area == "HP") & (w.species == "Pigeon")] for w in waves], axes[:, 2])
    plot_uniform_p(
        [w.loc[(w.area == "HP") & (w.species == "Pigeon")] for w in waves],
        axes[:, 3], percentile=90)

    [ax.set_xlabel("Neuron") for ax in axes[1, :]]
    axes[0, 0].set_title("All Neurons")
    axes[0, 1].set_title("Pigeon Neurons")
    axes[0, 2].set_title("Pigeon HP Neurons")
    axes[0, 3].set_title("Pigeon HP Neurons\n> 90%ile sensitivity")
    axes[0, 0].set_ylabel("Sorted p-values")
    axes[1, 0].set_ylabel("Sorted q-values")

    axes[0, 0].annotate("A", xy=(-0.1, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    axes[0, 1].annotate("B", xy=(-0.1, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    axes[0, 2].annotate("C", xy=(-0.1, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    axes[0, 3].annotate("D", xy=(-0.1, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)

    out_path = out_dir / "Fig3.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 3 (p/q-value uniformity)")
    parser.add_argument("--out-dir", default="../../figs/paper", help="Output directory for PDFs")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET,
                        help=f"Path to all_fourier_df.parquet (default: {DEFAULT_PARQUET})")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.parquet} ...")
    all_fourier_df = pd.read_parquet(args.parquet)
    plot_fig3(all_fourier_df, out_dir)


try:
    from IPython import get_ipython
    in_notebook = get_ipython() is not None
except ImportError:
    in_notebook = False

if __name__ == "__main__":
    if in_notebook:
        print("Running in Jupyter notebook.")
        print("Call plot_fig3() with your own args, or use: main()")
    else:
        main()
