"""Fig 2A with every imaging bar labelled, for checking which recording is which.

Not a paper figure: draws Fig 2A's excess-count panel exactly as fig2.py does
(statistics.plot_excess_counts on the same magnetic population), then writes a
short name above each zebrafish/medaka bar, with its suspect count against the
bound (red if it clears it). Bars are located by repeating plot_excess_counts'
own ordering (species in its fixed order, then area, then rec), one unit apart.

    python pipeline/manuscript/fig2A_imaging_annotated.py   # after aggregate.py
writes figs/paper/Fig2A_imaging_annotated.{pdf,png}.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from magpyneto2 import statistics  # noqa: E402

PARQUET = _REPO / "data" / "manuscript" / "all_fourier_df.parquet"
OUT = _REPO / "figs" / "paper" / "Fig2A_imaging_annotated"
# plot_excess_counts' own species order (its local `desired_order`)
SPECIES_ORDER = ["Owl", "zebra finch", "mouse", "Pigeon", "Quail", "zebrafish", "medaka"]
IMAGING = {"zebrafish", "medaka"}


def short_name(rec):
    """engert_20221002_fish2_magneto_0 -> 1002 fish2 magneto_0; fish3_8dpf_magneto_1.tif -> medaka magneto_1."""
    if rec.endswith(".tif"):
        return "medaka " + rec.removesuffix(".tif").split("8dpf_")[-1]
    return rec.removeprefix("engert_2022").replace("_", " ", 2)


def bar_positions(df):
    """(species, area, rec) -> bar centre, in plot_excess_counts' order."""
    pos, counter = {}, 0
    for species in SPECIES_ORDER:
        sdf = df.loc[df.species == species]
        for area, adf in sdf.drop_duplicates().groupby("area"):
            for rec, _ in adf.groupby("rec"):
                pos[(species, area, rec)] = counter + 0.5
                counter += 1
    return pos


def main():
    all_fourier_df = pd.read_parquet(PARQUET)
    mag, _, pos_control = statistics.get_poscontrols_negresults(all_fourier_df)
    freq_levels = np.concatenate([mag.freq.unique(), pos_control.freq.unique()])
    positions = bar_positions(mag)

    fig, ax = plt.subplots(figsize=(22, 7))
    statistics.plot_excess_counts(ax, mag, ylim=(-15, 95), freq_levels=freq_levels)
    for (species, area, rec), x in positions.items():
        if species not in IMAGING:
            continue
        g = mag.drop_duplicates().loc[(mag.species == species) & (mag.area == area) & (mag.rec == rec)]
        n, _, _, hi = statistics.suspect_count_significance(
            g["NFC"].values, 0.99, bound_percentile=0.95, eps=statistics.eps_from_Q(g["Q"].iloc[0]))
        flagged = n > hi
        ax.text(x, 52, f"{short_name(rec)}  ({n}/{hi:.0f})", rotation=90, ha="center", va="bottom",
                fontsize=7, color="#c0392b" if flagged else "#333333",
                fontweight="bold" if flagged else "normal")
    ax.set_title("Fig 2A, imaging bars labelled: name (suspects / 95% bound at the stimulus frequency); red = above the bound",
                 loc="left", fontsize=10)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"{OUT}.pdf / .png")


if __name__ == "__main__":
    main()
