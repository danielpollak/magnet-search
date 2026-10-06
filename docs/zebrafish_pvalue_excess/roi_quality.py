"""Does the zebrafish p-value excess in Fig2C track ROI quality (P(iscell), npix)?

    python docs/zebrafish_pvalue_excess/roi_quality.py

Joins every zebrafish magnetic-population row in all_fourier_df (the Fig2C
population, via get_poscontrols_negresults) back to its suite2p ROI's
P(iscell) and npix, then measures ECDF(p) - p within quantile bins of each.

The join: an engert fourier row's `id` is its position among the ROIs that
survived the iscell/npix mask AND flatline removal (analysis_stages/engert.py
builds `id` as arange over the post-filter traces), not the suite2p index.
`_load_from_nwb`'s `included_mask` is exactly that surviving set, so
roi_df[included_mask].iloc[id] is the row's ROI.

Outputs (next to this script):
    rois.csv                 one row per zebrafish row: rec, id, p_value, p_iscell, npix, ...
    fig_roi_quality.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "pipeline"))

from magpyneto2 import statistics as st  # noqa: E402
from schema import load_experiment  # noqa: E402
from analysis_stages.engert import _load_from_nwb  # noqa: E402

PARQUET = _REPO / "data" / "manuscript" / "all_fourier_df.parquet"
DEDUP_KEY = ["species", "ID", "date", "id"]
N_BINS = 5
# Ordinal blue ramp, 5 steps from the dataviz reference palette (250..650).
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
C_INK, C_NULL = "#333333", "#8a8a8a"
P_GRID = np.linspace(0, 1, 201)


def ecdf_dev(p):
    x = np.sort(p[np.isfinite(p)])
    return np.searchsorted(x, P_GRID, side="right") / len(x) - P_GRID


def join_roi_quality():
    df = pd.read_parquet(PARQUET)
    neg, _, _ = st.get_poscontrols_negresults(df)
    z = neg[neg["species"] == "zebrafish"].copy()
    out = []
    for rec, rows in z.groupby("rec"):
        cfg = load_experiment(str(_REPO / "experiments" / f"{rec}.yml"))
        _, roi_df, included, _ = _load_from_nwb(cfg.nwb_path(), cfg.iscell_threshold,
                                                cfg.npix_threshold, series="suite2p")
        kept = roi_df[included].reset_index(drop=True)
        kept["roi_index"] = np.where(included)[0]
        ids = rows["id"].astype(int).values
        assert ids.max() < len(kept), f"{rec}: id {ids.max()} >= {len(kept)} surviving ROIs"
        q = kept.iloc[ids]
        rows = rows.assign(roi_index=q["roi_index"].values, p_iscell=q["p_iscell"].values,
                           npix=q["npix"].values, iscell_threshold=cfg.iscell_threshold,
                           session=Path(cfg.session_path).name if cfg.session_path else rec)
        out.append(rows)
    return pd.concat(out, ignore_index=True)


def bin_summary(d, col):
    """Quantile bins of `col`: n, dev@0.5, null SD, z."""
    d = d.assign(bin=pd.qcut(d[col], N_BINS, duplicates="drop"))
    rows = []
    for b, sub in d.groupby("bin", observed=True):
        p = sub["p_value"].values
        dev = float(np.interp(0.5, P_GRID, ecdf_dev(p)))
        sd = np.sqrt(0.25 / len(p))
        rows.append(dict(var=col, bin=str(b), n=len(p), dev_at_half=dev, z=dev / sd,
                         curve=ecdf_dev(p)))
    return rows


def main():
    print("Joining zebrafish rows to ROI quality ...")
    z = join_roi_quality()
    # Same-ROI check across a session's repeat trials: `id` indexes the
    # post-filter set, which shares a segmentation across trials, but flatline
    # removal runs per trial -- if it drops different ROIs, the same `id` is a
    # different ROI in different trials and the neuron dedup key is wrong.
    chk = z.groupby(["ID", "date", "id"])["roi_index"].nunique()
    print(f"  ids mapping to >1 suite2p ROI across trials: {(chk > 1).sum()} of {len(chk)}")

    # One row per REAL ROI: within a session every trial shares one suite2p
    # segmentation, so (ID, date, roi_index) is the neuron identity. The
    # production key (ID, date, id) is not -- see the check above.
    print(f"  unique neurons by production key (ID,date,id): "
          f"{z.drop_duplicates(subset=DEDUP_KEY).shape[0]};  by (ID,date,roi_index): "
          f"{z.drop_duplicates(subset=['ID', 'date', 'roi_index']).shape[0]}")
    zu = z.drop_duplicates(subset=["ID", "date", "roi_index"], keep="first")
    zu = zu[np.isfinite(zu["p_value"])]
    zu.drop(columns=[c for c in zu.columns if c.startswith("2f_")]).to_csv(
        _HERE / "rois.csv", index=False)
    p_all = zu["p_value"].values
    dev_all = float(np.interp(0.5, P_GRID, ecdf_dev(p_all)))
    print(f"  zebrafish units (one row per neuron): {len(zu)}; dev@0.5 = {dev_all:+.4f} "
          f"({dev_all / np.sqrt(0.25 / len(zu)):.1f} null SD)")
    print(f"  P(iscell) range {zu.p_iscell.min():.2f}-{zu.p_iscell.max():.2f}, "
          f"npix range {zu.npix.min()}-{zu.npix.max()}")
    print(f"  Spearman(p_value, P(iscell)) = {zu[['p_value','p_iscell']].corr('spearman').iloc[0,1]:+.3f}; "
          f"Spearman(p_value, npix) = {zu[['p_value','npix']].corr('spearman').iloc[0,1]:+.3f}")

    summaries = {c: bin_summary(zu, c) for c in ["p_iscell", "npix"]}
    for c, rows in summaries.items():
        print(f"\n  by {c} quintile:")
        for r in rows:
            print(f"    {r['bin']:>22}  n={r['n']:5d}  dev@0.5={r['dev_at_half']:+.4f}  z={r['z']:+.1f}")

    # Per-session view: is the excess spread out or concentrated?
    print("\n  by recording (rec):")
    for rec, sub in zu.groupby("rec"):
        p = sub["p_value"].values
        dev = float(np.interp(0.5, P_GRID, ecdf_dev(p)))
        print(f"    {rec:38} n={len(p):4d}  dev@0.5={dev:+.4f}  z={dev / np.sqrt(0.25 / len(p)):+.1f}"
              f"  median P(iscell)={np.median(sub.p_iscell):.2f} median npix={np.median(sub.npix):.0f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    labels = {"p_iscell": "P(iscell)", "npix": "npix"}
    for ax, (c, rows) in zip(axes[:2], summaries.items()):
        for colr, r in zip(RAMP5, rows):
            ax.plot(P_GRID, r["curve"], color=colr, linewidth=1.6,
                    label=f"{r['bin']}  n={r['n']}")
        n_min = min(r["n"] for r in rows)
        half = 1.96 * np.sqrt(P_GRID * (1 - P_GRID) / n_min)
        ax.fill_between(P_GRID, -half, half, color=C_NULL, alpha=0.25, linewidth=0)
        ax.axhline(0, color=C_INK, linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("p-value")
        ax.set_title(f"Zebrafish, by {labels[c]} quintile\n"
                     f"(gray: null 95% for the smallest bin, n={n_min})", fontsize=9)
        ax.legend(fontsize=7, frameon=False, title=labels[c], title_fontsize=7,
                  loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    axes[0].set_ylabel("ECDF(p) - p")

    ax = axes[2]
    sc = ax.scatter(zu["npix"], zu["p_iscell"], c=zu["p_value"], s=4, cmap="Blues_r",
                    vmin=0, vmax=1, rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("npix")
    ax.set_ylabel("P(iscell)")
    ax.set_title("Zebrafish units, coloured by p-value", fontsize=9)
    fig.colorbar(sc, ax=ax, label="p-value")
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(_HERE / "fig_roi_quality.png", dpi=150, facecolor="white")
    print(f"\n  wrote {(_HERE / 'fig_roi_quality.png').name}")


if __name__ == "__main__":
    main()
