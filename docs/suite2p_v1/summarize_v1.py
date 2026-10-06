"""Across-session summary of the suite2p v1 comparison, from compare_v1.py's results/*.csv.

Run with the magneto2 env from the repo root, after compare_v1.py has been run for every session:

    C:/Users/dan/anaconda3/envs/magneto2/python.exe docs/suite2p_v1/summarize_v1.py

Prints the markdown tables used in docs/suite2p_v1_comparison.md and writes
docs/suite2p_v1/figs/overview_old_vs_v1.png (one point per session or recording, original on x,
v1 on y). Reads only results/*.csv; writes only that figure.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES, FIGS = HERE / "results", HERE / "figs"
BATCH_C = {"2022 Q1 zebrafish (0.4 Hz)": "#2a78d6", "0.3 Hz zebrafish": "#1baa7d",
           "0.1 Hz zebrafish": "#8a5cc4", "medaka": "#eb6834"}


def batch(session):
    s = session.split("/")[-1]
    if s.startswith("fish3"):
        return "medaka"
    if s.startswith(("2022_02", "2022_03")):
        return "2022 Q1 zebrafish (0.4 Hz)"
    if s.startswith(("2022_09",)):
        return "0.3 Hz zebrafish"
    return "0.1 Hz zebrafish"


def short_rec(r):
    for k in ("visual+magnet", "visual", "magnet", "nostim"):
        if f"_{k}_z_plane" in r:
            return k
    return r.replace(".tif", "")


def wide(df, keys, cols):
    o = df[df.run == "old"].set_index(keys)[cols]
    n = df[df.run == "v1"].set_index(keys)[cols]
    return o.join(n, lsuffix="_old", rsuffix="_v1").reset_index()


def main():
    s = pd.read_csv(RES / "sessions.csv")
    r = pd.read_csv(RES / "recordings.csv")
    h = pd.read_csv(RES / "halving.csv")
    for d in (s, r, h):
        d["batch"] = d.session.map(batch)
        d["name"] = d.session.str.split("/").str[-1]
    order = list(BATCH_C)
    s["bo"] = s.batch.map(order.index)

    # ---- table 1: per session counts and distributions
    w = wide(s, ["bo", "batch", "name"], ["suite2p_version", "n_rois", "n_p_gt_05", "n_npix_ge_10",
                                          "n_pass", "p_q50", "npix_q50", "npix_q90", "seconds_suite2p"])
    w = w.sort_values(["bo", "name"])
    print("| session | batch | original | ROIs | P(iscell) > 0.5 | npix >= 10 | passing both | change in passing |"
          " median P(iscell) | median npix | 90th pct npix |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, x in w.iterrows():
        ch = 100 * (x.n_pass_v1 / x.n_pass_old - 1)
        print(f"| {x['name']} | {x.batch} | {x.suite2p_version_old} | {x.n_rois_old} → {x.n_rois_v1} | "
              f"{x.n_p_gt_05_old} → {x.n_p_gt_05_v1} | {x.n_npix_ge_10_old} → {x.n_npix_ge_10_v1} | "
              f"{x.n_pass_old} → {x.n_pass_v1} | {ch:+.0f}% | {x.p_q50_old:.2f} → {x.p_q50_v1:.2f} | "
              f"{x.npix_q50_old:.0f} → {x.npix_q50_v1:.0f} | {x.npix_q90_old:.0f} → {x.npix_q90_v1:.0f} |")
    tot = w.groupby("batch", sort=False)[["n_rois_old", "n_rois_v1", "n_pass_old", "n_pass_v1"]].sum()
    print("\nper batch totals\n", tot.assign(pass_change=100 * (tot.n_pass_v1 / tot.n_pass_old - 1)).round(1))
    print("\nseconds (old total_plane_runtime -> v1):\n",
          w[["name", "seconds_suite2p_old", "seconds_suite2p_v1"]].round(0).to_string(index=False))

    # ---- table 2: per recording coverage
    rw = wide(r, ["batch", "name", "recording", "rois"],
              ["n", "coverage_q50", "share_coverage_ge_01", "share_clipped"])
    rw["bo"] = rw.batch.map(order.index)
    rw = rw.sort_values(["bo", "name", "recording"])
    print("\n| session | recording | all ROIs: share coverage >= 0.1 | passing: n | passing: median coverage |"
          " passing: share coverage >= 0.1 |")
    print("|---|---|---|---|---|---|")
    for (b, n, rec), g in rw.groupby(["bo", "name", "recording"], sort=True):
        a, p = g[g.rois == "all"].iloc[0], g[g.rois == "pass"].iloc[0]
        print(f"| {n} | {short_rec(rec)} | {a.share_coverage_ge_01_old:.2f} → {a.share_coverage_ge_01_v1:.2f} | "
              f"{p.n_old} → {p.n_v1} | {p.coverage_q50_old:.2f} → {p.coverage_q50_v1:.2f} | "
              f"{p.share_coverage_ge_01_old:.2f} → {p.share_coverage_ge_01_v1:.2f} |")
    print("\nshare of clipped traces (all ROIs), per batch, old / v1:")
    print(rw[rw.rois == "all"].groupby("batch")[["share_clipped_old", "share_clipped_v1"]].agg(["min", "max"]).round(3))

    # ---- table 3: halving
    print("\n| session | raw tiff: share at most common value (value) | raw distinct values (100 frames) |"
          " v1 data.bin / raw mean | v1 data.bin: share at most common value (value) | old data.bin: same |"
          " passing-ROI F at its most common value, old → v1 |")
    print("|---|---|---|---|---|---|---|")
    h["bo"] = h.batch.map(order.index)
    for (b, n), g in h.sort_values(["bo", "name"]).groupby(["bo", "name"], sort=True):
        g = g.set_index("source")
        raw, v1b = g.loc["raw tiff"], g.loc["v1 data.bin"]
        ob = g.loc["old data.bin"] if "old data.bin" in g.index else None
        fo, fv = g.loc["old F (passing ROIs)"], g.loc["v1 F (passing ROIs)"]
        obs = f"{ob.share_at_mode:.1%} ({ob['mode']:.0f})" if ob is not None else "n/a"
        print(f"| {n} | {raw.share_at_mode:.1%} ({raw['mode']:.0f}) | {raw.n_distinct:.0f} | "
              f"{v1b.mean_ratio_to_raw:.4f} | {v1b.share_at_mode:.1%} ({v1b['mode']:.0f}) | {obs} | "
              f"{fo.share_at_mode:.1%} → {fv.share_at_mode:.1%} (at {fo['mode']:.0f} / {fv['mode']:.0f}) |")

    plot(s, r)


def _style(ax):
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    ax.grid(color="#e5e4e0", lw=0.6)
    ax.set_axisbelow(True)


def _diag(ax, lo, hi, log=False):
    ax.plot([lo, hi], [lo, hi], color="#52514e", lw=0.8, ls=":")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")


def plot(s, r):
    w = wide(s, ["batch", "name"], ["n_pass", "p_q50", "p_mean", "npix_q90", "n_npix_ge_10", "n_rois"])
    rw = wide(r, ["batch", "name", "recording", "rois"], ["share_coverage_ge_01", "coverage_q50"])
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), constrained_layout=True)
    panels = [
        (axes[0], w, "n_pass", "ROIs passing P(iscell) > 0.5 and npix >= 10\n(one point per session)", True),
        (axes[1], w, "p_q50", "median P(iscell), all ROIs\n(one point per session)", False),
        (axes[2], w, "n_npix_ge_10", "ROIs with npix >= 10\n(one point per session)", True),
    ]
    for ax, d, c, t, log in panels:
        for b, col in BATCH_C.items():
            g = d[d.batch == b]
            ax.scatter(g[f"{c}_old"], g[f"{c}_v1"], s=28, color=col, label=b, zorder=3)
        v = np.r_[d[f"{c}_old"], d[f"{c}_v1"]]
        lo, hi = (v.min() / 1.4, v.max() * 1.4) if log else (0, 1)
        _diag(ax, lo, hi, log)
        ax.set_title(t, fontsize=9, loc="left")
        ax.set_xlabel("original suite2p")
        ax.set_ylabel("suite2p v1")
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    ax = axes[3]
    for (sub, mk) in [("all", "o"), ("pass", "^")]:
        for b, col in BATCH_C.items():
            g = rw[(rw.batch == b) & (rw.rois == sub)]
            ax.scatter(g.share_coverage_ge_01_old, g.share_coverage_ge_01_v1, s=22, marker=mk,
                       facecolor=col if sub == "pass" else "none", edgecolor=col, zorder=3)
    ax.scatter([], [], marker="o", facecolor="none", edgecolor="#52514e", label="all ROIs")
    ax.scatter([], [], marker="^", color="#52514e", label="passing ROIs")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _diag(ax, 0, 1.02)
    ax.set_title("share of ROIs with coverage >= 0.1\n(one point per recording)", fontsize=9, loc="left")
    ax.set_xlabel("original suite2p")
    ax.set_ylabel("suite2p v1")
    _style(ax)
    fig.savefig(FIGS / "overview_old_vs_v1.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
