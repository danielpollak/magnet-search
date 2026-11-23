"""Figures for docs/zplane.md, from the saved results (no NAS access).

    python docs/zplane/figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
SESSION = "2022_10_01-fish1_cytoGCamp"
C_MAG, C_NOMAG = "#2a78d6", "#e8590c"
GREY = "#8a8a8a"


def fig_zpos():
    Z = pd.read_csv(_HERE / "results_zpos.csv")
    z = Z[(Z.session == SESSION)].copy()
    t0 = z.mtime.min()
    z["t"] = (z.mtime - t0) / 60
    d = np.load(_HERE / "results_corr.npz", allow_pickle=True)
    v = d[[k for k in d.files if SESSION in k][0]].item()
    S, zs, ch, fo = v["S"], v["zs"], v["chunk"], v["cond_folder"]
    mt = z.set_index(["folder", "chunk"]).loc[list(zip(fo, ch)), "t"].values
    o = np.argsort(mt)

    INK = "#222222"
    spans = {f: (g.t.min() - 0.5, g.t.max() + 0.5) for f, g in z.groupby("folder")}
    label = {"magneto": "coil on", "no-magneto": "coil off"}

    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    im = ax[0].imshow(S[o].T, aspect="auto", origin="lower", cmap="viridis",
                      extent=(mt[o][0], mt[o][-1], zs[0] - 2.5, zs[-1] + 2.5), vmin=0, vmax=0.8)
    plt.colorbar(im, ax=ax[0], fraction=0.03, label="correlation")
    for f, g in z.groupby("folder"):
        ax[0].hlines(g.stage_z.iloc[0], *spans[f], color="white", lw=1.5, ls="--")
    ax[0].set_ylabel("stack plane (stage z)")
    ax[0].set_title("A  correlation of each 1-min chunk with each stack plane", fontsize=9, loc="left",
                    pad=16)

    top = zs[-1]
    ok = z.peak_corr >= 0.35
    inside, beyond, none = z[~z.at_edge & ok], z[z.at_edge & ok], z[~ok]
    ax[1].plot(inside.t, inside.z_est, "o", ms=3.5, color=INK, label="matched stack plane")
    ax[1].plot(beyond.t, beyond.z_est, "^", ms=4.5, mfc="none", color=INK,
               label="at or beyond the stack's end")
    ax[1].plot(none.t, np.full(len(none), top + 6), "x", ms=3.5, color=INK,
               label="no plane matches")
    for f, g in z.groupby("folder"):
        ax[1].hlines(g.stage_z.iloc[0], *spans[f], color=INK, lw=1.2, ls="--",
                     label="stage z" if f == "magneto" else None)
    for zz in (zs[0], zs[-1]):
        ax[1].axhline(zz, color="#3a6ea5", lw=0.8, ls=":")
    ax[1].text(spans["no-magneto"][1], zs[0] + 1, "stack's range (dotted)", fontsize=7,
               color="#3a6ea5", ha="right")
    ax[1].set_ylim(zs[0] - 3, top + 10)
    ax[1].set_ylabel("stage z")
    ax[1].set_title("B  matched stack plane and the stage z", fontsize=9, loc="left", pad=16)

    ax[2].plot(inside.t, inside.z_est - inside.stage_z, "o", ms=3.5, color=INK)
    ax[2].plot(beyond.t, beyond.z_est - beyond.stage_z, "^", ms=4.5, mfc="none", color=INK)
    ax[2].axhline(0, color=GREY, lw=0.5)
    ax[2].set_ylabel("matched plane minus stage z")
    ax[2].set_title("C  tissue position relative to the stage", fontsize=9, loc="left", pad=16)

    for a in ax:
        a.set_xlabel("acquisition time (min)")
        for f in ("magneto", "no-magneto"):
            lo, hi = spans[f]
            if a is not ax[0] and f == "magneto":
                a.axvspan(lo, hi, color="#f2f2f2", lw=0, zorder=0)
            a.annotate(label[f], xy=((lo + hi) / 2, 1.0), xycoords=("data", "axes fraction"),
                       xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                       fontsize=8, color="#444")
    ax[0].set_xlim(mt[o][0], mt[o][-1])
    fig.legend(*ax[1].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8,
               frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(_HERE / "fig_zpos.png", dpi=150)
    plt.close(fig)


def fig_roi_z():
    R = pd.read_csv(_HERE / "results_roi_z.csv")
    d = np.load(_HERE / "results_roi_z_series.npz", allow_pickle=True)
    t = (d["t"] - d["t"].min()) / 60
    o = np.argsort(t)          # the six trial tiffs interleave in time
    t = t[o]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    # trends are per stage unit of z; z falls over the recording, so convert to the percent
    # change from the first to the last analysed chunk (z's linear fit over time), positive =
    # brighter at the end
    dz = np.polyfit(t, d["z"][o], 1)[0] * (t.max() - t.min())
    x, y = 100 * R.pred_trend * dz, 100 * R.obs_trend * dz
    ax[0].scatter(x, y, s=5, color="#333", alpha=0.5, lw=0)
    lim = np.nanpercentile(np.abs(np.r_[x, y]), 99)
    ax[0].axhline(0, color=GREY, lw=0.5); ax[0].axvline(0, color=GREY, lw=0.5)
    ax[0].set_xlim(-lim, lim); ax[0].set_ylim(-lim, lim)
    ax[0].set_xlabel("predicted change over the recording\n(% of the ROI's mean brightness)")
    ax[0].set_ylabel("observed change over the recording\n(% of the ROI's mean brightness)")
    for (px, py, txt, ha, va) in ((0.97, 0.97, "plane arriving:\nbrighter at the end", "right", "top"),
                                  (0.03, 0.03, "plane leaving:\ndimmer at the end", "left", "bottom")):
        ax[0].text(px, py, txt, transform=ax[0].transAxes, ha=ha, va=va, fontsize=7, color="#555",
                   bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    from scipy import stats
    rho = stats.spearmanr(R.pred_trend, R.obs_trend)[0]
    ax[0].set_title(f"A  {len(R)} ROIs, Spearman {rho:.2f}", fontsize=9, loc="left")
    # examples: strongest predicted brightening and dimming among well-measured ROIs
    ok = R[R["mean"] > R["mean"].median()]
    ex = [ok.pred_trend.idxmax(), ok.pred_trend.idxmin()]
    for k, (i, title) in enumerate(zip(ex, ["B  predicted to dim (plane leaving)",
                                            "C  predicted to brighten (plane arriving)"])):
        a = ax[1 + k]
        obs, pred = d["obs"][i][o], d["pred"][i][o]
        a.plot(t, obs, "o", ms=3.5, color="#333", label="recording: this ROI, each minute")
        a.plot(t, pred, "-", color=C_MAG, label="stack: this ROI, in the plane matched that minute")
        a.set_xlabel("acquisition time (min)")
        a.set_ylabel("ROI brightness (photons per frame,\naveraged over the ROI's pixels)", fontsize=8)
        a.set_title(f"{title}\nROI {int(R.roi[i])}, {int(R.npix[i])} px", fontsize=9, loc="left")
        a.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(_HERE / "fig_roi_z.png", dpi=150)
    plt.close(fig)


def fig_similarity():
    P = pd.read_csv(_HERE / "results_similarity.csv")
    P["lag"] = (P.t_j - P.t_i).abs()
    P["on_i"] = P.folder_i == "magneto"
    P["on_j"] = P.folder_j == "magneto"
    sess = list(dict.fromkeys(P.session))
    vmax = np.nanpercentile(P[P.i != P.j].S, 99)          # one colour scale for all sessions
    fig, ax = plt.subplots(2, len(sess), figsize=(2.8 * len(sess), 6.4),
                           gridspec_kw=dict(height_ratios=[1, 0.9]))
    for k, s in enumerate(sess):
        g = P[P.session == s]
        n = g.i.max() + 1
        M = np.full((n, n), np.nan)
        M[g.i, g.j] = g.S
        first = g.drop_duplicates("i").sort_values("i")
        tt = first.t_i.values
        im = ax[0, k].imshow(M, cmap="viridis", vmin=0, vmax=vmax, origin="lower",
                             extent=(tt[0], tt[-1], tt[0], tt[-1]))
        # the switch from coil on to coil off
        t_sw = 0.5 * (first[first.on_i].t_i.max() + first[~first.on_i].t_i.min())
        for f in (ax[0, k].axvline, ax[0, k].axhline):
            f(t_sw, color="white", lw=0.8, ls="--")
        ax[0, k].text(t_sw / 2, tt[-1] * 1.01, "coil on", ha="center", va="bottom", fontsize=7)
        ax[0, k].text((t_sw + tt[-1]) / 2, tt[-1] * 1.01, "coil off", ha="center", va="bottom",
                      fontsize=7)
        ax[0, k].set_title(s.replace("_cytoGCaMP", "").replace("_cytoGCamp", ""), fontsize=8,
                           pad=14)
        ax[0, k].set_xlabel("minute j (min)", fontsize=7)
        if k == 0:
            ax[0, k].set_ylabel("minute i (min)", fontsize=7)
        # bottom: within one recording, r against how far apart the two minutes were taken
        for on, ls, lab in ((True, "-", "coil on"), (False, "--", "coil off")):
            w = g[(g.on_i == on) & (g.on_j == on) & (g.i != g.j)]
            bins = np.arange(0, w.lag.max() + 5, 5)
            mid = 0.5 * (bins[1:] + bins[:-1])
            med = [w[(w.lag >= lo) & (w.lag < hi)].S.median() for lo, hi in zip(bins[:-1], bins[1:])]
            ax[1, k].plot(mid, med, ls, marker="o", ms=2.5, color="#333", lw=1.2,
                          label=f"{lab}: two minutes of this recording")
        ax[1, k].set_ylim(0, 1)
        ax[1, k].set_xlabel("time between the two minutes (min)", fontsize=7)
        if k == 0:
            ax[1, k].set_ylabel("r between the two minutes' images\n(median over all such pairs)", fontsize=7)
    fig.subplots_adjust(left=0.05, right=0.93, bottom=0.12, top=0.9, hspace=0.45, wspace=0.3)
    pos = ax[0, -1].get_position()
    cax = fig.add_axes((0.945, pos.y0, 0.008, pos.height))
    fig.colorbar(im, cax=cax, label="r")
    h, l = ax[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.savefig(_HERE / "fig_similarity.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_zpos()
    fig_roi_z()
    if (_HERE / "results_similarity.csv").exists():
        fig_similarity()
