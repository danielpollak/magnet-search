"""Coverage and floors in the full-resolution imaging traces.

    python docs/coverage_full_resolution/coverage_full_res.py              # ~5 min
    python docs/coverage_full_resolution/coverage_full_res.py --figures-only

Write-up: docs/coverage_full_resolution.md.

The coverage threshold (pipeline/roi_coverage.py) measured how often a trace rose above its
floor: the value it sits at in >= 20% of frames. suite2p's traces had a floor because suite2p's
int16 conversion erased the signal between events (docs/cv2.md). Since 2026-10-05 production
analyses full-resolution traces (pipeline/ophys_extraction.py), and the threshold is off. Some
full-resolution traces still sit at one value in many frames; this script asks which, why,
and whether they need a coverage threshold.

Per ROI of the production population (P(iscell) > 0.5, npix >= 10, inside the fish outline,
flatline removal on the full-resolution traces), every imaging recording in the Fig 2C
magnetic pool:
  mode_share       share of frames at the trace's most common value, full resolution
  mode_value       that value (raw tiff units)
  offset           the recording's detector offset: the median of mode_value over its ROIs
                   (100; 25600 in 20221002_fish1, whose tiffs are stored x256)
  mode_share_s2p   the same for suite2p's trace of the ROI
  mean_above       mean of the full-resolution trace above the offset, in counts per pixel per
                   frame (raw levels divided by offset / 100: 20221002_fish1 is stored x256)
  n_eff            1 / sum(w^2) of the ROI's weights: its effective number of pixels
  expected_share   exp(-mean_above * n_eff): the share of frames in which every pixel of the
                   ROI is at the offset, if the counts were Poisson and equal across pixels
  coverage         pipeline/roi_coverage.activity on the full-resolution trace
  cv2              whole-spectrum CV^2 of the full-resolution trace (docs/cv2.md)
  dev              dev@0.5 over the test frequencies (production null)
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
sys.path.insert(0, str(_HERE.parent / "guard_band_null"))
sys.path.insert(0, str(_HERE.parent / "cv2"))
import guard_band as gb  # noqa: E402
import cv2_report as cr  # noqa: E402
from guard_band import sv, P_GRID, C_INK, C_MUTED, _style  # noqa: E402
from pipeline import nwb_io  # noqa: E402
from pipeline.roi_coverage import activity  # noqa: E402

OUT = _HERE / "results_coverage_full_res.csv"
OUT_ECDF = _HERE / "results_coverage_full_res_ecdf.npz"
SETS = ["zebrafish 0.4 Hz (2022 Q1)", "zebrafish 0.3 Hz", "zebrafish 0.1 Hz", "medaka 0.1 Hz"]
SHORT = {"zebrafish 0.4 Hz (2022 Q1)": "zebrafish 0.4 Hz (2022 Q1)", "zebrafish 0.3 Hz": "zebrafish 0.3 Hz",
         "zebrafish 0.1 Hz": "zebrafish 0.1 Hz", "medaka 0.1 Hz": "medaka 0.1 Hz"}
SHARE_BINS = [0, 0.1, 0.2, 0.35, 0.5, 1.01]
SHARE_LABELS = ["< 0.1", "0.1-0.2", "0.2-0.35", "0.35-0.5", ">= 0.5"]
SHARE_COLOURS = ["#8a8a8a", "#f2a36b", "#eb6834", "#b8431a", "#6b2a0d"]
C_S2P, C_FULL = "#eb6834", "#2a78d6"
EXAMPLE_SHARES = [0.05, 0.15, 0.3, 0.45, 0.65]


def mode_stats(F):
    """(share of frames at the most common value, that value) per trace."""
    share, value = np.empty(len(F)), np.empty(len(F))
    for i, r in enumerate(F):
        v, c = np.unique(r, return_counts=True)
        share[i], value[i] = c.max() / len(r), v[c.argmax()]
    return share, value


def n_eff(roi_df, rows):
    """Effective number of pixels, 1 / sum(w^2), of each ROI's (normalised) weights."""
    out = []
    for m in roi_df["pixel_mask"].iloc[rows]:
        w = np.array([p[2] for p in m], float)
        w = w / w.sum()
        out.append(1.0 / np.sum(w ** 2))
    return np.array(out)


def compute():
    rows, E = [], []
    for name, batch, _ in sv.pool()[0]:
        if batch not in SETS:
            continue
        cfg, F, roi = sv.load(name, outline=True, series="full")
        io, nwbfile = nwb_io.read_nwbfile(cfg.nwb_path())
        Fs, roi_df = nwb_io.read_roi_data(nwbfile, "suite2p")
        io.close()
        Fs = Fs[roi, :F.shape[1]].astype(float)
        s = cr.spectrum(F, cfg)
        ok = s["ok"]
        share, value = mode_stats(F)
        offset = float(np.median(value))   # the detector offset: 100, or 25600 for 20221002_fish1
        above = (F.mean(1) - offset) / (offset / 100)
        ne = n_eff(roi_df, roi)
        P = gb.p_from_nfc(s["X"], s["M"])
        r = pd.DataFrame(dict(
            experiment=name, batch=batch, roi=roi, offset=offset, mode_share=share, mode_value=value,
            mode_share_s2p=mode_stats(Fs)[0], mean_above=above, n_eff=ne,
            expected_share=np.exp(-np.clip(above, 0, None) * ne),
            coverage=activity(F, cfg.sample_period)["coverage"],
            coverage_s2p=activity(Fs, cfg.sample_period)["coverage"],
            cv2=s["cv2"], dev=np.mean(P <= 0.5, 1) - 0.5))
        rows.append(r[ok])
        E.append(cr.ecdf_rows(P[ok]))
        print(f"  {name}: {ok.sum()} ROIs", flush=True)
    d = pd.concat(rows, ignore_index=True)
    E = np.concatenate(E)
    d.to_csv(OUT, index=False)
    np.savez_compressed(OUT_ECDF, E=E)
    return d, E


def share_bin(x):
    return pd.cut(x, SHARE_BINS, labels=SHARE_LABELS, right=False)


# ------------------------------------------------------------------ figures
def fig_shares(d, out):
    """Figure 1: share of frames at the most common value, suite2p against full resolution."""
    fig, axes = plt.subplots(1, len(SETS), figsize=(4.4 * len(SETS), 3.8), sharey=True)
    for ax, s in zip(axes, SETS):
        e = d[d["batch"] == s]
        for col, c, lab in [("mode_share_s2p", C_S2P, "suite2p"), ("mode_share", C_FULL, "full resolution")]:
            x = np.sort(e[col].values)
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=2,
                    label=f"{lab}: median {np.median(x):.0%}")
        ax.axvline(0.2, color=C_MUTED, lw=0.8, ls="--")
        ax.set_xlim(0, 1)
        ax.set_title(f"{SHORT[s]} ({len(e)} ROIs)", fontsize=9)
        ax.set_xlabel("share of frames at the trace's most common value", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        _style(ax)
    axes[0].set_ylabel("ECDF over ROIs", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def examples(d):
    """(experiment, roi) of one 0.1 Hz zebrafish recording's ROIs spanning mode_share: per
    target, the closest ROI."""
    e = d[d["experiment"] == cr.EXAMPLE_REC]
    return [int(e.iloc[(e["mode_share"] - t).abs().argmin()]["roi"]) for t in EXAMPLE_SHARES]


def fig_traces(d, out):
    """Figure 2: full-resolution traces (with suite2p's) from low to high mode share."""
    ids = examples(d)
    cfg, F, roi = sv.load(cr.EXAMPLE_REC, outline=True, series="full")
    io, nwbfile = nwb_io.read_nwbfile(cfg.nwb_path())
    Fs, _ = nwb_io.read_roi_data(nwbfile, "suite2p")
    io.close()
    pos = [int(np.where(roi == i)[0][0]) for i in ids]
    rows = d[d["experiment"] == cr.EXAMPLE_REC].set_index("roi").loc[ids]
    t = np.arange(F.shape[1]) * cfg.sample_period
    z0, z1 = 300, 420
    fig, axes = plt.subplots(len(ids), 2, figsize=(13, 1.9 * len(ids)),
                             gridspec_kw=dict(width_ratios=[3, 1.2]))
    for i, (ax, axz) in enumerate(axes):
        r = rows.iloc[i]
        for a in (ax, axz):
            a.plot(t, F[pos[i]] - r["offset"], color=C_FULL, lw=0.6, label="full resolution")
            a.plot(t, 2 * Fs[ids[i], :F.shape[1]] - r["offset"], color=C_S2P, lw=0.6, label="suite2p F x 2")
            _style(a)
        ax.set_xlim(0, t[-1])
        axz.set_xlim(z0, z1)
        ax.axvspan(z0, z1, color=C_MUTED, alpha=0.15, lw=0)
        ax.set_ylabel("above offset", fontsize=7)
        at = "the offset" if np.isclose(r["mode_value"], r["offset"]) else f"{r['mode_value'] - r['offset']:.2f} above it"
        ax.set_title(f"ROI {ids[i]}: {r['mode_share']:.0%} of frames at {at} "
                     f"(suite2p {r['mode_share_s2p']:.0%}); mean {r['mean_above']:.3f} above offset, "
                     f"{r['n_eff']:.0f} effective pixels; coverage {r['coverage']:.2f}, CV$^2$ {r['cv2']:.2f}",
                     fontsize=8, loc="left")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper right")
    axes[0, 1].set_title(f"zoom, {z0}-{z1} s", fontsize=8)
    axes[-1, 0].set_xlabel("time (s)", fontsize=8)
    axes[-1, 1].set_xlabel("time (s)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_why(d, out):
    """Figure 3: the most common value is the offset, and its share is what Poisson counts
    predict from the ROI's brightness and size."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    e = d[d["batch"] != "zebrafish 0.4 Hz (2022 Q1)"]
    at_offset = np.isclose(e["mode_value"], e["offset"])
    ax = axes[0]
    ax.scatter(e["expected_share"], e["mode_share"], s=3, alpha=0.35, lw=0, rasterized=True,
               c=np.where(at_offset, C_FULL, C_S2P))
    ax.plot([0, 1], [0, 1], color=C_INK, lw=0.6, ls=":")
    ax.set_xlabel("expected share of frames with every pixel at the offset, exp(-mean x n$_{eff}$)", fontsize=8)
    ax.set_ylabel("observed share at the most common value", fontsize=8)
    ax.set_title(f"0.3 Hz, 0.1 Hz zebrafish and medaka: most common value is the offset for "
                 f"{at_offset.mean():.0%} of ROIs (blue)", fontsize=8)
    _style(ax)
    ax = axes[1]
    for s, c in zip(SETS[1:], ["#2a78d6", "#eb6834", "#0d366b"]):
        x = d[d["batch"] == s]
        ax.scatter(x["n_eff"], x["mean_above"], s=3, alpha=0.35, lw=0, color=c, rasterized=True,
                   label=SHORT[s])
    nn = np.logspace(0.5, 3, 50)
    for share in [0.5, 0.2, 0.05]:
        ax.plot(nn, -np.log(share) / nn, color=C_INK, lw=0.8, ls="--")
        ax.text(nn[-1], -np.log(share) / nn[-1], f" {share:.0%}", fontsize=7, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("effective number of pixels in the ROI", fontsize=8)
    ax.set_ylabel("mean above the offset (counts per pixel per frame)", fontsize=8)
    ax.set_title("dashed: expected share of frames at the offset", fontsize=8)
    ax.legend(fontsize=7, frameon=False, markerscale=3)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_calibration(d, E, out):
    """Figure 4: calibration where nothing was presented, ROIs binned by mode share."""
    fig, axes = plt.subplots(1, len(SETS), figsize=(4.4 * len(SETS), 3.8), sharey=True)
    b = share_bin(d["mode_share"])
    for ax, s in zip(axes, SETS):
        for lab, c in zip(SHARE_LABELS, SHARE_COLOURS):
            sel = ((d["batch"] == s) & (b == lab)).values
            if sel.sum() < 20:
                continue
            y = E[sel].mean(0) - P_GRID
            ax.plot(P_GRID, y, color=c, lw=2, label=f"{lab}: {sel.sum()} ROIs, "
                    f"dev@0.5 {np.interp(0.5, P_GRID, y):+.3f}")
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_title(SHORT[s], fontsize=9)
        ax.set_xlabel("p", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False, loc="best", title="share at most common value",
                  title_fontsize=6.5)
        _style(ax)
    axes[0].set_ylabel("ECDF(p) - p over test frequencies", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def summary(d, E):
    b = share_bin(d["mode_share"])
    t = d.assign(bin=b).groupby(["batch", "bin"], observed=True).agg(
        n=("cv2", "size"), cv2=("cv2", "median"), dev=("dev", "mean"),
        cov=("coverage", "median"))
    print(t.round(3).to_string())
    g = d.groupby("batch").agg(
        n=("cv2", "size"), share=("mode_share", "median"), share_s2p=("mode_share_s2p", "median"),
        over20=("mode_share", lambda x: (x >= 0.2).mean()),
        over20_s2p=("mode_share_s2p", lambda x: (x >= 0.2).mean()),
        cov_below01=("coverage", lambda x: (x < 0.1).mean()),
        cov_below01_s2p=("coverage_s2p", lambda x: (x < 0.1).mean()),
        cv2=("cv2", "median"))
    print(g.round(3).to_string())
    for s in SETS:
        ins = (d["batch"] == s).values
        print(s, "dev@0.5 all ROIs", round(np.interp(0.5, P_GRID, E[ins].mean(0) - P_GRID), 4),
              "| coverage >= 0.1 only", round(np.interp(0.5, P_GRID, E[ins & (d["coverage"] >= 0.1).values].mean(0) - P_GRID), 4))
    e = d[d["batch"] != "zebrafish 0.4 Hz (2022 Q1)"]
    print("corr(expected, observed share)", round(np.corrcoef(e["expected_share"], e["mode_share"])[0, 1], 3))


def main():
    if "--figures-only" in sys.argv:
        d = pd.read_csv(OUT)
        E = np.load(OUT_ECDF)["E"]
    else:
        d, E = compute()
    if d["mean_above"].max() > 50:      # results written before mean_above was put in counts
        d["mean_above"] /= d["offset"] / 100
        d["expected_share"] = np.exp(-d["mean_above"].clip(lower=0) * d["n_eff"])
    summary(d, E)
    fig_shares(d, _HERE / "fig_cov_shares.png")
    fig_traces(d, _HERE / "fig_cov_traces.png")
    fig_why(d, _HERE / "fig_cov_why.png")
    fig_calibration(d, E, _HERE / "fig_cov_calibration.png")
    print("  wrote fig_cov_*.png")


if __name__ == "__main__":
    main()
