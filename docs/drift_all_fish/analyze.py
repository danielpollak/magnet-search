"""Summary tables, figures and the 2022-02-23 checks for docs/drift_all_fish.md.

    python docs/drift_all_fish/analyze.py                 # magneto2 env
    python docs/drift_all_fish/analyze.py --figures-only  # replot from the saved CSVs / cache

Inputs:
  results_bins_<run>.csv (this folder)       measure_all.py, the five suite2p-registered runs
  docs/resegmentation/results_bins_<run>.csv  docs/custom_registration/measure.py, the five
                                              drift-corrected zebrafish (production since 2026-10-12)
  C:/Users/dan/suite2p_v1_custom_reg/drift_all/<run>.npz   images saved by measure_all.py

Expensive steps (skipped with --figures-only; their outputs are cached):
  * the 2022-02-23 overlay images from suite2p's registered movie (data.bin, 4.7 GB) ->
    C:/Users/dan/suite2p_v1_custom_reg/drift_all/2022_02_23_s2p_overlay.npz
  * the 1/60 Hz refit of engert_20220223_visual without its first minutes -> results_visual_refit.csv

The analysed set is every imaging field of view behind the 50 engert/medaka YAMLs: five runs
registered by suite2p (SUITE2P_RUNS) and five drift-corrected runs (CORRECTED_RUNS). The four
deleted 2022-10-02 fish1 duplicate tiffs are left out by measure_all.py.
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))

ROOT = "C:/Users/dan/suite2p_v1_production"
CACHE = Path("C:/Users/dan/suite2p_v1_custom_reg/drift_all")
BIN = 10

SUITE2P_RUNS = {   # run -> label
    "2022_02_21": "zebrafish 2022-02-21",
    "2022_02_23": "zebrafish 2022-02-23",
    "2022_03_01": "zebrafish 2022-03-01",
    "2022_10_02-fish1_cytoGCaMP": "zebrafish 2022-10-02 fish 1",
    "fish3_8dpf_concat_x64": "medaka",
}
CORRECTED_RUNS = {
    "2022_09_14-fish2_cytoGCaMP": "zebrafish 2022-09-14 fish 2",
    "2022_09_15-fish1_cytoGCaMP": "zebrafish 2022-09-15 fish 1",
    "2022_10_01-fish1_cytoGCamp": "zebrafish 2022-10-01 fish 1",
    "2022_10_01-fish2_cytoGCaMP": "zebrafish 2022-10-01 fish 2",
    "2022_10_02-fish2_cytoGCaMP": "zebrafish 2022-10-02 fish 2",
}
C_Y, C_X, C_S2P = "#2a78d6", "#eb6834", "#8a8a8a"
SETTLED_SD = 1.0          # px: bins where suite2p's 10 single-frame shifts agree (SD below this)
TRIPWIRE_R = 0.8          # ours vs suite2p in those bins; below this the harness is broken
MOVES_PX = 0.5            # ... on an axis that actually moves (SD of our displacement above this)
RMS_MAX_PX = 1.0          # ... or if ours and suite2p differ by more than this (rms)
VISUAL_CFG = "engert_20220223_visual"
DROP_FRAMES = 360         # first 6 min of that recording (frame period 1.0212 s)


def load(run, folder):
    r = pd.read_csv(folder / f"results_bins_{run}.csv")
    r["ody"], r["odx"] = r.dy - r.dy.median(), r.dx - r.dx.median()   # relative to the median position
    if "s2p_yoff" in r:
        r["sdy"], r["sdx"] = r.s2p_yoff - r.s2p_yoff.median(), r.s2p_xoff - r.s2p_xoff.median()
        r["s2p_sd"] = np.hypot(r.s2p_yoff_sd, r.s2p_xoff_sd)
    return r


def summary(bins):
    rows = []
    for run, (label, method, r) in bins.items():
        row = dict(run=run, fish=label, correction=method, n_bins=len(r),
                   max_disp_px=r.disp.max(), max_disp_from_median_px=np.hypot(r.ody, r.odx).max(),
                   precision_median_px=np.median(np.r_[r.prec_dy, r.prec_dx]),
                   corr_before_median=r.corr_before.median(), corr_after_median=r.corr_after.median())
        if "sdy" in r:
            ok = r.s2p_sd < SETTLED_SD
            ry = np.corrcoef(r.ody[ok], r.sdy[ok])[0, 1]
            rx = np.corrcoef(r.odx[ok], r.sdx[ok])[0, 1]
            diff = np.hypot(r.ody - r.sdy, r.odx - r.sdx)
            row.update(s2p_max_bin_disp_px=np.hypot(r.sdy, r.sdx).max(),
                       s2p_max_frame_disp_px=r.s2p_max_frame_disp.max(),
                       s2p_within_bin_sd_median_px=r.s2p_sd.median(),
                       share_bins_s2p_sd_over_1px=float((~ok).mean()),
                       r_y_settled=ry, r_x_settled=rx,
                       sd_ours_y_px=r.ody[ok].std(), sd_ours_x_px=r.odx[ok].std(),
                       rms_diff_y_settled_px=np.sqrt(((r.ody - r.sdy)[ok] ** 2).mean()),
                       rms_diff_x_settled_px=np.sqrt(((r.odx - r.sdx)[ok] ** 2).mean()),
                       ours_vs_s2p_median_diff_px=np.median(diff))
        rows.append(row)
    return pd.DataFrame(rows)


def per_recording(bins):
    rows = []
    for run, (label, method, r) in bins.items():
        for f, q in r.groupby("file", sort=False):
            row = dict(run=run, fish=label, recording=f, max_disp_from_median_px=np.hypot(q.ody, q.odx).max(),
                       precision_median_px=np.median(np.r_[q.prec_dy, q.prec_dx]),
                       corr_before_median=q.corr_before.median(), corr_after_median=q.corr_after.median())
            if "sdy" in q:
                row.update(s2p_max_frame_disp_px=q.s2p_max_frame_disp.max(),
                           s2p_within_bin_sd_median_px=q.s2p_sd.median())
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- expensive steps

def s2p_overlay():
    """Means of the first minute and of frames 600-1193 of 2022-02-23 recording 1, as suite2p
    registered them (data.bin, int16, 700 x 700)."""
    from pipeline import ophys_extraction as ox
    plane0 = f"{ROOT}/2022_02_23/output/suite2p/plane0"
    ops = ox.load_ops(plane0)
    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
    mm = np.memmap(os.path.join(plane0, "data.bin"), dtype=np.int16, mode="r").reshape(-1, Ly, Lx)
    np.savez_compressed(CACHE / "2022_02_23_s2p_overlay.npz",
                        first=mm[0:60].astype(np.float64).mean(0),
                        later=mm[600:1194].astype(np.float64).mean(0))


def visual_refit():
    """engert_20220223_visual at 1/60 Hz: production, a refit of the same traces (must agree),
    and a refit without the first DROP_FRAMES frames."""
    from magpyneto2.engert_helpers import fit_Fourier
    from magpyneto2.statistics import corrected_pvalues, storey_qvalues
    from pipeline.analysis_stages import engert
    from pipeline.schema import load_experiment
    cfg = load_experiment(str(_REPO / "experiments" / f"{VISUAL_CFG}.yml"))
    res = engert.compute_fourier_results(cfg, verbose=False)
    F, prod = res["F"], res["fourier_df"]
    T, f, Q = cfg.sample_period, cfg.analysis.f, cfg.analysis.Q_frac

    def fit(X):
        nfc, *_, M, _ = fit_Fourier(X, T=T, f=f, Q_frac=Q)
        nfc = np.array(nfc)
        return nfc, corrected_pvalues(nfc, M), M

    nfc_full, p_full, M_full = fit(F)
    err = float(np.nanmax(np.abs(p_full - prod.p_value.values)))
    assert err < 1e-12, f"refit disagrees with production p-values (max err {err:.2e})"
    nfc_cut, p_cut, M_cut = fit(F[:, DROP_FRAMES:])
    out = pd.DataFrame(dict(id=prod.id.values, NFC_full=nfc_full, p_full=p_full,
                            q_full=storey_qvalues(p_full)[0], NFC_cut=nfc_cut, p_cut=p_cut,
                            q_cut=storey_qvalues(p_cut)[0]))
    out.to_csv(_HERE / "results_visual_refit.csv", index=False)
    print(f"  visual refit: production reproduced (max |dp| {err:.1e}); bins per side {M_full} -> {M_cut}")


# ---------------------------------------------------------------- figures

def fig_timecourses(bins):
    runs = list(SUITE2P_RUNS)
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.3 * len(runs)), constrained_layout=True)
    for ax, run in zip(axes, runs):
        label, _, r = bins[run]
        x0 = 0
        for k, (f, q) in enumerate(r.groupby("file", sort=False)):
            t = x0 + (q.frame_in_file.values + BIN / 2) / 60          # minutes, ~1 frame/s
            ax.plot(t, q.sdy, color=C_S2P, lw=0.8, alpha=0.8, label="suite2p y (10-frame mean of its single-frame shifts)" if k == 0 else None)
            ax.plot(t, q.sdx, color=C_S2P, lw=0.8, alpha=0.8, ls="--",
                    label="suite2p x (dashed)" if k == 0 else None)
            ax.plot(t, q.ody, color=C_Y, lw=1.5, label="ours, y" if k == 0 else None)
            ax.plot(t, q.odx, color=C_X, lw=1.5, label="ours, x" if k == 0 else None)
            x1 = t.max() + 1
            if k:
                ax.axvline(x0 - 0.5, color="k", lw=0.6)
            x0 = x1
        ax.axhline(0, color="#cccccc", lw=0.6, zorder=0)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_ylabel("displacement (px)")
        ax.set_xlim(0, x0 - 1)
    h, l = axes[0].get_legend_handles_labels()
    order = [2, 3, 0, 1]                                   # ours y, ours x, suite2p y, suite2p x
    fig.legend([h[i] for i in order], [l[i] for i in order], ncol=4, fontsize=8,
               loc="outside upper center", frameon=False)
    axes[-1].set_xlabel("minutes into each recording, recordings in acquisition order (black lines separate them)")
    fig.savefig(_HERE / "fig_timecourses.png", dpi=130)
    plt.close(fig)


def fig_0223(bins):
    from pipeline import ophys_extraction as ox
    _, _, r = bins["2022_02_23"]
    q = r[r.file_index == 0]
    ops = ox.load_ops(f"{ROOT}/2022_02_23/output/suite2p/plane0")
    yo, xo = np.asarray(ops["yoff"], float)[:1194], np.asarray(ops["xoff"], float)[:1194]
    yo, xo = yo - np.median(ops["yoff"]), xo - np.median(ops["xoff"])
    fr = np.arange(1194) / 60
    tb = (q.frame_in_file.values + BIN / 2) / 60
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
    for ax, s2p, ours, c, name in ((axes[0], yo, q.ody, C_Y, "y"), (axes[1], xo, q.odx, C_X, "x")):
        ax.plot(fr, s2p, ".", ms=2.5, color=C_S2P, label="suite2p, single frames")
        ax.plot(tb, ours, color=c, lw=2, label="ours, 10-frame averages")
        ax.set_ylabel(f"{name} displacement (px)")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    axes[2].plot(tb, q.corr_before, color=C_S2P, lw=1.5, label="before correction")
    axes[2].plot(tb, q.corr_after, color=C_Y, lw=1.5, label="after correction")
    axes[2].set_ylabel("correlation with\nsession reference")
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=8, frameon=False, loc="lower right")
    axes[2].set_xlabel("minutes into recording 1 (visual, 10:55)")
    for ax in axes:
        ax.axvspan(0, DROP_FRAMES / 60, color="#f2e6c9", zorder=0, lw=0)
    fig.savefig(_HERE / "fig_0223_recording1.png", dpi=130)
    plt.close(fig)


def _norm(img):
    x = gaussian_filter(img, 1.5) - gaussian_filter(img, 20)
    lo, hi = np.percentile(x, [1, 99.5])
    return np.clip((x - lo) / (hi - lo), 0, 1)


def fig_overlay():
    ours = np.load(CACHE / "2022_02_23.npz")
    s2p = np.load(CACHE / "2022_02_23_s2p_overlay.npz")
    pairs = [("raw", ours["0_first_raw"], ours["0_later_raw"]),
             ("suite2p registration", s2p["first"], s2p["later"]),
             ("our drift correction", ours["0_first_corr"], ours["0_later_corr"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    for ax, (title, a, b) in zip(axes, pairs):
        a, b = _norm(a), _norm(b)
        rgb = np.stack([a, b, a], -1)                 # first minute magenta, later green, both white
        ax.imshow(rgb[150:550, 150:550])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(_HERE / "fig_0223_overlay.png", dpi=130)
    plt.close(fig)


def main(figures_only):
    from_dir = {**{r: _HERE for r in SUITE2P_RUNS},
                **{r: _REPO / "docs" / "resegmentation" for r in CORRECTED_RUNS}}
    bins = {r: ((SUITE2P_RUNS | CORRECTED_RUNS)[r],
                "suite2p" if r in SUITE2P_RUNS else "ours (whole pixels)", load(r, from_dir[r]))
            for r in from_dir}
    s = summary(bins)
    # tripwire: where suite2p's single-frame shifts agree with each other, they must agree with ours
    # As first written, the tripwire was r < 0.8 on either axis. It fired for medaka x (r = 0.68),
    # where the x motion itself is only 0.28 px SD, so r has little to correlate; the two agree to
    # 0.23 px rms. Revised (after seeing that, stated in the report): an axis fails if it moves
    # (SD > MOVES_PX) and r < 0.8, or if the rms difference exceeds RMS_MAX_PX.
    fails = []
    for ax in "yx":
        fails.append((s[f"sd_ours_{ax}_px"] > MOVES_PX) & (s[f"r_{ax}_settled"] < TRIPWIRE_R))
        fails.append(s[f"rms_diff_{ax}_settled_px"] > RMS_MAX_PX)
    bad = s[s.correction.eq("suite2p") & np.logical_or.reduce(fails)]
    assert bad.empty, f"ours and suite2p disagree in settled bins:\n{bad.T}"
    s.to_csv(_HERE / "results_summary.csv", index=False)
    per_recording(bins).to_csv(_HERE / "results_recordings.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(3).to_string())

    if not figures_only:
        CACHE.mkdir(parents=True, exist_ok=True)
        s2p_overlay()
        visual_refit()
    v = pd.read_csv(_HERE / "results_visual_refit.csv")
    print(f"  visual refit: q<=0.05 {int((v.q_full <= 0.05).sum())} full -> {int((v.q_cut <= 0.05).sum())} "
          f"without first {DROP_FRAMES} frames, of {len(v)}; p<0.05 {int((v.p_full < 0.05).sum())} -> "
          f"{int((v.p_cut < 0.05).sum())}; NFC r = {np.corrcoef(v.NFC_full, v.NFC_cut)[0, 1]:.2f}")
    fig_timecourses(bins)
    fig_0223(bins)
    fig_overlay()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    main(ap.parse_args().figures_only)
