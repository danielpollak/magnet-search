"""Frame timing of the 2022 Q1 zebrafish recordings (2022_02_21, 2022_02_23, 2022_03_01).

    python docs/frame_timing/frame_timing.py                 # reads the NAS once (~1 min), caches
    python docs/frame_timing/frame_timing.py --figures-only  # from the cache

Write-up: docs/frame_timing.md.

Each recording's original acquisition folder on the NAS (read-only here) holds:
  rawdata/z_plane0000_trial000_imaging_information.npy  one row per frame; column 0 = the
                                                         time the frame was logged (s)
  galvo_waveform.npy                                     the scan waveform played for every
                                                         frame (analog-out samples)
  experiment_information.txt                             settings, incl. galvo_scanning_AOrate
                                                         (analog-out samples per second) and
                                                         the trial time
The scan waveform length / AO rate is the frame period by construction; the logged timestamps
measure it. This script compares the two and asks how precisely the period is known.

Outputs (docs/frame_timing/): results_timestamps.npz (cache), results_frame_timing.csv, figures.
"""
import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
NAS = "//datanas/family/data_aggregated/Engert"
SESSIONS = ["2022_02_21", "2022_02_23", "2022_03_01"]
CACHE = _HERE / "results_timestamps.npz"
TABLE = _HERE / "results_frame_timing.csv"
OUTLIER_MS = 10.0          # an interval more than this far from the recording's mean is an outlier
F_TEST = 0.4               # Hz, the magnet frequency of these sessions


def extract():
    arrays, rows = {}, []
    for s in SESSIONS:
        for d in sorted(glob.glob(f"{NAS}/{s}/20*")):
            if not os.path.isdir(d):
                continue
            name = os.path.basename(d)
            t = np.load(glob.glob(f"{d}/rawdata/*imaging_information.npy")[0], allow_pickle=True)[:, 0].astype(float)
            txt = open(f"{d}/experiment_information.txt", errors="ignore").read()
            ao = float(re.search(r"galvo_scanning_AOrate: ([\d.]+)", txt).group(1))
            trial = float(re.search(r"trial_time: ([\d.]+)", txt).group(1))
            n_wave = np.load(f"{d}/galvo_waveform.npy", mmap_mode="r").shape[0]
            arrays[name] = t
            rows.append(dict(recording=name, session=s, ao_rate=ao, wave_samples=n_wave, trial_time=trial))
    np.savez_compressed(CACHE, **arrays)
    pd.DataFrame(rows).to_csv(_HERE / "results_hardware.csv", index=False)


def summarise(ts, hw):
    rows = []
    for name, t in ts.items():
        dt = np.diff(t)
        T_end = (t[-1] - t[0]) / (len(t) - 1)                 # the clock's average period
        resid = t - (t[0] + np.arange(len(t)) * T_end)         # timestamp minus a perfect clock
        h = hw.loc[hw.recording == name].iloc[0]
        dev_ms = (dt - dt.mean()) * 1e3
        rows.append(dict(
            recording=name, session=h.session, frames=len(t), trial_time=h.trial_time,
            T_hardware=h.wave_samples / h.ao_rate, T_mean=dt.mean(), T_endpoints=T_end,
            T_median=np.median(dt), sd_ms=dt.std() * 1e3, min_s=dt.min(), max_s=dt.max(),
            share_beyond_2ms=np.mean(abs(dev_ms) > 2), share_beyond_5ms=np.mean(abs(dev_ms) > 5),
            share_beyond_10ms=np.mean(abs(dev_ms) > OUTLIER_MS),
            lag1_corr=np.corrcoef(dt[:-1], dt[1:])[0, 1],
            resid_sd_ms=resid.std() * 1e3, resid_max_ms=abs(resid).max() * 1e3,
            T_se_endpoints=np.sqrt(2) * resid.std() / (len(t) - 1)))
    return pd.DataFrame(rows)


def short(name):
    return name[5:].replace("_", " ", 1)


def fig_intervals(ts, d, out):
    fig, axes = plt.subplots(3, 4, figsize=(17, 9), sharey=True)
    for ax, (name, t) in zip(axes.ravel(), ts.items()):
        dt = np.diff(t)
        x = (t[1:] - t[0]) / 60
        far = abs(dt - dt.mean()) * 1e3 > OUTLIER_MS
        ax.scatter(x[~far], dt[~far], s=2, color="#555555", lw=0)
        ax.scatter(x[far], dt[far], s=10, color="#c0392b", lw=0)
        ax.axhline(d.loc[d.recording == name, "T_hardware"].iloc[0], color="#2a78d6", lw=1, zorder=0)
        ax.set_title(f"{short(name)}  ({far.sum()} of {len(dt)} beyond ±{OUTLIER_MS:.0f} ms)", fontsize=9, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time since the first frame (min)")
    for ax in axes[:, 0]:
        ax.set_ylabel("interval to the previous frame (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white")


def fig_distribution(ts, d, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    dev = np.concatenate([(np.diff(t) - np.diff(t).mean()) * 1e3 for t in ts.values()])
    ax = axes[0]
    ax.hist(dev, bins=np.arange(-90, 91, 1), color="#555555")
    ax.set_yscale("log")
    for v in (-OUTLIER_MS, OUTLIER_MS):
        ax.axvline(v, color="#c0392b", lw=0.8, ls="--")
    ax.set_xlabel("interval minus its recording's mean (ms)")
    ax.set_ylabel("frames (log scale)")
    ax.set_title(f"A. All {len(dev):,} intervals of the 12 recordings", loc="left", fontsize=10)
    ax = axes[1]
    for name, t in ts.items():
        dt = (np.diff(t) - np.diff(t).mean()) * 1e3
        ax.scatter(dt[:-1], dt[1:], s=2, alpha=0.35, lw=0)
    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60); ax.set_aspect("equal")
    ax.axhline(0, color="#aaaaaa", lw=0.5); ax.axvline(0, color="#aaaaaa", lw=0.5)
    ax.set_xlabel("interval k, minus the mean (ms)")
    ax.set_ylabel("interval k+1, minus the mean (ms)")
    ax.set_title(f"B. Consecutive intervals (correlation {d.lag1_corr.mean():.2f})", loc="left", fontsize=10)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white")


def fig_residual(ts, out):
    fig, ax = plt.subplots(figsize=(14, 4))
    for name, t in ts.items():
        T_end = (t[-1] - t[0]) / (len(t) - 1)
        r = (t - (t[0] + np.arange(len(t)) * T_end)) * 1e3
        ax.plot((t - t[0]) / 60, r, lw=0.5, label=short(name))
    ax.set_xlabel("time since the first frame (min)")
    ax.set_ylabel("timestamp minus a perfect clock (ms)")
    ax.legend(fontsize=7, ncol=6, frameon=False, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white")


def fig_precision(d, out):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    err = np.logspace(-6, -1, 200)
    for n, ls in ((1194, "-"), (2936, "--")):
        ax.plot(err, F_TEST * n * err, color="#333333", ls=ls, label=f"{n} frames ({n * 1.0212 / 60:.0f} min)")
    ax.axhline(0.1, color="#c0392b", lw=0.8, ls=":")
    ax.text(1.2e-6, 0.115, "0.1 bin", color="#c0392b", fontsize=8)
    T_true = d.T_hardware.iloc[0]
    for v, lab in ((1.0, "1.0"), (1.02, "1.02"), (1.021, "1.021"), (1.0212, "1.0212")):
        e = abs(v - T_true)
        ax.axvline(e, color="#2a78d6", lw=0.8, alpha=0.6)
        ax.text(e * 1.08, 30, f"T = {lab}", rotation=90, fontsize=8, color="#2a78d6", va="top")
    ax.axvspan(1e-6, abs(d.T_mean - T_true).max(), color="#2ca25f", alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("error in the assumed frame period (s)")
    ax.set_ylabel(f"bins the {F_TEST} Hz test lands off the stimulus")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    if not a.figures_only or not CACHE.exists():
        extract()
    z = np.load(CACHE)
    ts = {k: z[k] for k in z.files}
    hw = pd.read_csv(_HERE / "results_hardware.csv")
    d = summarise(ts, hw)
    d.to_csv(TABLE, index=False)
    fig_intervals(ts, d, _HERE / "fig_intervals.png")
    fig_distribution(ts, d, _HERE / "fig_distribution.png")
    fig_residual(ts, _HERE / "fig_residual.png")
    fig_precision(d, _HERE / "fig_precision.png")
    pd.set_option("display.width", 250)
    print(d.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
