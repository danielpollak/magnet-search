"""What the full-resolution traces change, at the stimulus frequency and across the spectrum.

    python docs/full_resolution_rerun/rerun_summary.py              # ~5 min
    python docs/full_resolution_rerun/rerun_summary.py --figures-only

Write-up: docs/full_resolution_rerun.md.

For every 0.3 Hz and 0.1 Hz zebrafish and medaka recording, magneto and no_magneto trials
alike (20221002_fish1's duplicated magneto_2/3 and no_magneto_2/3 left out), on the
production ROIs of the full-resolution traces (P(iscell) > 0.5, npix >= 10, inside the fish
outline, flatline removal; no coverage threshold), using both stored traces of the same ROIs:
  p at the stimulus frequency, production's null F(2, 4M) (within 0.005 of production's
  eps-corrected null), suite2p traces and full-resolution traces;
  dev@0.5 (share of p <= 0.5, minus 0.5) at the stimulus frequency and at every test frequency
  (guard_band.test_bins), per recording;
  the population's mean normalised power at each frequency (each ROI's periodogram divided by
  its 51-bin running mean, averaged over ROIs, divided by its median over 0.03-0.45 Hz), which
  shows stimulus-locked components shared by many ROIs.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE.parent / "guard_band_null"))
sys.path.insert(0, str(_HERE.parent / "cv2"))
import guard_band as gb  # noqa: E402
import cv2_report as cr  # noqa: E402
from guard_band import sv, P_GRID, C_INK, C_MUTED, _style, ecdf_dev  # noqa: E402
from pipeline import nwb_io  # noqa: E402

OUT = _HERE / "results_rerun.csv"
OUT_P = _HERE / "results_rerun_p.npz"
OUT_SPEC = _HERE / "results_rerun_spectra.npz"
C_S2P, C_FULL = "#eb6834", "#2a78d6"
SETS = ["zebrafish 0.3 Hz", "zebrafish 0.1 Hz", "medaka 0.1 Hz"]
SPEC_EXAMPLE = "engert_20221001_fish2_{}magneto_{}"


def recordings():
    names = sorted(os.path.basename(p)[:-4] for p in
                   glob.glob(str(_REPO / "experiments" / "engert_2022091*.yml"))
                   + glob.glob(str(_REPO / "experiments" / "engert_202210*.yml"))
                   + glob.glob(str(_REPO / "experiments" / "medaka_*.yml")))
    return [n for n in names if not ("20221002_fish1" in n and n.endswith(("_2", "_3")))]


def set_of(name):
    return ("medaka 0.1 Hz" if name.startswith("medaka") else
            "zebrafish 0.3 Hz" if "2022091" in name else "zebrafish 0.1 Hz")


def population_spectrum(Y):
    I = np.abs(Y) ** 2
    m = (I / uniform_filter1d(I, gb.MED_WIN, axis=1, mode="nearest")).mean(0)
    return m


def compute():
    rows, pvals, spectra = [], {}, {}
    for name in recordings():
        cfg, F, roi = sv.load(name, outline=True, series="full")
        io, nwbfile = nwb_io.read_nwbfile(cfg.nwb_path())
        Fs, _ = nwb_io.read_roi_data(nwbfile, "suite2p")
        io.close()
        Fs = Fs[roi, :F.shape[1]].astype(float)
        out = dict(rec=name, set=set_of(name), trial=int(name[-1]),
                   cond="no_magneto" if "no_magneto" in name else "magneto")
        for tag, X in [("s2p", Fs), ("full", F)]:
            s = cr.spectrum(X, cfg)
            ok = s["ok"]
            p0 = gb.p_from_nfc(gb.nfc(s["Y"][ok], s["f0"], s["M"], 0), s["M"])
            dev_test = np.mean(gb.p_from_nfc(s["X"][ok], s["M"]) <= 0.5, 0) - 0.5
            out.update({f"n_{tag}": int(ok.sum()), f"dev_stim_{tag}": np.mean(p0 <= 0.5) - 0.5,
                        f"p05_{tag}": np.mean(p0 < 0.05), f"test_mean_{tag}": dev_test.mean(),
                        f"test_sd_{tag}": dev_test.std()})
            pvals[f"{name}|{tag}"] = p0
            if tag == "full":
                f = np.fft.rfftfreq(X.shape[1], cfg.sample_period)
                m = population_spectrum(s["Y"])
                spectra[f"{name}|f"], spectra[f"{name}|m"] = f, m / np.median(m[(f > 0.03) & (f < 0.45)])
                out.update({f"h{k}": float(spectra[f"{name}|m"][int(np.argmin(abs(f - k / 60)))])
                            for k in range(1, 9)})
                out["stim_power"] = float(spectra[f"{name}|m"][s["f0"]])
        rows.append(out)
        print(f"  {name}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    np.savez_compressed(OUT_P, **pvals)
    np.savez_compressed(OUT_SPEC, **spectra)
    return d, pvals, spectra


# ------------------------------------------------------------------ figures
def fig_stimulus(d, pvals, out):
    """Figure 1: p at the stimulus frequency, suite2p vs full resolution, per set (trials 0
    and trials 1-2 separately)."""
    fig, axes = plt.subplots(2, len(SETS), figsize=(4.6 * len(SETS), 7.2), sharey=True)
    for j, s in enumerate(SETS):
        for i, (lab, sel) in enumerate([("trial 0", d["trial"] == 0), ("trials 1-2", d["trial"] > 0)]):
            if s == "zebrafish 0.1 Hz":     # 20221002_fish1's only real trial is named _1, and has no visual stimulus
                fish1 = d["rec"].str.contains("20221002_fish1")
                sel = (sel & ~fish1) | (fish1 if lab == "trial 0" else False)
            recs = d[(d["set"] == s) & sel]["rec"]
            ax = axes[i, j]
            for tag, c, name in [("s2p", C_S2P, "suite2p"), ("full", C_FULL, "full resolution")]:
                p = np.concatenate([pvals[f"{r}|{tag}"] for r in recs])
                ax.plot(P_GRID, ecdf_dev(p), color=c, lw=2,
                        label=f"{name}: {len(p)} ROI traces, dev@0.5 {np.mean(p <= 0.5) - 0.5:+.3f}")
            ax.axhline(0, color=C_INK, lw=0.6)
            ax.set_title(f"{s}, {lab} ({len(recs)} recordings)", fontsize=9)
            ax.set_xlabel("p at the stimulus frequency", fontsize=8)
            ax.legend(fontsize=7, frameon=False, loc="lower center")
            _style(ax)
        axes[0, j].set_ylim(-0.06, 0.12)
    for ax in axes[:, 0]:
        ax.set_ylabel("ECDF(p) - p", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_stim_vs_test(d, out):
    """Figure 2: each recording's deviation at the stimulus frequency against the spread of its
    deviation at single test frequencies."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, tag, name in [(axes[0], "s2p", "suite2p"), (axes[1], "full", "full resolution")]:
        for k, s in enumerate(SETS):
            e = d[d["set"] == s]
            for trial_sel, mk in [(e["trial"] == 0, "o"), (e["trial"] > 0, "^")]:
                x = e[trial_sel]
                ax.errorbar(np.full(len(x), k) + np.linspace(-0.25, 0.25, len(x)), x[f"dev_stim_{tag}"],
                            yerr=2 * x[f"test_sd_{tag}"], fmt=mk, ms=5, lw=0.8, capsize=0,
                            color=[C_S2P, C_FULL, "#0d366b"][k], alpha=0.85)
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_xticks(range(len(SETS)), SETS, fontsize=8)
        ax.set_title(f"{name}: dev@0.5 at the stimulus frequency per recording", fontsize=9)
        _style(ax)
    axes[0].set_ylabel("dev@0.5 (bars: ±2 SD over single test frequencies)", fontsize=8)
    axes[1].plot([], [], "o", color=C_MUTED, label="trial 0")
    axes[1].plot([], [], "^", color=C_MUTED, label="trials 1-2")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_visual(d, spectra, out):
    """Figure 3: population spectra of one fish's six trials, and power at harmonics of 1/60 Hz
    in every recording."""
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax = fig.add_subplot(gs[0, :])
    cols = {0: "#8a8a8a", 1: C_S2P, 2: "#b8431a"}
    for cond in ["", "no_"]:
        for t in range(3):
            n = SPEC_EXAMPLE.format(cond, t)
            f, m = spectra[f"{n}|f"], spectra[f"{n}|m"]
            ax.plot(f, m, color=cols[t], lw=0.9 if cond else 1.3, ls="-" if not cond else "--",
                    label=f"{cond}magneto_{t}")
    for k in range(1, 9):
        ax.axvline(k / 60, color=C_MUTED, lw=0.5, ls=":")
    ax.axvline(0.1, color=C_INK, lw=0.8)
    ax.text(0.1, ax.get_ylim()[1] if False else 6.5, " 0.1 Hz (magnet)", fontsize=7)
    ax.set_xlim(0.005, 0.2)
    ax.set_ylim(0, 7)
    ax.set_xlabel("frequency (Hz); dotted: k/60 Hz", fontsize=8)
    ax.set_ylabel("population mean normalised power", fontsize=8)
    ax.set_title(f"{SPEC_EXAMPLE.format('', 'N').replace('magneto_N', '*')}: population spectrum of each trial", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    _style(ax)
    ax = fig.add_subplot(gs[1, :])
    H = d[[f"h{k}" for k in range(1, 9)]].values
    im = ax.imshow(np.log2(H.T), aspect="auto", cmap="RdBu_r", vmin=-4.5, vmax=4.5)
    ax.set_yticks(range(8), [f"{k}/60" for k in range(1, 9)], fontsize=7)
    ax.set_xticks(range(len(d)), [r.replace("engert_", "").replace("medaka_", "") for r in d["rec"]],
                  rotation=90, fontsize=6)
    ax.set_ylabel("frequency (Hz)", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.02)
    cb.set_label("log2 population power", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def summary(d):
    pd.set_option("display.width", 200)
    print(d.groupby("set")[["n_s2p", "n_full"]].sum())
    g = d.assign(vis=np.where(d["h1"] > 2, "1/60 Hz present", "absent")).groupby(["set", "vis"]).agg(
        recs=("rec", "size"), dev_s2p=("dev_stim_s2p", "mean"), dev_full=("dev_stim_full", "mean"),
        test_full=("test_mean_full", "mean"), sd_full=("test_sd_full", "mean"),
        p05_full=("p05_full", "mean"), h1=("h1", "median"), stim_power=("stim_power", "median"))
    print(g.round(3).to_string())
    print(d[["rec", "trial", "h1", "h3", "h5", "stim_power", "dev_stim_s2p", "dev_stim_full", "test_sd_full"]]
          .round(3).to_string(index=False))


def main():
    if "--figures-only" in sys.argv:
        d = pd.read_csv(OUT)
        pvals, spectra = dict(np.load(OUT_P)), dict(np.load(OUT_SPEC))
    else:
        d, pvals, spectra = compute()
    summary(d)
    fig_stimulus(d, pvals, _HERE / "fig_rerun_stimulus.png")
    fig_stim_vs_test(d, _HERE / "fig_rerun_stim_vs_test.png")
    fig_visual(d, spectra, _HERE / "fig_rerun_visual.png")
    print("  wrote fig_rerun_*.png")


if __name__ == "__main__":
    main()
