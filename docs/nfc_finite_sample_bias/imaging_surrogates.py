"""Why is the imaging (GCaMP) NFC deviation larger than the ephys one?

Companion to simulate.py, which showed that the ephys deviation is fully
accounted for by finite-sample bias in the Rayleigh null, and that the imaging
deviation (+0.0254 vs +0.0070) is not.

    python docs/nfc_finite_sample_bias/imaging_surrogates.py                 # full run (~10 min)
    python docs/nfc_finite_sample_bias/imaging_surrogates.py --figures-only  # replot

Outputs, written next to this script:
    results_imaging_surrogates.csv
    fig_imaging_surrogates.png

The hypothesis
--------------
GCaMP traces are sparse large transients, not Gaussian noise. `fit_Fourier`'s
null assumes the on- and off-frequency FFT coefficients are iid complex
Gaussian; for a transient-dominated trace they are not, and the effective
sample size is far below the frame count.

The surrogates
--------------
Each replaces the real traces with something that shares a specific property,
so the deviation each one reproduces identifies what is responsible.

  real            the traces as recorded.

  phase_only      FFT magnitudes kept EXACTLY, phases randomized. This is the
                  naive "phase randomization" surrogate, and it is a CONTROL
                  THAT MUST DO NOTHING: NFC is built purely from coefficient
                  magnitudes (see compute_NFC), so preserving them exactly
                  cannot change NFC at all. If this column is not identical to
                  `real` the harness is wrong, not the data.

  gaussian_psd    Gaussian process with the same EXPECTED power spectrum:
                  coefficients redrawn as |Y_k| * (g1 + i*g2)/sqrt(2) with
                  g ~ N(0,1), so E|Z_k|^2 = |Y_k|^2 but the magnitudes are now
                  Rayleigh rather than whatever the real trace had. Keeps the
                  spectral shape; discards non-Gaussianity.

  gaussian_white  iid N(0,1) per frame. Discards both. Isolates the pure
                  finite-N floor for this N and M.

Reading the result
------------------
  real ~= gaussian_psd  -> the SPECTRUM SHAPE drives it (sigma-hat estimated
                           over a non-flat window), not non-Gaussianity.
  real >> gaussian_psd  -> NON-GAUSSIANITY (transient structure) drives it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "pipeline"))

from magpyneto2.engert_helpers import fit_Fourier  # noqa: E402
from magpyneto2.statistics import corrected_pvalues  # noqa: E402
from pipeline.analysis_stages.engert import _load_from_nwb  # noqa: E402
from pipeline import schema  # noqa: E402

# Same validated palette as simulate.py (dataviz validate_palette.js, light).
C_SIM = "#1b6ac9"
C_REAL = "#e8590c"
C_INK = "#333333"
C_MUTED = "#8a8a8a"

N_SURROGATE_DRAWS = 6   # surrogate columns are stochastic; average over draws


# ---------------------------------------------------------------- surrogates
def _rfft_parts(F):
    """Per-trace rFFT of the mean-removed traces."""
    Y = np.fft.rfft(F - F.mean(axis=1, keepdims=True), axis=1)
    return Y


def surrogate(F, kind, rng):
    """Return a surrogate trace matrix of the same shape as F."""
    n_frames = F.shape[1]
    if kind == "real":
        return F
    if kind == "gaussian_white":
        return rng.standard_normal(F.shape)

    Y = _rfft_parts(F)
    mag = np.abs(Y)
    if kind == "phase_only":
        # Magnitudes preserved exactly; phases uniform.
        phase = rng.uniform(0, 2 * np.pi, Y.shape)
        Z = mag * np.exp(1j * phase)
    elif kind == "gaussian_psd":
        # Complex Gaussian with E|Z_k|^2 = |Y_k|^2.
        g = (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape)) / np.sqrt(2)
        Z = mag * g
    else:
        raise ValueError(kind)

    # DC (and Nyquist, when n_frames is even) must stay real for irfft to
    # return a real-valued trace; leaving them complex silently drops their
    # imaginary part and changes the realised spectrum.
    Z[:, 0] = np.abs(Z[:, 0])
    if n_frames % 2 == 0:
        Z[:, -1] = np.abs(Z[:, -1])
    return np.fft.irfft(Z, n=n_frames, axis=1)


# ---------------------------------------------------------------- statistic
def dev_at_half(p):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return np.nan
    x = np.sort(p)
    ecdf = np.arange(1, len(p) + 1) / len(p)
    return float((ecdf - x)[max(np.searchsorted(x, 0.5) - 1, 0)])


def frac_below(p, thr=0.01):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    return float((p < thr).mean()) if len(p) else np.nan


def nfc_and_p(F, T, freq, Q_frac):
    NFC_l, _, _, _, M, _ = fit_Fourier(F, T=T, f=freq, Q_frac=Q_frac)
    NFC = np.asarray(NFC_l)
    return NFC, corrected_pvalues(NFC, M), M


# ---------------------------------------------------------------- driver
def analyse(cfg_name, rng):
    cfg = schema.load_experiment(str(_REPO_ROOT / "experiments" / f"{cfg_name}.yml"))
    F, roi_df, _, _ = _load_from_nwb(cfg.nwb_path(), cfg.iscell_threshold,
                                      cfg.npix_threshold)
    freq, Q_frac, T = cfg.analysis.f, cfg.analysis.Q_frac, cfg.sample_period
    rows = []
    for kind in ("real", "phase_only", "gaussian_psd", "gaussian_white"):
        draws = 1 if kind == "real" else N_SURROGATE_DRAWS
        devs, lows = [], []
        for _ in range(draws):
            NFC, p, M = nfc_and_p(surrogate(F, kind, rng), T, freq, Q_frac)
            devs.append(dev_at_half(p))
            lows.append(frac_below(p))
        rows.append(dict(experiment=cfg_name, kind=kind, n_cells=len(F),
                          n_frames=F.shape[1], freq=freq, M=M,
                          dev_at_half=float(np.mean(devs)),
                          dev_sd=float(np.std(devs)),
                          p_below_01=float(np.mean(lows))))
        print(f"    {kind:15} dev@0.5={rows[-1]['dev_at_half']:+.4f} "
              f"(sd {rows[-1]['dev_sd']:.4f})  P(p<.01)={rows[-1]['p_below_01']:.4f}")
    return rows


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=C_MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=C_INK, labelsize=9)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)


def make_figure(df, out):
    kinds = ["real", "phase_only", "gaussian_psd", "gaussian_white"]
    nice = {"real": "real traces", "phase_only": "phase randomised\n(control: must match real)",
            "gaussian_psd": "Gaussian,\nmatched spectrum",
            "gaussian_white": "Gaussian,\nwhite"}
    g = df.groupby("kind")["dev_at_half"].agg(["mean", "std", "count"])
    g["sem"] = g["std"] / np.sqrt(g["count"])

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(kinds))
    colors = [C_REAL, C_REAL, C_SIM, C_SIM]
    ax.bar(x, [g.loc[k, "mean"] for k in kinds], yerr=[g.loc[k, "sem"] for k in kinds],
           color=colors, width=0.62, capsize=4)
    for i, k in enumerate(kinds):
        m, sem = g.loc[k, "mean"], g.loc[k, "sem"]
        # Clear of the error bar, and on the outside of the bar: above for a
        # positive bar, below for a negative one. Anchoring everything above
        # put the label on the whisker for the positive bars and inside the
        # bar for the negative one.
        up = m >= 0
        ax.annotate(f"{m:+.4f}", (i, m + (sem if up else -sem)),
                    xytext=(0, 7 if up else -16), textcoords="offset points",
                    ha="center", fontsize=9, color=C_INK)
    ax.axhline(0, color=C_INK, linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([nice[k] for k in kinds], fontsize=9)
    ax.set_ylabel("dev@0.5  (ECDF(0.5) - 0.5)")
    ax.set_title(f"Imaging NFC deviation by surrogate "
                 f"({g.loc['real','count']:.0f} zebrafish recordings)",
                 fontsize=10, color=C_INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main():
    # --figures-only replots from the saved CSV; the full run reloads every
    # NWB and takes ~10 min, which a layout tweak should not require.
    if "--figures-only" in sys.argv:
        print("Replotting from results_imaging_surrogates.csv (--figures-only)")
        df = pd.read_csv(_HERE / "results_imaging_surrogates.csv")
        make_figure(df, _HERE / "fig_imaging_surrogates.png")
        return

    # EXACTLY the zebrafish recordings that make up the imaging half of the
    # magnetic (negative-result) population in Fig2 C, so the numbers here are
    # directly comparable with the +0.0254 that population shows. Derived from
    # statistics.get_poscontrols_negresults, not hand-picked.
    #
    # Deliberately NOT included:
    #   - the `visual` recordings, which contain genuine visual responses
    #     (dev@0.5 ~ +0.37) and belong to the positive control, not here;
    #   - `nostim`, which get_poscontrols_negresults also excludes;
    #   - medaka's 3 recs, whose `rec` names are tif basenames rather than
    #     experiment names (578 of 11329 imaging rows).
    experiments = [
        "engert_20220221_magnet", "engert_20220221_visualmagnet",
        "engert_20220223_magnet", "engert_20220223_visualmagnet",
        "engert_20220301_visualmagnet_a", "engert_20220301_visualmagnet_b",
        "engert_20220914_fish2_magneto_0", "engert_20220914_fish2_magneto_1",
        "engert_20220914_fish2_magneto_2", "engert_20220915_fish1_magneto_0",
        "engert_20220915_fish1_magneto_1", "engert_20220915_fish1_magneto_2",
        "engert_20221001_fish1_magneto_0", "engert_20221001_fish1_magneto_1",
        "engert_20221001_fish1_magneto_2", "engert_20221001_fish2_magneto_0",
        "engert_20221001_fish2_magneto_1", "engert_20221001_fish2_magneto_2",
        "engert_20221002_fish1_magneto_1", "engert_20221002_fish1_magneto_2",
        "engert_20221002_fish1_magneto_3", "engert_20221002_fish2_magneto_0",
        "engert_20221002_fish2_magneto_1", "engert_20221002_fish2_magneto_2",
    ]
    rng = np.random.default_rng(0)
    rows = []
    for name in experiments:
        print(f"\n{name}")
        try:
            rows += analyse(name, rng)
        except Exception as exc:                      # noqa: BLE001
            print(f"    ! skipped: {type(exc).__name__}: {exc}")
    df = pd.DataFrame(rows)
    df.to_csv(_HERE / "results_imaging_surrogates.csv", index=False)
    print(f"\nwrote results_imaging_surrogates.csv ({len(df)} rows)")

    print("\nMeans across recordings:")
    print(df.groupby("kind")["dev_at_half"].agg(["mean", "std", "count"]).round(5).to_string())

    # The control has to be exact, not just close.
    piv = df.pivot_table(index="experiment", columns="kind", values="dev_at_half")
    if {"real", "phase_only"} <= set(piv.columns):
        worst = float(np.max(np.abs(piv["real"] - piv["phase_only"])))
        print(f"\nphase_only vs real, max |difference| across recordings: {worst:.2e}")
        print("  (must be ~0 -- NFC depends only on coefficient MAGNITUDES)")

    make_figure(df, _HERE / "fig_imaging_surrogates.png")


if __name__ == "__main__":
    main()
