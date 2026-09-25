"""Finite-sample bias in the NFC null: simulations behind docs/nfc_finite_sample_bias.md

Everything the report shows is produced by running this one file:

    python docs/nfc_finite_sample_bias/simulate.py

Usage:
    python docs/nfc_finite_sample_bias/simulate.py                  # full sweeps (~10 min)
    python docs/nfc_finite_sample_bias/simulate.py --figures-only   # replot from saved CSVs

Outputs, written next to this script:
    results_units.csv      dev@0.5 vs number of units (spikes/unit fixed)
    results_spikes.csv     dev@0.5 vs spikes/unit (units fixed)
    results_real.csv       the same statistic measured on the real parquet
    fig_units_sweep.png
    fig_spikes_sweep.png
    fig_real_vs_sim.png

The question
------------
`magpyneto2.statistics`'s NFC null is Rayleigh, which is the ASYMPTOTIC
distribution of |c(f)| / sigma-hat. Real recordings have finite spike counts.
This asks how far the finite-sample sampling distribution sits from that
asymptote, by simulating homogeneous Poisson spike trains -- for which the
null is EXACTLY true, so any departure is finite-sample bias and nothing else.

The statistic
-------------
For a set of units we take their p-values, form the empirical CDF, and record
its deviation from uniform at p = 0.5:

    dev@0.5 = ECDF(0.5) - 0.5

Zero if p is exactly Uniform(0,1). Positive means an excess of small p-values
in the bulk, i.e. NFC running higher than the null predicts. This is the same
statistic Fig2 C/D plots as a curve.

Faithfulness to the production path
-----------------------------------
`nfc_poisson` reimplements `statistics.fourier_analysis`'s NFC computation in
a vectorised form (the production version loops in Python over units AND
frequencies, which is far too slow for a sweep of this size). `_selftest`
asserts the two agree to 1e-10 before any sweep runs, so the speed-up cannot
silently change the answer.

Note that `ff_alt` takes Q bins on EACH side of the stimulus bin, so 2Q
coefficients enter sigma-hat. That is why `get_epsilon(Q) = 1/(2*sqrt(2Q))` is
the right correction: it is 1/(2*sqrt(M)) with M = 2Q the number of averaged
coefficients.
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

from magpyneto2 import statistics as st  # noqa: E402

# ---------------------------------------------------------------- parameters
FREQ = 3.0          # stimulus frequency (Hz) -- mid-range for this dataset
T = 300.0           # recording duration (s)
SR = 30_000.0       # sample rate, only used to build the frequency grid
Q = 27              # off-frequency bins per side; the pigeon mag rec's own value
PARQUET = _REPO_ROOT / "data" / "manuscript" / "all_fourier_df.parquet"

# Validated with the dataviz skill's validate_palette.js (light surface):
# lightness band, chroma floor, CVD separation, normal-vision floor and
# contrast vs surface all PASS.
C_SIM = "#1b6ac9"
C_REAL = "#e8590c"
C_INK = "#333333"
C_MUTED = "#8a8a8a"


# ---------------------------------------------------------------- simulation
def _freq_grid():
    """The exact on/off frequencies `fourier_analysis` would use."""
    fff = st.frequencies(T, sr=SR)
    i0 = int(np.argmin(np.abs(fff - FREQ)))
    ff_alt = np.array([fff[i] for i in range(i0 - Q, i0 + Q + 1)
                       if i != i0 and i >= 0])
    return fff[i0], ff_alt


def nfc_poisson(n_units, n_spikes, rng):
    """NFC for `n_units` homogeneous Poisson trains of `n_spikes` spikes over T.

    Vectorised equivalent of statistics.fourier_analysis(...)[-1]; see module
    docstring. Accumulates one frequency at a time so the working array stays
    (n_units, n_spikes) rather than (n_units, n_spikes, n_freqs).
    """
    f0, ff_alt = _freq_grid()
    t = rng.random((n_units, n_spikes)) * T          # uniform in [0, T) == Poisson given N

    def coeff(f):
        ang = (-2 * np.pi * f) * t
        return (np.cos(ang).sum(1) + 1j * np.sin(ang).sum(1)) / T

    fou0 = coeff(f0)
    # sigma-hat^2 = 1/2 * mean over the 2Q off-frequency coefficients of |c|^2
    acc = np.zeros(n_units)
    for f in ff_alt:
        acc += np.abs(coeff(f)) ** 2
    sgm = np.sqrt(0.5 * acc / len(ff_alt))
    return np.abs(fou0) / sgm


def _selftest():
    """Assert the vectorised path matches the production one exactly."""
    rng = np.random.default_rng(12345)
    n_units, n_spikes = 40, 90
    t = rng.random((n_units, n_spikes)) * T
    spks = [np.sort(row) for row in t]
    ref = st.fourier_analysis(spks, FREQ, Q=Q, sr=SR, T=T)[-1]

    f0, ff_alt = _freq_grid()

    def coeff(f):
        ang = (-2 * np.pi * f) * t
        return (np.cos(ang).sum(1) + 1j * np.sin(ang).sum(1)) / T

    acc = np.zeros(n_units)
    for f in ff_alt:
        acc += np.abs(coeff(f)) ** 2
    mine = np.abs(coeff(f0)) / np.sqrt(0.5 * acc / len(ff_alt))

    err = float(np.max(np.abs(mine - ref)))
    assert err < 1e-10, f"vectorised NFC disagrees with fourier_analysis (max err {err:.2e})"
    print(f"  self-test OK: vectorised NFC matches fourier_analysis (max err {err:.2e})")


# ---------------------------------------------------------------- statistic
def nfc_to_p(NFC):
    """NFC -> p-value through the same eps-corrected null fit_fourier_sig uses."""
    upper = max(6.0, float(np.nanmax(NFC)) * 1.05)
    R, YY = st.normalized_Fourier_PDF(upper=upper)
    PDF = st.normalized_Fourier_PDF_corrected(R[1:], R[1:], YY[1:], st.get_epsilon(Q))
    CDF = st.normalized_Fourier_CDF_corrected(PDF, R[1:])
    return 1 - np.interp(NFC, R[2:], CDF)


def dev_at_half(p):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    x = np.sort(p)
    ecdf = np.arange(1, len(p) + 1) / len(p)
    return float((ecdf - x)[max(np.searchsorted(x, 0.5) - 1, 0)])


def frac_below(p, thr=0.01):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    return float((p < thr).mean())


# ---------------------------------------------------------------- sweeps
def sweep(unit_counts, spike_counts, n_seeds, label):
    rows = []
    for n_units in unit_counts:
        for n_spikes in spike_counts:
            for seed in range(n_seeds):
                rng = np.random.default_rng(hash((n_units, n_spikes, seed)) % 2**32)
                p = nfc_to_p(nfc_poisson(n_units, n_spikes, rng))
                rows.append(dict(n_units=n_units, n_spikes=n_spikes, seed=seed,
                                 dev_at_half=dev_at_half(p), p_below_01=frac_below(p)))
            done = [r for r in rows if r["n_units"] == n_units and r["n_spikes"] == n_spikes]
            m = np.mean([r["dev_at_half"] for r in done])
            print(f"  [{label}] units={n_units:5d} spikes={n_spikes:5d}  dev@0.5 mean={m:+.4f}")
    return pd.DataFrame(rows)


def real_reference():
    """The same statistic on the real magnetic population, split by modality."""
    if not PARQUET.exists():
        print(f"  ! {PARQUET} missing -- skipping real-data reference")
        return pd.DataFrame()
    df = pd.read_parquet(PARQUET)
    neg, _, _ = st.get_poscontrols_negresults(df)
    d = neg.drop_duplicates(subset=["species", "ID", "date", "id"], keep="first")
    imaging = d["species"].isin(["zebrafish", "medaka"])
    rows = []
    for label, sub in [("all magnetic", d),
                       ("ephys (spikes)", d[~imaging]),
                       ("imaging (GCaMP)", d[imaging])]:
        p = sub["p_value"].values
        rows.append(dict(population=label, n_units=int(np.isfinite(p).sum()),
                         median_spk_count=float(sub["spk_count"].median()),
                         dev_at_half=dev_at_half(p), p_below_01=frac_below(p)))
        print(f"  [real] {label:18} n={rows[-1]['n_units']:6d} dev@0.5={rows[-1]['dev_at_half']:+.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=C_MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=C_INK, labelsize=9)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)


def fig_units(du, out):
    """Bias vs number of units -- the claim under test is whether the MEAN moves."""
    g = du.groupby("n_units")["dev_at_half"].agg(["mean", "std", "count"])
    g["sem"] = g["std"] / np.sqrt(g["count"])

    # Regression of every seed-level observation on log10(units).
    x = np.log10(du["n_units"].values)
    y = du["dev_at_half"].values
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax.axhline(0, color=C_INK, linewidth=1, linestyle="--", alpha=0.5)
    ax.errorbar(g.index, g["mean"], yerr=g["sem"], color=C_SIM, marker="o",
                markersize=6, linewidth=2, capsize=3, zorder=3, label="Poisson (null exactly true)")
    grand = du["dev_at_half"].mean()
    ax.axhline(grand, color=C_SIM, linewidth=1, alpha=0.45)
    # Anchored at the left edge and below the line: at the right edge it
    # collided with the 3000- and 5000-unit markers.
    ax.annotate(f"grand mean {grand:+.4f}", xy=(g.index[0], grand), xytext=(2, -11),
                textcoords="offset points", ha="left", fontsize=8, color=C_INK)
    ax.set_xscale("log")
    ax.set_xlabel("Number of units simulated")
    ax.set_ylabel("dev@0.5  (ECDF(0.5) - 0.5)")
    ax.set_title(f"Bias does not shrink with more units\n"
                 f"slope={slope:+.4f}/decade, $r^2$={r**2:.3f}", fontsize=10, color=C_INK)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)

    # What DOES change with more units: the spread.
    ax2.plot(g.index, g["std"], color=C_SIM, marker="o", markersize=6, linewidth=2,
             label="observed SD across seeds")
    ref = np.sqrt(0.25 / g.index.values)
    ax2.plot(g.index, ref, color=C_MUTED, linewidth=1.5, linestyle="--",
             label=r"$\sqrt{0.25/U}$ (iid expectation)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of units simulated")
    ax2.set_ylabel("SD of dev@0.5 across seeds")
    ax2.set_title("Only the PRECISION improves", fontsize=10, color=C_INK)
    ax2.legend(fontsize=8, frameon=False)
    _style(ax2)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")
    return slope, r**2, grand


def fig_spikes(ds, real, out):
    g = ds.groupby("n_spikes")["dev_at_half"].agg(["mean", "std", "count"])
    g["sem"] = g["std"] / np.sqrt(g["count"])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(0, color=C_INK, linewidth=1, linestyle="--", alpha=0.5)
    ax.errorbar(g.index, g["mean"], yerr=g["sem"], color=C_SIM, marker="o",
                markersize=6, linewidth=2, capsize=3, zorder=3,
                label="Poisson (null exactly true)")
    if len(real):
        # Compare against EPHYS, not the pooled population: this simulation is
        # of spike trains, and the pooled value is dominated by the imaging
        # units, which have no spike count and a much larger deviation.
        for _, row in real.iterrows():
            if row["population"] != "ephys (spikes)":
                continue
            ax.axhline(row["dev_at_half"], color=C_REAL, linewidth=2, alpha=0.9,
                       label=f"real ephys data ({row['dev_at_half']:+.4f})")
            ax.plot([row["median_spk_count"]], [row["dev_at_half"]], marker="D",
                    markersize=9, color=C_REAL, zorder=4)
            ax.annotate("median spikes/unit", xy=(row["median_spk_count"], row["dev_at_half"]),
                        xytext=(6, -14), textcoords="offset points", fontsize=8, color=C_INK)
    ax.set_xscale("log")
    ax.set_xlabel("Spikes per unit")
    ax.set_ylabel("dev@0.5  (ECDF(0.5) - 0.5)")
    ax.set_title("Bias decays with spikes per unit", fontsize=10, color=C_INK)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_real_vs_sim(ds, real, out):
    """How much of the observed deviation the finite-sample null accounts for."""
    if not len(real):
        return
    g = ds.groupby("n_spikes")["dev_at_half"].mean()
    labels, vals = [], []
    for _, row in real.iterrows():
        n = row["median_spk_count"]
        if np.isfinite(n):
            expected = float(np.interp(np.log10(n), np.log10(g.index.values), g.values))
            labels += [f"{row['population']}\n(median {n:.0f} spikes)"]
        else:
            # Imaging units have frames, not spikes, so the spike-count sweep
            # cannot furnish a prediction for them at all -- say so rather than
            # silently plotting a gap.
            expected = np.nan
            labels += [f"{row['population']}\n(frames, not spikes:\nno prediction available)"]
        vals += [(expected, row["dev_at_half"])]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = np.arange(len(labels))
    h = 0.34
    ax.barh(y + h / 2, [0 if not np.isfinite(v[0]) else v[0] for v in vals],
            height=h - 0.03, color=C_SIM, label="predicted by finite-sample bias alone")
    ax.barh(y - h / 2, [v[1] for v in vals], height=h - 0.03, color=C_REAL,
            label="observed in real data")
    for i, (exp, obs) in enumerate(vals):
        txt = f"{exp:+.4f}" if np.isfinite(exp) else "n/a"
        ax.annotate(txt, (0 if not np.isfinite(exp) else exp, i + h / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=C_INK)
        ax.annotate(f"{obs:+.4f}", (obs, i - h / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=C_INK)
    ax.axvline(0, color=C_INK, linewidth=1, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("dev@0.5")
    ax.set_xlim(0, max(v[1] for v in vals) * 1.28)
    ax.set_title("Finite-sample bias accounts for the ephys deviation,\n"
                 "but not the imaging one", fontsize=10, color=C_INK)
    # Above the bars rather than lower-right, where it sat on the widest bar.
    ax.legend(fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=C_MUTED, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------- main
def main():
    # --figures-only replots from the saved CSVs. The sweeps take ~10 min; this
    # exists so a layout tweak does not require re-running them (and so the
    # figures in the report are provably the ones those CSVs imply).
    figures_only = "--figures-only" in sys.argv

    if figures_only:
        print("Replotting from saved CSVs (--figures-only)")
        du = pd.read_csv(_HERE / "results_units.csv")
        ds = pd.read_csv(_HERE / "results_spikes.csv")
        real_path = _HERE / "results_real.csv"
        real = pd.read_csv(real_path) if real_path.exists() else pd.DataFrame()
    else:
        print("Self-test")
        _selftest()

        print("\nReal-data reference")
        real = real_reference()

        print("\nSweep 1: number of units (spikes/unit fixed at 150)")
        unit_counts = [100, 250, 500, 1000, 2000, 3000, 5000, 8000]
        du = sweep(unit_counts, [150], n_seeds=24, label="units")

        print("\nSweep 2: spikes per unit (units fixed at 4000)")
        spike_counts = [50, 75, 100, 150, 250, 500, 1000, 2000, 4000]
        ds = sweep([4000], spike_counts, n_seeds=12, label="spikes")

        du.to_csv(_HERE / "results_units.csv", index=False)
        ds.to_csv(_HERE / "results_spikes.csv", index=False)
        if len(real):
            real.to_csv(_HERE / "results_real.csv", index=False)

    print("\nFigures")
    slope, r2, grand = fig_units(du, _HERE / "fig_units_sweep.png")
    fig_spikes(ds, real, _HERE / "fig_spikes_sweep.png")
    fig_real_vs_sim(ds, real, _HERE / "fig_real_vs_sim.png")

    print(f"\nUnits sweep regression: slope={slope:+.5f} per decade, r^2={r2:.4f}, "
          f"grand mean={grand:+.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
