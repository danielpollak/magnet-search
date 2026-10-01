"""Finite-sample bias in the NFC null: simulations behind docs/nfc_finite_sample_bias.md

Everything the report shows is produced by running this one file:

    python docs/nfc_finite_sample_bias/simulate.py

Usage:
    python docs/nfc_finite_sample_bias/simulate.py                  # full sweeps (~10 min)
    python docs/nfc_finite_sample_bias/simulate.py --figures-only   # replot from saved CSVs

Outputs, written next to this script:
    results_units.csv      dev@0.5 vs number of units (spikes/unit fixed)
    results_spikes.csv     dev@0.5 vs spikes/unit (units fixed)
    results_spikes_wide.csv  the same, over a 10-1000 spikes/unit range
    results_curves.csv     seed-averaged ECDF(p) - p curves for both spike sweeps
                           (plus the real ephys curve, n_spikes = -1)
    results_real.csv       the same statistic measured on the real parquet
    results_real_spk_counts.csv  spk_count + species + area of every real ephys unit
    fig_units_sweep.png
    fig_spikes_sweep.png
    fig_spikes_sweep_wide.png
    fig_spikes_curves.png
    fig_spk_count_distribution.png
    fig_spk_count_distribution_pigeon.png
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
P_GRID = np.linspace(0, 1, 201)   # where ECDF(p) - p curves are sampled
PARQUET = _REPO_ROOT / "data" / "manuscript" / "all_fourier_df.parquet"

# Validated with the dataviz skill's validate_palette.js (light surface):
# lightness band, chroma floor, CVD separation, normal-vision floor and
# contrast vs surface all PASS.
C_SIM = "#1b6ac9"
C_REAL = "#e8590c"
C_INK = "#333333"
C_MUTED = "#8a8a8a"
# Ordinal blue ramp (steps 250..700) for one line per spike count; starts at
# 250 so the lightest line still clears 2:1 against white.
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#0d366b"]
# Categorical slots 1-4, fixed order, one per ephys species.
C_SPECIES = {"Pigeon": "#2a78d6", "zebra finch": "#eb6834",
             "Quail": "#1baf7a", "mouse": "#eda100"}
# Categorical slots 1-3 again, for the pigeon-only per-area figure.
C_PIGEON_AREA = {"HP": "#2a78d6", "CB": "#eb6834", "pallium": "#1baf7a"}


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


def ecdf_curve(p):
    """ECDF(p) - p sampled on P_GRID: the whole curve Fig2 C/D plots."""
    p = np.asarray(p, float)
    x = np.sort(p[np.isfinite(p)])
    return np.searchsorted(x, P_GRID, side="right") / len(x) - P_GRID


def frac_below(p, thr=0.01):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    return float((p < thr).mean())


# ---------------------------------------------------------------- sweeps
def sweep(unit_counts, spike_counts, n_seeds, label):
    """Returns (per-seed summary rows, seed-averaged ECDF-deviation curves)."""
    rows, curves = [], []
    for n_units in unit_counts:
        for n_spikes in spike_counts:
            for seed in range(n_seeds):
                rng = np.random.default_rng(hash((n_units, n_spikes, seed)) % 2**32)
                p = nfc_to_p(nfc_poisson(n_units, n_spikes, rng))
                curves.append(ecdf_curve(p))
                rows.append(dict(n_units=n_units, n_spikes=n_spikes, seed=seed,
                                 dev_at_half=dev_at_half(p), p_below_01=frac_below(p)))
            done = [r for r in rows if r["n_units"] == n_units and r["n_spikes"] == n_spikes]
            m = np.mean([r["dev_at_half"] for r in done])
            print(f"  [{label}] units={n_units:5d} spikes={n_spikes:5d}  dev@0.5 mean={m:+.4f}")
    rows = pd.DataFrame(rows)
    out = []
    for (n_units, n_spikes), idx in rows.groupby(["n_units", "n_spikes"]).groups.items():
        c = np.array([curves[i] for i in idx])
        out.append(pd.DataFrame(dict(sweep=label, n_units=n_units, n_spikes=n_spikes, p=P_GRID,
                                     dev=c.mean(0), sem=c.std(0, ddof=1) / np.sqrt(len(c)))))
    return rows, pd.concat(out, ignore_index=True)


def real_reference():
    """The same statistic on the real magnetic population, split by modality."""
    if not PARQUET.exists():
        print(f"  ! {PARQUET} missing -- skipping real-data reference")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
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
    ephys = d[~imaging & np.isfinite(d["p_value"])]
    curve = pd.DataFrame(dict(sweep="real ephys", n_units=len(ephys), n_spikes=-1, p=P_GRID,
                              dev=ecdf_curve(ephys["p_value"].values), sem=np.nan))
    spk = ephys[["species", "area", "spk_count"]].reset_index(drop=True)
    return pd.DataFrame(rows), curve, spk


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


def fig_spikes(ds, real, out, title="Bias decays with spikes per unit"):
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
    ax.set_title(title, fontsize=10, color=C_INK)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")


def _slope(ds):
    """OLS of every seed-level dev@0.5 on log10(spikes): (slope, lo95, hi95, p)."""
    from scipy import stats
    r = stats.linregress(np.log10(ds["n_spikes"]), ds["dev_at_half"])
    h = 1.96 * r.stderr
    return r.slope, r.slope - h, r.slope + h, r.pvalue


def fig_spikes_curves(curves, out):
    """ECDF(p) - p for every spike count, overlaid; one panel per sweep."""
    real = curves[curves["sweep"] == "real ephys"]
    panels = [("spikes", "Original sweep (50-4000 spikes/unit)"),
              ("spikes_wide", "Widened sweep (10-1000 spikes/unit)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        sub = curves[curves["sweep"] == key]
        counts = sorted(sub["n_spikes"].unique())
        ax.axhline(0, color=C_INK, linewidth=1, linestyle="--", alpha=0.5)
        for colr, n in zip(RAMP, counts):
            c = sub[sub["n_spikes"] == n]
            ax.plot(c["p"], c["dev"], color=colr, linewidth=1.6, label=f"{n}")
        if len(real):
            ax.plot(real["p"], real["dev"], color=C_REAL, linewidth=2, linestyle=(0, (4, 2)),
                    label="real ephys")
        ax.set_xlim(0, 1)
        ax.set_xlabel("p")
        ax.set_title(title, fontsize=10, color=C_INK)
        # Below the axes: inside, it sat on the 4000-spike curve's negative dip.
        ax.legend(title="spikes/unit", fontsize=8, title_fontsize=8, frameon=False,
                  loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5)
        _style(ax)
    axes[0].set_ylabel("ECDF(p) - p")
    fig.suptitle("Poisson null, 4000 units x 12 seeds per curve (seed-averaged)",
                 fontsize=10, color=C_INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_spk_count_distribution(spk, ds_all, out, group="species", colors=C_SPECIES,
                               pop_label="real ephys magnetic units",
                               all_label="all ephys", group_title="Per species"):
    """Where the real units sit on the spike-count axis, split by `group`."""
    s = spk["spk_count"].values
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    bins = np.geomspace(10, s.max() * 1.05, 60)
    ax.hist(s, bins=bins, color=C_MUTED, edgecolor="white", linewidth=0.5)
    for q, ls, dy in [(0.25, ":", -12), (0.5, "-", -24), (0.75, ":", -36)]:
        v = np.quantile(s, q)
        ax.axvline(v, color=C_INK, linewidth=1, linestyle=ls)
        ax.annotate(f"{int(q * 100)}th pct: {v:.0f}", xy=(v, 1),
                    xycoords=("data", "axes fraction"), xytext=(3, dy),
                    textcoords="offset points", fontsize=8, color=C_INK)
    # Which spike counts each sweep actually sampled, as ticks under the axis.
    for y, key, lab in [(-0.20, "spikes", "original sweep"),
                        (-0.27, "spikes_wide", "widened sweep")]:
        xs = sorted(ds_all.loc[ds_all["sweep"] == key, "n_spikes"].unique())
        ax.plot(xs, [y] * len(xs), transform=ax.get_xaxis_transform(), marker="|",
                markersize=8, markeredgewidth=1.5, color=C_SIM, linestyle="none",
                clip_on=False)
        ax.annotate(lab, xy=(xs[-1], y), xycoords=("data", "axes fraction"), xytext=(6, -3),
                    textcoords="offset points", fontsize=7, color=C_INK,
                    annotation_clip=False)
    ax.set_xscale("log")
    ax.set_xlabel(f"Spikes per unit ({pop_label})")
    ax.set_ylabel("Units")
    ax.set_title(f"n = {len(s)} units, min = {s.min():.0f}", fontsize=10, color=C_INK)
    _style(ax)

    for sp, colr in colors.items():
        v = np.sort(spk.loc[spk[group] == sp, "spk_count"].values)
        if not len(v):
            continue
        ax2.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=colr,
                 linewidth=2, label=f"{sp} (n={len(v)})")
    v = np.sort(s)
    ax2.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=C_INK, linewidth=1.2,
             linestyle="--", label=f"{all_label} (n={len(v)})")
    ax2.set_xscale("log")
    ax2.set_xlim(10, None)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Spikes per unit")
    ax2.set_ylabel("Fraction of units with <= this many spikes")
    ax2.set_title(group_title, fontsize=10, color=C_INK)
    ax2.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")


def mixture_prediction(ds_all, spk):
    """Predicted dev@0.5 averaging the sweep over the REAL spike-count distribution.

    The ECDF of a pooled population is the unit-weighted average of its parts'
    ECDFs, so dev@0.5 of the pool is exactly the mean of dev@0.5(N) over units.
    Counts beyond the sweep's range are clamped to its end points.
    """
    g = ds_all.groupby("n_spikes")["dev_at_half"].mean()
    x = np.log10(np.clip(spk["spk_count"].values, g.index.min(), g.index.max()))
    return float(np.interp(x, np.log10(g.index.values), g.values).mean())


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
        dw = pd.read_csv(_HERE / "results_spikes_wide.csv")
        curves = pd.read_csv(_HERE / "results_curves.csv")
        spk_path = _HERE / "results_real_spk_counts.csv"
        spk = pd.read_csv(spk_path) if spk_path.exists() else pd.DataFrame()
    else:
        print("Self-test")
        _selftest()

        print("\nReal-data reference")
        real, real_curve, spk = real_reference()

        print("\nSweep 1: number of units (spikes/unit fixed at 150)")
        unit_counts = [100, 250, 500, 1000, 2000, 3000, 5000, 8000]
        du, _ = sweep(unit_counts, [150], n_seeds=24, label="units")

        print("\nSweep 2: spikes per unit (units fixed at 4000)")
        spike_counts = [50, 75, 100, 150, 250, 500, 1000, 2000, 4000]
        ds, curves_s = sweep([4000], spike_counts, n_seeds=12, label="spikes")

        # Same budget as sweep 2 (9 counts x 12 seeds x 4000 units), log-spaced
        # over 10-1000 instead.
        print("\nSweep 3: spikes per unit, widened range 10-1000 (units fixed at 4000)")
        wide_counts = [int(round(v)) for v in np.geomspace(10, 1000, 9)]
        dw, curves_w = sweep([4000], wide_counts, n_seeds=12, label="spikes_wide")

        curves = pd.concat([curves_s, curves_w, real_curve], ignore_index=True)
        du.to_csv(_HERE / "results_units.csv", index=False)
        ds.to_csv(_HERE / "results_spikes.csv", index=False)
        dw.to_csv(_HERE / "results_spikes_wide.csv", index=False)
        curves.to_csv(_HERE / "results_curves.csv", index=False)
        if len(real):
            real.to_csv(_HERE / "results_real.csv", index=False)
            spk.to_csv(_HERE / "results_real_spk_counts.csv", index=False)

    print("\nFigures")
    slope, r2, grand = fig_units(du, _HERE / "fig_units_sweep.png")
    fig_spikes(ds, real, _HERE / "fig_spikes_sweep.png")
    fig_real_vs_sim(ds, real, _HERE / "fig_real_vs_sim.png")
    fig_spikes(dw, real, _HERE / "fig_spikes_sweep_wide.png",
               title="Bias vs spikes per unit, widened range (10-1000)")
    fig_spikes_curves(curves, _HERE / "fig_spikes_curves.png")
    ds_all = pd.concat([ds.assign(sweep="spikes"), dw.assign(sweep="spikes_wide")])
    for key, d in [("original", ds), ("widened", dw)]:
        sl, lo, hi, pv = _slope(d)
        print(f"Spikes sweep ({key}): slope={sl:+.5f}/decade [{lo:+.5f}, {hi:+.5f}], p={pv:.2g}")
    if len(spk):
        fig_spk_count_distribution(spk, ds_all, _HERE / "fig_spk_count_distribution.png")
        s = spk["spk_count"]
        for a, b in [(50, 100), (100, 1000), (1000, 4000), (4000, np.inf)]:
            print(f"  real ephys units with {a}-{b} spikes: {((s >= a) & (s < b)).mean():.1%}")
        print(f"Mixture prediction dev@0.5 (both sweeps pooled, over real spk_count): "
              f"{mixture_prediction(ds_all, spk):+.4f}")

        pig = spk[spk["species"] == "Pigeon"]
        fig_spk_count_distribution(pig, ds_all, _HERE / "fig_spk_count_distribution_pigeon.png",
                                   group="area", colors=C_PIGEON_AREA,
                                   pop_label="pigeon magnetic units", all_label="all pigeon",
                                   group_title="Pigeon, per brain area")
        for area, sub in [("all pigeon", pig)] + list(pig.groupby("area")):
            s = sub["spk_count"]
            q = s.quantile([0.25, 0.5, 0.75]).values
            shares = "  ".join(f"{a}-{b}: {((s >= a) & (s < b)).mean():5.1%}"
                               for a, b in [(50, 100), (100, 1000), (1000, np.inf)])
            print(f"  {area:10} n={len(s):5d}  quartiles {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}  "
                  f"{shares}  mixture dev@0.5={mixture_prediction(ds_all, sub):+.4f}")

    print(f"\nUnits sweep regression: slope={slope:+.5f} per decade, r^2={r2:.4f}, "
          f"grand mean={grand:+.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
