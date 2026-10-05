"""Prototype: does a guard band around the analysis frequency (Q_ignore) restore the imaging null?

    python docs/guard_band_null/guard_band.py                  # ~10 min
    python docs/guard_band_null/guard_band.py --figures-only   # replot from the CSVs

Write-up: docs/guard_band_null.md. Zebrafish only: the 0.3 Hz and 0.1 Hz sets of recordings
(floor-clipped, miscalibrated) and the 2022 Q1 0.4 Hz set (unclipped, calibrated) as a control.

NFC compares the power at the analysis bin with the mean power of M bins on each side. Here the
M noise bins on each side start G bins away instead of right next to it (G = 0 is production):

    noise bins = f0 - G - M .. f0 - G - 1   and   f0 + G + 1 .. f0 + G + M

Two nulls are compared for each G:
  nominal    the noise mean averages 2M independent ordinates: NFC^2 / 2 ~ F(2, 4M);
  effective  the noise mean averages 2 M_eff ordinates, M_eff = M / VIF, where VIF is the
             variance inflation of a mean of M correlated ordinates,
             VIF = 1 + 2 sum_{L=1}^{M-1} (1 - L/M) rho_L,
             and rho_L is the ROI's own correlation between ordinates L bins apart, measured
             on its whole spectrum away from the stimulus (each ordinate first divided by a
             running median of its neighbours, so a sloping spectrum does not count as
             correlation): NFC^2 / 2 ~ F(2, 4 M_eff).
F(2, 4M) is the exact null of NFC for independent Gaussian ordinates; at integer M it agrees
with production's eps-corrected Rayleigh null to within 0.005 in p for M >= 19 (0.011 at M = 8).

Calibration is measured where nothing was presented: at every 3rd bin from 0.05 Hz up to the
highest bin whose widest window still fits below Nyquist, skipping bins whose widest window
(G = G_MAX) would reach the stimulus bin. The same test bins are used for every G. Each ROI
contributes one p-value per test bin; dev@0.5 = ECDF(0.5) - 0.5 is computed per test bin and
averaged over test bins.

The population is the production one before the coverage threshold (P(iscell) > 0.5,
npix >= 10, inside the fish outline, flatline removal): the aim is to keep as many neurons as
possible.

Simulated traces (null true by construction; docs/nfc_finite_sample_bias/slow_variation_sim.py)
check the same thing where the answer is known, and measure the cost in power: a
stimulus-locked rate modulation is added and the share of traces with p < 0.05 is counted.

Outputs (next to this script): results_gb_*.csv, fig_gb_*.png. The machinery (test bins,
calibration, simulated traces, power) is reused at G = 0 by
docs/dispersion_matched_null/dispersion_null.py, which swaps in its own nulls.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import median_filter, uniform_filter1d

_HERE = Path(__file__).resolve().parent
_NFC = _HERE.parent / "nfc_finite_sample_bias"
sys.path.insert(0, str(_NFC))
import slow_variation as sv  # noqa: E402
import slow_variation_sim as svs  # noqa: E402
from slow_variation import P_GRID, C_INK, C_MUTED, _band, _style, ecdf_dev  # noqa: E402

SETS = ["zebrafish 0.4 Hz (2022 Q1)", "zebrafish 0.3 Hz", "zebrafish 0.1 Hz"]
GS = [0, 2, 5, 10, 15, 20]          # guard widths tried (bins each side)
G_MAX = max(GS)
F_MIN = 0.05                         # lowest test frequency (Hz)
TEST_STEP = 3                        # every 3rd bin
MAX_LAG = 40                         # ordinate correlations measured up to this lag
MED_WIN = 51                         # running-median width for normalising ordinates (bins)
N_SIM = 4000                         # simulated traces per condition
SIM_DEPTHS = [0.0, 2.0, 3.0]         # log-rate SD of the tau = 100 s modulation (0 = stationary)
MOD_DEPTHS = [0.0, 0.1, 0.2, 0.3, 0.5]   # stimulus-locked rate modulation, for power
CONFIGS = {"zebrafish 0.3 Hz": (0.3, 1260, 0.15), "zebrafish 0.1 Hz": (0.1, 1080, 0.30)}

C_G = {0: "#b0b0b0", 2: "#a6c8f0", 5: "#6fa3e3", 10: "#2a78d6", 15: "#1c5cab", 20: "#0d366b"}
C_NOM, C_EFF = "#eb6834", "#2a78d6"


# ------------------------------------------------------------------ method
def noise_bins(f0, M, G):
    return np.r_[f0 - G - M:f0 - G, f0 + G + 1:f0 + G + M + 1]


def nfc(Y, f0, M, G):
    off = Y[:, noise_bins(f0, M, G)]
    return np.abs(Y[:, f0]) / np.sqrt(0.5 * np.mean(np.abs(off) ** 2, axis=1))


def p_from_nfc(x, M_eff):
    """Exact null for a noise mean of 2 M_eff independent exponential ordinates."""
    return stats.f.sf(x ** 2 / 2, 2, 4 * np.asarray(M_eff, float))


def ordinate_corr(Y, lo, hi, exclude):
    """(n_roi, MAX_LAG) correlation of normalised ordinates L bins apart, over bins lo..hi
    minus the `exclude` bins (the stimulus and its window)."""
    I = np.abs(Y[:, lo:hi]) ** 2
    I = I / np.maximum(median_filter(I, size=(1, MED_WIN), mode="nearest"), 1e-30)
    ok = np.ones(hi - lo, bool)
    ok[np.clip(np.asarray(exclude) - lo, 0, hi - lo - 1)] = False
    rho = np.zeros((len(Y), MAX_LAG))
    for L in range(1, MAX_LAG + 1):
        both = ok[:-L] & ok[L:]
        a, b = I[:, :-L][:, both], I[:, L:][:, both]
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        rho[:, L - 1] = (a * b).mean(1) / np.sqrt((a * a).mean(1) * (b * b).mean(1))
    return np.nan_to_num(rho)                     # degenerate spectrum: no correlation


def ordinate_shape(Y, lo, hi, exclude):
    """Per ROI, k = 1 / CV^2 of its periodogram ordinates over bins lo..hi minus `exclude`,
    each ordinate divided by the running mean of its MED_WIN neighbours (so the spectrum's
    shape does not count as spread). Gaussian Fourier coefficients give exponential ordinates,
    CV^2 = 1, k = 1. A trace whose power comes from a few events gives CV^2 < 1."""
    I = np.abs(Y[:, lo:hi]) ** 2
    I = I / np.maximum(uniform_filter1d(I, MED_WIN, axis=1, mode="nearest"), 1e-30)
    ok = np.ones(hi - lo, bool)
    ok[np.clip(np.asarray(exclude) - lo, 0, hi - lo - 1)] = False
    I = I[:, ok]
    cv2 = I.var(1) / I.mean(1) ** 2
    cv2 = np.where(np.isfinite(cv2), cv2, 1.0)    # degenerate spectrum: fall back to k = 1
    return 1.0 / np.clip(cv2, 0.02, None), cv2


def nulls(x, M, me, k):
    """p-values of NFC values x under the nulls compared here. `k` (the ROI's ordinate shape)
    is unused in this report; docs/dispersion_matched_null/dispersion_null.py replaces this
    function to compare the dispersion-matched null instead."""
    return (("nominal", p_from_nfc(x, M)), ("effective", p_from_nfc(x, me)))


def m_eff(rho, M):
    L = np.arange(1, M)
    r = rho[:, :M - 1] if M - 1 <= rho.shape[1] else np.pad(rho, ((0, 0), (0, M - 1 - rho.shape[1])))
    vif = 1 + 2 * (r * (1 - L / M)).sum(1)
    return M / np.maximum(vif, 1.0)


def test_bins(n_bins, f0, M, res):
    lo = max(int(np.ceil(F_MIN / res)), G_MAX + M + 1)
    hi = n_bins - 1 - G_MAX - M
    k = np.arange(lo, hi, TEST_STEP)
    return k[np.abs(k - f0) > G_MAX + M + 1]


def calibrate(Y, f0, M, res, rho, k_shape):
    """Per test bin and G: dev@0.5 under the nominal and the effective null, plus the pooled
    p-values (for ECDF curves), plus the numerator/noise coupling."""
    ks = test_bins(Y.shape[1], f0, M, res)
    out, pooled = [], {}
    for G in GS:
        me = m_eff(rho, M)
        for k in ks:
            x = nfc(Y, k, M, G)
            for null, p in nulls(x, M, me, k_shape):
                p = p[np.isfinite(p)]
                out.append(dict(G=G, k=k, null=null, n=len(p), dev=float(np.mean(p <= 0.5) - 0.5)))
                pooled.setdefault((G, null), []).append(p)
    return pd.DataFrame(out), {key: np.concatenate(v) for key, v in pooled.items()}, ks


def coupling(Y, ks, M, G):
    """Mean over ROIs of the within-ROI correlation, across test bins, between the analysis
    ordinate and its noise mean: the link the guard band is meant to cut."""
    I = np.abs(Y) ** 2
    I = I / np.maximum(uniform_filter1d(I, MED_WIN, axis=1, mode="nearest"), 1e-30)
    num = np.stack([I[:, k] for k in ks], 1)
    den = np.stack([np.mean(I[:, noise_bins(k, M, G)], 1) for k in ks], 1)
    num, den = np.log(num + 1e-30), np.log(den)
    num -= num.mean(1, keepdims=True)
    den -= den.mean(1, keepdims=True)
    r = (num * den).mean(1) / np.sqrt((num ** 2).mean(1) * (den ** 2).mean(1))
    return float(np.nanmean(r))


# ------------------------------------------------------------------ real data
def run_real():
    recs, _ = sv.pool()
    rows, curves, coup, meff, stim = [], [], [], [], []
    pooled_all = {}
    for name, batch, _ in recs:
        if batch not in SETS:
            continue
        cfg, F, _ = sv.load(name, outline=True)
        T, N = cfg.sample_period, F.shape[1]
        f0, M = sv.window_bins(N, T, cfg.analysis.f, cfg.analysis.Q_frac)
        res = 1.0 / (N * T)
        Y = np.fft.rfft(F - F.mean(axis=1, keepdims=True), axis=1)
        lo = max(int(np.ceil(F_MIN / res)) - MED_WIN, 1)
        excl = np.arange(f0 - G_MAX - M, f0 + G_MAX + M + 1)
        rho = ordinate_corr(Y, lo, Y.shape[1] - 1, excl)
        k_shape, cv2 = ordinate_shape(Y, lo, Y.shape[1] - 1, excl)
        d, pooled, ks = calibrate(Y, f0, M, res, rho, k_shape)
        rows.append(d.assign(experiment=name, batch=batch, M=M))
        for key, p in pooled.items():
            pooled_all.setdefault((batch,) + key, []).append(p)
        for G in GS:
            coup.append(dict(experiment=name, batch=batch, G=G, n_roi=len(F),
                             r=coupling(Y, ks, M, G)))
        P0 = np.stack([p_from_nfc(nfc(Y, k, M, 0), M) for k in ks], 1)
        meff.append(pd.DataFrame(dict(experiment=name, batch=batch, M=M, M_eff=m_eff(rho, M),
                                      cv2=cv2, dev_roi=np.mean(P0 <= 0.5, 1) - 0.5,
                                      **{f"rho{L + 1}": rho[:, L] for L in range(MAX_LAG)})))
        for G in GS:                                   # read-out at the stimulus frequency
            x = nfc(Y, f0, M, G)
            for null, p in nulls(x, M, m_eff(rho, M), k_shape):
                stim.append(pd.DataFrame(dict(experiment=name, batch=batch, G=G, null=null, p=p)))
        print(f"  {name}: {len(F)} ROIs, M={M}, {len(ks)} test bins, "
              f"median M_eff={np.median(m_eff(rho, M)):.1f}", flush=True)
    for (batch, G, null), ps in pooled_all.items():
        p = np.concatenate(ps)
        curves.append(dict(source="real", batch=batch, G=G, null=null, n=len(p),
                           **{f"c{i}": v for i, v in enumerate(ecdf_dev(p))}))
    return (pd.concat(rows), pd.DataFrame(curves), pd.DataFrame(coup), pd.concat(meff),
            pd.concat(stim))


# ------------------------------------------------------------------ simulation
def sim_traces(depth, N, T, f, mod, rng):
    """Floor-clipped GCaMP traces (slow_variation_sim's model) with tau = 100 s log-rate
    modulation of SD `depth`, times an optional stimulus-locked rate modulation
    (1 + mod sin 2 pi f t)."""
    k = np.exp(-np.arange(0, 10 * svs.TAU_CA, T) / svs.TAU_CA)
    lam = svs.RATE * np.exp(depth * svs.ou(N_SIM, N, T, 100.0, rng) - depth ** 2 / 2)
    lam = lam * (1 + mod * np.sin(2 * np.pi * f * np.arange(N) * T))
    spikes = rng.poisson(lam * T).astype(float)
    ca = np.array([np.convolve(s, k)[:N] for s in spikes])
    F = np.maximum(ca + svs.NOISE_SD * rng.standard_normal((N_SIM, N)) - svs.CLIP, 0.0)
    return F[F.std(axis=1) > 0]


def run_sim():
    rng = np.random.default_rng(1)
    cal, power, curves = [], [], []
    for batch, (f, N, Q_frac) in CONFIGS.items():
        T = 1.0
        f0, M = sv.window_bins(N, T, f, Q_frac)
        res = 1.0 / (N * T)
        for depth in SIM_DEPTHS:
            for mod in MOD_DEPTHS:
                F = sim_traces(depth, N, T, f, mod, rng)
                Y = np.fft.rfft(F - F.mean(axis=1, keepdims=True), axis=1)
                lo = max(int(np.ceil(F_MIN / res)) - MED_WIN, 1)
                excl = np.arange(f0 - G_MAX - M, f0 + G_MAX + M + 1)
                rho = ordinate_corr(Y, lo, Y.shape[1] - 1, excl)
                k_shape, _ = ordinate_shape(Y, lo, Y.shape[1] - 1, excl)
                me = m_eff(rho, M)
                if mod == 0:
                    d, pooled, _ = calibrate(Y, f0, M, res, rho, k_shape)
                    cal.append(d.assign(batch=batch, depth=depth))
                    for (G, null), p in pooled.items():
                        curves.append(dict(source="simulated", batch=batch, depth=depth, G=G,
                                           null=null, n=len(p),
                                           **{f"c{i}": v for i, v in enumerate(ecdf_dev(p))}))
                for G in GS:
                    x = nfc(Y, f0, M, G)
                    for null, p in nulls(x, M, me, k_shape):
                        power.append(dict(batch=batch, depth=depth, mod=mod, G=G, null=null,
                                          n=len(p), detected=float(np.mean(p < 0.05)),
                                          dev=float(np.mean(p <= 0.5) - 0.5)))
                print(f"  sim {batch} depth={depth} mod={mod}: {len(F)} traces, "
                      f"median M_eff={np.median(me):.1f} (M={M})", flush=True)
    return pd.concat(cal), pd.DataFrame(power), pd.DataFrame(curves)


# ------------------------------------------------------------------ figures
def _summary(d):
    """Mean over test bins of dev@0.5, per set of recordings, G and null."""
    g = d.groupby(["batch", "G", "null"])
    return g.agg(dev=("dev", "mean"), n=("n", "median"), n_bins=("dev", "size")).reset_index()


C_SETS = {"zebrafish 0.4 Hz (2022 Q1)": "#8a8a8a", "zebrafish 0.3 Hz": "#2a78d6",
          "zebrafish 0.1 Hz": "#eb6834"}
NULL_STYLE = {"nominal": ("#eb6834", "-", "nominal null, F(2, 4M)"),
              "effective": ("#8a8a8a", ":", "effective-M null, F(2, 4 M_eff)")}
SHOW = [(0, "#8a8a8a", "-", "production (G 0)"),
        (10, "#2a78d6", "-", "G 10"),
        (20, "#0d366b", "--", "G 20")]


def fig_diagnose(meff, out):
    """Figure 1: how far apart must two bins be to be independent, and what does the
    correlation do to the effective number of noise bins?"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for b, c in C_SETS.items():
        r = meff.loc[meff["batch"] == b, [f"rho{L + 1}" for L in range(MAX_LAG)]].mean().values
        ax.plot(np.arange(1, MAX_LAG + 1), r, color=c, lw=2, label=b)
    ax.axhline(0, color=C_INK, lw=0.6)
    ax.set_xlabel("distance between periodogram bins (bins)", fontsize=8)
    ax.set_ylabel("correlation of normalised ordinates", fontsize=8)
    ax.set_title("A. Bins closer than ~10 apart are correlated", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    ax = axes[1]
    for b, c in C_SETS.items():
        d = meff[meff["batch"] == b]
        x = np.sort((d["M_eff"] / d["M"]).values)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=2, label=b)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("M_eff / M (1 = noise bins independent)", fontsize=8)
    ax.set_ylabel("ECDF over ROI traces", fontsize=8)
    ax.set_title("B. Effective share of independent noise bins", fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_calibration(real, sim, out):
    """Figure 2: dev@0.5 where nothing was presented, vs guard band, for each null."""
    s_real = _summary(real)
    s_sim = _summary(sim.assign(batch=sim["batch"] + ", depth " + sim["depth"].astype(str)))
    panels = [(s_real, b, f"real: {b}") for b in SETS] + \
             [(s_sim, f"{b}, depth {d}", f"simulated {b.split()[1]} Hz config, log-rate SD {d:g}"
               + (" (stationary)" if d == 0 else ""))
              for b in CONFIGS for d in SIM_DEPTHS]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False)
    for ax, (s, key, title) in zip(axes.flat, panels):
        d = s[s["batch"] == key]
        for null, (c, ls, lab) in NULL_STYLE.items():
            e = d[d["null"] == null]
            ax.plot(e["G"], e["dev"], color=c, ls=ls, lw=2, marker="o", ms=3.5, label=lab)
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("guard band G (bins each side)", fontsize=8)
        ax.set_ylabel("dev@0.5, mean over test freqs", fontsize=7)
        _style(ax)
    axes[0, 0].legend(fontsize=6.5, frameon=False)
    fig.suptitle("Calibration where nothing was presented (0 = calibrated). Production = G 0, "
                 "nominal null", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_curves(curves, out):
    """Figure 3: the whole p-value distribution where nothing was presented."""
    real = curves[(curves["source"] == "real") & (curves["null"] == "nominal")]
    fig, axes = plt.subplots(1, len(SETS), figsize=(4.4 * len(SETS), 3.6), sharey=True)
    for ax, b in zip(axes, SETS):
        d = real[real["batch"] == b]
        for G, c, ls, lab in SHOW:
            row = d[d["G"] == G].iloc[0]
            y = row[[f"c{i}" for i in range(len(P_GRID))]].values.astype(float)
            ax.plot(P_GRID, y, color=c, ls=ls, lw=1.8, label=lab)
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_title(b, fontsize=9)
        ax.set_xlabel("p", fontsize=8)
        _style(ax)
    axes[0].set_ylabel("ECDF(p) - p, pooled over test freqs", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("p-value distribution where nothing was presented, nominal null "
                 "(flat at 0 = calibrated)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_power(power, out):
    """Figure 4: detection of a known stimulus-locked modulation in simulated traces."""
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(5.6 * len(CONFIGS), 3.9), squeeze=False)
    for ax, b in zip(axes[0], CONFIGS):
        d = power[(power["batch"] == b) & (power["depth"] == 2.0) & (power["null"] == "nominal")]
        for G, c, ls, lab in SHOW:
            e = d[d["G"] == G].sort_values("mod")
            ax.plot(e["mod"], e["detected"], color=c, ls=ls, lw=2, marker="o", ms=4, label=lab)
        ax.axhline(0.05, color=C_INK, lw=0.6, ls=":")
        ax.set_xlabel("stimulus-locked modulation depth m: rate x (1 + m sin 2 pi f t)", fontsize=8)
        ax.set_ylabel("share of traces with p < 0.05", fontsize=8)
        ax.set_title(f"simulated {b.split()[1]} Hz config, floor-clipped, log-rate SD 2", fontsize=9)
        _style(ax)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Power, nominal null. At m = 0 the share should be 0.05 (dotted)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_stimulus(stim, out):
    """Figure 5: read-out at the stimulus frequency."""
    fig, axes = plt.subplots(1, len(SETS), figsize=(4.4 * len(SETS), 3.6), sharey=True)
    for ax, b in zip(axes, SETS):
        d = stim[(stim["batch"] == b) & (stim["null"] == "nominal")]
        for G, c, ls, lab in SHOW:
            p = d[d["G"] == G]["p"].values
            ax.plot(P_GRID, ecdf_dev(p), color=c, ls=ls, lw=1.8,
                    label=f"{lab}: dev@0.5 {np.mean(p <= 0.5) - 0.5:+.3f}")
        _band(ax, len(p))
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_title(f"{b} (n = {len(p)} ROI traces)", fontsize=9)
        ax.set_xlabel("p", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False, loc="lower center")
        _style(ax)
    axes[0].set_ylabel("ECDF(p) - p at the stimulus frequency", fontsize=8)
    fig.suptitle("Read-out at the stimulus frequency, nominal null. Gray: 95% binomial band",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def figures(real, coup, meff, stim, sim, power, curves):
    fig_diagnose(meff, _HERE / "fig_gb_diagnose.png")
    fig_calibration(real, sim, _HERE / "fig_gb_calibration.png")
    fig_curves(curves, _HERE / "fig_gb_curves.png")
    fig_power(power, _HERE / "fig_gb_power.png")
    fig_stimulus(stim, _HERE / "fig_gb_stimulus.png")
    print("  wrote fig_gb_*.png")


def main():
    names = ["real", "coupling", "meff", "stim", "sim", "power", "curves"]
    if "--figures-only" in sys.argv:
        figures(*[pd.read_csv(_HERE / f"results_gb_{n}.csv") for n in names])
        return
    real, curves_real, coup, meff, stim = run_real()
    sim, power, curves_sim = run_sim()
    curves = pd.concat([curves_real, curves_sim], ignore_index=True)
    for n, d in zip(names, [real, coup, meff, stim, sim, power, curves]):
        d.to_csv(_HERE / f"results_gb_{n}.csv", index=False)
    print(_summary(real).round(4).to_string(index=False))
    print(_summary(sim.assign(batch=sim["batch"] + " d" + sim["depth"].astype(str))).round(4).to_string(index=False))
    print(power[power["depth"] == 2.0].round(3).to_string(index=False))
    figures(real, coup, meff, stim, sim, power, curves)


if __name__ == "__main__":
    main()
