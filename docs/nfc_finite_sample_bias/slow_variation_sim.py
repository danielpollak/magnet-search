"""Simulated slow variation: what does each kind do to the NFC p-value ECDF?

    python docs/nfc_finite_sample_bias/slow_variation_sim.py                 # ~3 min
    python docs/nfc_finite_sample_bias/slow_variation_sim.py --figures-only

The advisor's proposed test, plus imaging analogs. Everything is a valid null
by construction: nothing is locked to the analysis frequency, so any departure
of the p-value ECDF from uniform is a property of the null's assumptions, not
a signal.

Ephys (simulate.py's setup: 3 Hz, T = 300 s, Q = 27, ~134 spikes/unit)
    Poisson spike trains whose rate is lambda0 * exp(s*x(t) - s^2/2), x an
    Ornstein-Uhlenbeck process (unit variance, timescale tau). tau = 0 is
    the homogeneous Poisson baseline simulate.py already characterised.

Imaging (one config per batch: f, frame count, Q_frac as in production)
    F(t) = (spikes * exp(-t / TAU_CA))(t) + noise(t), spikes Poisson per frame.
    event rate (stationary)   constant rate, swept from ~3 to ~300 events/trace
    rate x OU(tau)            firing rate slowly modulated (the advisor's model)
    noise SD x OU(tau)        measurement-noise level slowly modulated
    + additive drift          a large slow OU added to F (bleaching / baseline wander)
    floor-clipped, rate x OU  max(F - CLIP, 0): the 2022-09/10 zebrafish and
                              medaka traces sit at one floor value for ~90-98%
                              of frames; CLIP = 2 noise SD reproduces that.
                              tau = 0 is the stationary clipped trace.
    floor-clipped, depth      the same at tau = 100 s, log-rate SD swept 0..3:
                              at SD 3 activity is confined to epochs, as in
                              many of the real 0.1/0.3 Hz traces.

Outputs: results_sv_sim_curves.csv, results_sv_sim_coupling.csv,
fig_sv_sim_ephys.png, fig_sv_sim_imaging.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import simulate as sim  # noqa: E402
from slow_variation import (P_GRID, MAX_LAG, C_INK, _band, _style, dev_half, ecdf_dev,  # noqa: E402
                            lag_corr, nfc_from_fft, pvals, window_bins)

N_UNITS = 16000                  # binomial SE of dev@0.5 = 0.004
S_MOD = 1.0                      # OU modulation depth (log-rate SD)
TAUS = [3.0, 10.0, 30.0, 100.0, 300.0]
TAU_CA = 2.0                     # GCaMP decay (s)
RATE = 0.1                       # imaging spikes/s (sparse, transient-dominated traces)
NOISE_SD = 0.5                   # measurement noise, in units of one spike's peak dF
RATES = [0.003, 0.01, 0.03, 0.1, 0.3]   # stationary event-rate sweep (spikes/s)
CLIP = 2 * NOISE_SD
IMAGING_CONFIGS = [("0.4 Hz config (N=1194, Q_frac=0.15)", 0.4, 1194, 0.15),
                   ("0.3 Hz config (N=1260, Q_frac=0.15)", 0.3, 1260, 0.15),
                   ("0.1 Hz config (N=1080, Q_frac=0.30)", 0.1, 1080, 0.30)]
TAU_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


def ou(n_units, n, dt, tau, rng):
    """Unit-variance OU sampled every dt (exact AR(1) discretisation)."""
    a = np.exp(-dt / tau)
    x = np.empty((n_units, n))
    x[:, 0] = rng.standard_normal(n_units)
    e = rng.standard_normal((n_units, n)) * np.sqrt(1 - a * a)
    for i in range(1, n):
        x[:, i] = a * x[:, i - 1] + e[:, i]
    return x


# ------------------------------------------------------------------ ephys
def ephys_nfc(tau, rng, mean_spikes=134, chunk=2000):
    """NFC for rate-modulated Poisson trains; same estimator as simulate.nfc_poisson."""
    dt = 0.05
    n = int(sim.T / dt)
    f0, ff_alt = sim._freq_grid()
    t_bins = np.arange(n) * dt
    nfc, n_spk = [], []
    for start in range(0, N_UNITS, chunk):       # chunked: (units x bins) arrays get large
        m = min(chunk, N_UNITS - start)
        if tau == 0:
            lam = np.full((m, n), mean_spikes / sim.T)
        else:
            lam = (mean_spikes / sim.T) * np.exp(S_MOD * ou(m, n, dt, tau, rng) - S_MOD ** 2 / 2)
        counts = rng.poisson(lam * dt)
        for u in range(m):
            # spike times: bin start + uniform jitter within the bin
            idx = np.repeat(np.arange(n), counts[u])
            t = t_bins[idx] + rng.random(len(idx)) * dt

            def coeff(f):
                ang = (-2 * np.pi * f) * t
                return (np.cos(ang).sum() + 1j * np.sin(ang).sum()) / sim.T

            acc = sum(abs(coeff(f)) ** 2 for f in ff_alt)
            nfc.append(abs(coeff(f0)) / np.sqrt(0.5 * acc / len(ff_alt)))
        n_spk.append(counts.sum(1))
    return np.array(nfc), np.concatenate(n_spk)


# ------------------------------------------------------------------ imaging
IMAGING_KINDS = ["event rate (stationary)", "rate x OU", "noise SD x OU", "+ additive drift",
                 "floor-clipped, rate x OU", "floor-clipped, depth (tau 100 s)"]
DEPTHS = [0.0, 1.0, 2.0, 3.0]


def imaging_traces(kind, x, N, T, rng):
    """x is the event rate for the stationary sweep, the OU timescale otherwise."""
    k = np.exp(-np.arange(0, 10 * TAU_CA, T) / TAU_CA)
    rate = x if kind == "event rate (stationary)" else RATE
    if kind in ("rate x OU", "floor-clipped, rate x OU") and x > 0:
        lam = rate * np.exp(S_MOD * ou(N_UNITS, N, T, x, rng) - S_MOD ** 2 / 2)
    elif kind == "floor-clipped, depth (tau 100 s)":
        lam = rate * np.exp(x * ou(N_UNITS, N, T, 100.0, rng) - x ** 2 / 2)
    else:
        lam = np.full((N_UNITS, N), rate)
    spikes = rng.poisson(lam * T).astype(float)
    ca = np.array([np.convolve(s, k)[:N] for s in spikes])
    noise = NOISE_SD * rng.standard_normal((N_UNITS, N))
    if kind == "noise SD x OU":
        noise *= np.exp(S_MOD * ou(N_UNITS, N, T, x, rng) - S_MOD ** 2)
    F = ca + noise
    if kind == "+ additive drift":
        F += 5 * ca.std() * ou(N_UNITS, N, T, x, rng)
    if kind.startswith("floor-clipped"):
        F = np.maximum(F - CLIP, 0.0)
        # a trace with no frame above the floor is dropped by remove_flatlines in production
        F = F[F.std(axis=1) > 0]
    return F


def _params(kind):
    if kind == "event rate (stationary)":
        return RATES
    if kind == "floor-clipped, rate x OU":
        return [0.0] + TAUS
    if kind.startswith("floor-clipped, depth"):
        return DEPTHS
    return TAUS


def run():
    rng = np.random.default_rng(0)
    curves, coup = [], []

    print("ephys")
    for tau in [0.0] + TAUS:
        nfc, n_spk = ephys_nfc(tau, rng)
        p = sim.nfc_to_p(nfc)
        curves.append(dict(modality="ephys", config="3 Hz, T=300 s, Q=27", kind="rate x OU",
                           x=tau, n=len(p), dev_at_half=dev_half(p),
                           p_below_01=float(np.mean(p < 0.01)), median_spikes=float(np.median(n_spk)),
                           **{f"c{i}": v for i, v in enumerate(ecdf_dev(p))}))
        print(f"  tau={tau:5.0f}  dev@0.5={curves[-1]['dev_at_half']:+.4f}  "
              f"P(p<.01)={curves[-1]['p_below_01']:.4f}", flush=True)

    print("imaging")
    for label, f, N, Q_frac in IMAGING_CONFIGS:
        T = 1.0
        f0, M = window_bins(N, T, f, Q_frac)
        for kind in IMAGING_KINDS:
            for x in _params(kind):
                F = imaging_traces(kind, x, N, T, rng)
                Y = np.fft.rfft(F - F.mean(axis=1, keepdims=True), axis=1)
                p = pvals(nfc_from_fft(Y, f0, M), M)
                floor = float(np.median(np.mean(F == F.min(axis=1, keepdims=True), axis=1)))
                curves.append(dict(modality="imaging", config=label, kind=kind, x=x, n=len(p),
                                   dev_at_half=dev_half(p), p_below_01=float(np.mean(p < 0.01)),
                                   floor_frac=floor,
                                   **{f"c{i}": v for i, v in enumerate(ecdf_dev(p))}))
                lc = np.nanmean(lag_corr(np.abs(Y[:, f0 - M:f0 + M + 1]) ** 2), axis=0)
                coup.append(dict(config=label, kind=kind, x=x,
                                 **{f"lag{L + 1}": v for L, v in enumerate(lc)}))
                print(f"  {label[:6]} {kind:26} x={x:7.3f}  dev@0.5={curves[-1]['dev_at_half']:+.4f}  "
                      f"lag1={lc[0]:+.3f}  floor={floor:.2f}", flush=True)
    return pd.DataFrame(curves), pd.DataFrame(coup)


# ------------------------------------------------------------------ figures
def _curve(row):
    return row[[f"c{i}" for i in range(len(P_GRID))]].values.astype(float)


def _xlabel(kind, x):
    if kind == "event rate (stationary)":
        return f"{x * 1000:.0f} events/1000 s"
    if kind.startswith("floor-clipped, depth"):
        return f"log-rate SD {x:g}"
    return "stationary" if x == 0 else f"tau {x:.0f} s"


def fig_ephys(curves, out):
    d = curves[curves["modality"] == "ephys"]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    _band(ax, N_UNITS)
    for i, (_, r) in enumerate(d.iterrows()):
        col = "#b0b0b0" if r["x"] == 0 else TAU_RAMP[i - 1]
        lab = "homogeneous Poisson" if r["x"] == 0 else f"rate x OU, tau = {r['x']:.0f} s"
        ax.plot(P_GRID, _curve(r), color=col, lw=1.5, label=f"{lab}  ({r['dev_at_half']:+.3f})")
    ax.axhline(0, color=C_INK, lw=0.6)
    ax.set_xlabel("p", fontsize=8)
    ax.set_ylabel("ECDF(p) - p", fontsize=8)
    ax.set_title(f"Ephys: slowly rate-modulated Poisson ({N_UNITS} units, log-rate SD {S_MOD:g}, "
                 f"3 Hz, Q=27)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower center")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_imaging(curves, coup, out):
    d = curves[curves["modality"] == "imaging"]
    cfgs = [c[0] for c in IMAGING_CONFIGS]
    nr = len(IMAGING_KINDS) + 1
    fig, axes = plt.subplots(nr, len(cfgs), figsize=(3.6 * len(cfgs), 2.6 * nr), squeeze=False,
                             sharey="row")
    for c, cfg in enumerate(cfgs):
        for r, kind in enumerate(IMAGING_KINDS):
            ax = axes[r, c]
            _band(ax, N_UNITS)
            rows = d[(d["config"] == cfg) & (d["kind"] == kind)].sort_values("x")
            n_zero = int((rows["x"] == 0).sum())
            for i, (_, row) in enumerate(rows.iterrows()):
                col = "#b0b0b0" if row["x"] == 0 else TAU_RAMP[i - n_zero]
                extra = f", floor {row['floor_frac']:.2f}" if kind.startswith("floor") else ""
                ax.plot(P_GRID, _curve(row), color=col, lw=1.3,
                        label=f"{_xlabel(kind, row['x'])} ({row['dev_at_half']:+.3f}{extra})")
            ax.axhline(0, color=C_INK, lw=0.6)
            ax.set_title(f"{cfg}\n{kind}" if r == 0 else kind, fontsize=7)
            ax.legend(fontsize=5.5, frameon=False, loc="lower center")
            if c == 0:
                ax.set_ylabel("ECDF(p) - p", fontsize=7)
            _style(ax)
        ax = axes[-1, c]
        lags = np.arange(1, MAX_LAG + 1)
        g = coup[coup["config"] == cfg]
        cols = [f"lag{L}" for L in lags]
        ax.plot(lags, g[(g["kind"] == "event rate (stationary)") & np.isclose(g["x"], RATE)][cols].values[0],
                color="#b0b0b0", lw=1.5, marker="o", ms=2.5, label="stationary")
        for kind, ls in zip(IMAGING_KINDS[1:5], ("-", "--", ":", "-.")):
            row = g[(g["kind"] == kind) & (g["x"] == 100.0)]
            ax.plot(lags, row[cols].values[0], color=TAU_RAMP[3], ls=ls, lw=1.3,
                    label=f"{kind}, tau 100 s")
        row = g[(g["kind"] == IMAGING_KINDS[5]) & (g["x"] == 3.0)]
        ax.plot(lags, row[cols].values[0], color="#e8590c", lw=1.5,
                label="floor-clipped, tau 100 s, log-rate SD 3")
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_xlabel("lag (bins)", fontsize=7)
        if c == 0:
            ax.set_ylabel("within-trace corr(I_k, I_k+lag)", fontsize=7)
        ax.legend(fontsize=5.5, frameon=False)
        _style(ax)
    fig.suptitle(f"Imaging: simulated GCaMP ({N_UNITS} traces/condition; {RATE} spikes/s unless swept, "
                 f"decay {TAU_CA:g} s, noise SD {NOISE_SD}; OU log-SD {S_MOD:g}); gray band = 95% binomial",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    if "--figures-only" in sys.argv:
        curves = pd.read_csv(_HERE / "results_sv_sim_curves.csv")
        coup = pd.read_csv(_HERE / "results_sv_sim_coupling.csv")
    else:
        curves, coup = run()
        curves.to_csv(_HERE / "results_sv_sim_curves.csv", index=False)
        coup.to_csv(_HERE / "results_sv_sim_coupling.csv", index=False)
    fig_ephys(curves, _HERE / "fig_sv_sim_ephys.png")
    fig_imaging(curves, coup, _HERE / "fig_sv_sim_imaging.png")
    print("  wrote fig_sv_sim_*.png")


if __name__ == "__main__":
    main()
