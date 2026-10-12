"""Nonstationarity of the imaging traces: do a ROI's statistics change within a recording more
than a stationary trace's would?

    python docs/nonstationarity/nonstationarity.py --workers 24     # ~minutes
    python docs/nonstationarity/nonstationarity.py --figures-only

Write-up: docs/nonstationarity.md.

Every trace is cut into K = 5, 10 and 20 equal chunks, each a whole number of cycles of the
recording's analysis frequency f, and nested: a K = 5 chunk is exactly two K = 10 chunks and four
K = 20 chunks, so all three resolutions use the same frames (chunk_length; exact when T = 1 s).
The frames after the last chunk are dropped. In the 2022 Q1 recordings (T = 1.0212 s) a cycle is
not a whole number of frames, and the leftover (<= 0.07 cycles) is removed by shifting the chunk's
Fourier grid, as production fit_Fourier(align="shift") does (docs/fourier_grid_shift.md), so a
coefficient sits exactly on f.
In each chunk:
  cv2   CV^2 of the chunk's periodogram ordinates from 0.05 Hz to just below Nyquist, stimulus
        coefficient excluded, each ordinate divided by that ROI's mean ordinate at the same
        frequency over the K chunks (so a coloured spectrum doesn't inflate it). 1 for Gaussian
        noise.
  rms   RMS about the whole trace's mean (level and amplitude together).
  band  mean periodogram power in the NFC reference window: the M = round(Q_frac * cycles)
        coefficients adjacent to the stimulus coefficient on each side (production's
        bins_for_fraction at the chunk's resolution, without its minimum-bins floor), stimulus
        coefficient excluded. NaN if M = 0.

Run from a git worktree with NS_DATA_DIR=<main checkout>/data (the NWB files are untracked).
Across the K chunks the dispersion is std / mean (scale-free). Its null comes from 1,000
replicates of each of two shuffles of the whole trace, one permutation per replicate shared by
all ROIs of a recording:
  full   every frame permuted (tests "independent frames", not only "stationary");
  block  contiguous 10 s blocks permuted (keeps short-range autocorrelation and the marginal
         distribution, destroys slower structure).
p = (1 + #null >= observed) / 1001.

Control: up to 20 ROIs per recording are replaced by stationary Gaussian AR(1) traces with the
same lag-1 autocorrelation and run through the same procedure (rows with synthetic=True).

Population: all 50 imaging recordings (experiments/engert_*.yml, medaka_*.yml), production ROIs
(P(iscell), npix, body outline, flatline removal; no coverage threshold), full-resolution traces.

Outputs (next to this script): results_nonstationarity.csv, results_nonstationarity_examples.npz
(gitignored), fig_ns_*.png.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import re
import sys
import zlib
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.fft as sfft
from scipy import stats
from scipy.signal import lfilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE.parent / "nfc_finite_sample_bias"))
import slow_variation as sv  # noqa: E402
from slow_variation import C_INK, C_MUTED, _style  # noqa: E402

KS = (5, 10, 20)
METRICS = [("cv2", "CV² of whitened ordinates"), ("rms", "RMS about the trace mean"),
           ("band", "NFC reference-window power")]
NULLS = ("full", "block")
N_REP = 1000
BATCH = 25
BLOCK_S = 10.0
F_MIN = 0.05
CYCLE_TOL = 0.07     # cycles; the leftover is then removed exactly by the grid shift
N_SYNTH = 20
ALPHA = 0.05
BATCHES = ["zebrafish 0.4 Hz (2022 Q1)", "zebrafish 0.3 Hz", "zebrafish 0.1 Hz", "medaka 0.1 Hz"]
SHORT = {BATCHES[0]: "zfish 0.4 Hz\n(2022 Q1)", BATCHES[1]: "zfish 0.3 Hz",
         BATCHES[2]: "zfish 0.1 Hz", BATCHES[3]: "medaka 0.1 Hz"}
EXAMPLES = ["engert_20220221_magnet", "engert_20220914_fish2_magneto_0",
            "engert_20221001_fish2_magneto_0", "medaka_fish3_8dpf_magneto_0"]
C_K = {5: "#0d366b", 10: "#3987e5", 20: "#86b6ef"}     # ordinal blue ramp, coarse = dark
C_FULL = "#b0b0b0"
OUT_CSV = _HERE / "results_nonstationarity.csv"
OUT_NPZ = _HERE / "results_nonstationarity_examples.npz"
# NS_DATA_DIR: read the NWB files and the parquet from another checkout's data/ (they are
# untracked, so a git worktree has none of its own).
_DATA = Path(os.environ.get("NS_DATA_DIR", _REPO / "data"))
PARQUET = _DATA / "manuscript" / "all_fourier_df.parquet"
if "NS_DATA_DIR" in os.environ:
    _load_experiment = sv.schema.load_experiment

    def _load_with_data_dir(path):
        cfg = _load_experiment(path)
        cfg.data_dir = str(_DATA)
        return cfg
    sv.schema.load_experiment = _load_with_data_dir


# ------------------------------------------------------------------ recordings
def recordings():
    names = sorted(p.stem for p in (_REPO / "experiments").glob("*.yml")
                   if p.stem.startswith(("engert_", "medaka_")))
    return names


def describe(name, f):
    """(batch, coil on, visual stimulus, trial) from the experiment name and frequency."""
    if name.startswith("medaka"):
        batch = BATCHES[3]
    else:
        batch = {0.4: BATCHES[0], 0.017: BATCHES[0], 0.3: BATCHES[1], 0.1: BATCHES[2]}[round(f, 3)]
    coil = bool(re.search(r"(?<!no_)magnet", name))
    m = re.search(r"_(\d)$", name)
    trial = int(m.group(1)) if m else -1
    if batch == BATCHES[0]:
        visual = "visual" in name
    else:   # trials 1 and 2 had the grating, except 20221002_fish1's only trial (named _1)
        visual = trial in (1, 2) and "20221002_fish1" not in name
    return batch, coil, visual, trial


# ------------------------------------------------------------------ metrics
def chunk_length(N, K, f, T):
    """(L, cycles, delta) of the K-chunk split. The resolutions nest: the finest chunk L20 is the
    longest length <= N // max(KS) whose multiples L20, 2*L20 and 4*L20 (the K = 20, 10 and 5
    chunks) all hold a whole number of stimulus cycles, f * L * T, to within CYCLE_TOL, and
    L = L20 * max(KS) // K. So every K uses the same max(KS) * L20 frames and every coarse boundary
    is also a fine one. delta is what's left over (0 when T = 1 s, where a cycle of 0.1 Hz is 10
    frames and of 0.3 Hz 3 cycles per 10 frames)."""
    kmax = max(KS)
    for L20 in range(N // kmax, 0, -1):
        if round(f * L20 * T) >= 1 and all(
                abs(f * m * L20 * T - round(f * m * L20 * T)) <= CYCLE_TOL for m in (1, 2, 4)):
            L = L20 * kmax // K
            kappa = f * L * T
            return L, int(round(kappa)), kappa - round(kappa)
    raise ValueError(f"no nested whole-cycle chunks for N={N}, f={f}, T={T}")


def chunk_metrics(X, T, f, qf, mu, K):
    """Per-chunk metrics of traces X (..., ROI, N): dict name -> (..., ROI, K), and the number
    of reference-window bins. Chunks hold whole stimulus cycles (chunk_length); a leftover
    below CYCLE_TOL is removed by shifting the chunk's Fourier grid by delta bins, as production
    fit_Fourier(align="shift") does, so a coefficient sits exactly on f and the reference window is
    its adjacent coefficients."""
    L, kappa, delta = chunk_length(X.shape[-1], K, f, T)
    C = X[..., :K * L].reshape(*X.shape[:-1], K, L)
    rms = np.sqrt(np.mean((C - mu[:, None, None]) ** 2, -1))
    C = C - C.mean(-1, keepdims=True)
    nb = L // 2 + 1
    if abs(delta) < 1e-9:
        Y = sfft.rfft(C, axis=-1)
    else:
        n = np.arange(L)
        Y = sfft.fft(C * np.exp(-2j * np.pi * delta * n / L).astype(np.complex64), axis=-1)[..., :nb]
    P = np.abs(Y) ** 2 / L
    idx = np.arange(nb)
    xf = (idx + delta) / (L * T)
    k0 = kappa
    sel = (xf >= F_MIN) & (idx != k0) & (idx < L / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        W = P[..., sel] / P[..., sel].mean(-3, keepdims=True)
        cv2 = W.var(-1) / W.mean(-1) ** 2
    Mc = int(round(qf * kappa))
    band = (np.abs(idx - k0) <= Mc) & (idx != k0)
    bp = P[..., band].mean(-1) if band.any() else np.full(rms.shape, np.nan)
    return dict(cv2=cv2, rms=rms, band=bp), int(band.sum())


def dispersion(v):
    with np.errstate(invalid="ignore", divide="ignore"):
        return v.std(-1, ddof=1) / v.mean(-1)


def permutation(kind, N, B, rng):
    if kind == "full":
        return rng.permutation(N)
    starts = np.arange(0, N, B)
    return np.concatenate([np.arange(s, min(s + B, N)) for s in rng.permutation(starts)])


def ar1(F, rng):
    """Stationary Gaussian AR(1) traces with each row's lag-1 autocorrelation and variance."""
    Z = F - F.mean(1, keepdims=True)
    rho = np.clip((Z[:, 1:] * Z[:, :-1]).sum(1) / (Z ** 2).sum(1), 0, 0.999)
    out = np.empty_like(F)
    for i, r in enumerate(rho):
        e = rng.standard_normal(F.shape[1] + 500)
        x = lfilter([1.0], [1.0, -r], e)[500:]
        out[i] = F[i].mean() + x / x.std() * F[i].std()
    return out, rho


# ------------------------------------------------------------------ one recording
def run(name):
    cfg, F, roi = sv.load(name, outline=True, series="full")
    T, f, qf = cfg.sample_period, cfg.analysis.f, cfg.analysis.Q_frac
    batch, coil, visual, trial = describe(name, f)
    seed = zlib.crc32(name.encode())
    rng = np.random.default_rng([seed, 99])
    pick = np.unique(np.linspace(0, len(F) - 1, min(N_SYNTH, len(F))).astype(int))
    S, rho = ar1(F[pick], rng)
    X0 = np.concatenate([F, S]).astype(np.float32)
    synth = np.r_[np.zeros(len(F), bool), np.ones(len(S), bool)]
    rois = np.r_[roi, roi[pick]]
    mu = X0.mean(1)
    N, B = X0.shape[1], int(round(BLOCK_S / T))

    obs, series, nbins = {}, {}, {}
    for K in KS:
        m, nbins[K] = chunk_metrics(X0[None], T, f, qf, mu, K)
        for k, v in m.items():
            obs[k, K] = dispersion(v[0])
            series[k, K] = v[0]
    null = {(nl, k, K): np.empty((N_REP, len(X0)), np.float32)
            for nl in NULLS for k, _ in METRICS for K in KS}
    for j, nl in enumerate(NULLS):
        r = np.random.default_rng([seed, j])
        for b0 in range(0, N_REP, BATCH):
            idx = np.stack([permutation(nl, N, B, r) for _ in range(BATCH)])
            X = X0[:, idx].transpose(1, 0, 2)
            for K in KS:
                m, _ = chunk_metrics(X, T, f, qf, mu, K)
                for k, v in m.items():
                    null[nl, k, K][b0:b0 + BATCH] = dispersion(v)

    d = pd.DataFrame(dict(experiment=name, batch=batch, coil=coil, visual=visual, trial=trial,
                          roi=rois, synthetic=synth, N=N, T=T))
    d.loc[synth, "rho"] = rho
    for k, _ in METRICS:
        for K in KS:
            d[f"obs_{k}_{K}"] = obs[k, K]
            if k == "band":
                L, kappa, delta = chunk_length(N, K, f, T)
                d[f"nbins_band_{K}"] = nbins[K]
                d[f"L_{K}"], d[f"cycles_{K}"], d[f"delta_{K}"] = L, kappa, delta
            for nl in NULLS:
                z = null[nl, k, K]
                d[f"p_{nl}_{k}_{K}"] = (1 + (z >= obs[k, K]).sum(0)) / (1 + N_REP)
                d[f"ratio_{nl}_{k}_{K}"] = obs[k, K] / np.median(z, 0)
                d.loc[~np.isfinite(obs[k, K]), f"p_{nl}_{k}_{K}"] = np.nan
    ex = None
    if name in EXAMPLES:
        n = len(F)
        ex = dict(F=F, roi=roi, T=T, f=f,
                  **{f"series_{k}_{K}": series[k, K][:n] for k, _ in METRICS for K in KS},
                  **{f"null_{nl}_{k}_{K}": null[nl, k, K][:, :n]
                     for nl in NULLS for k, _ in METRICS for K in KS})
    print(f"  {name}: {len(F)} ROIs + {len(S)} AR(1), N={N}, block={B} frames, "
          f"band bins {[nbins[K] for K in KS]}", flush=True)
    return d, ex


def compute(workers):
    names = recordings()
    with ProcessPoolExecutor(workers) as pool:
        res = list(pool.map(run, names))
    d = pd.concat([r[0] for r in res], ignore_index=True)
    # each ROI's production 1F p-value, for the suspect comparison
    pq = pd.read_parquet(PARQUET, columns=["rec", "freq", "id", "p_value"])
    d["rec"] = d.experiment.map(
        {n: n.removeprefix("medaka_") + ".tif" if n.startswith("medaka") else n for n in names})
    f_of = {name: sv.schema.load_experiment(str(_REPO / "experiments" / f"{name}.yml")).analysis.f
            for name in names}
    d["freq"] = d.experiment.map(f_of)
    pq = pq[pq.rec.isin(d.rec.unique())].copy()
    pq["fkey"] = pq.freq.round(4)
    d["fkey"] = d.freq.round(4)
    d = d.merge(pq[["rec", "fkey", "id", "p_value"]].rename(columns={"id": "roi", "p_value": "p_1F"}),
                on=["rec", "fkey", "roi"], how="left")
    d.loc[d.synthetic, "p_1F"] = np.nan
    d.drop(columns="fkey").to_csv(OUT_CSV, index=False)
    exs = {n: r[1] for n, r in zip(names, res) if r[1] is not None}
    np.savez_compressed(OUT_NPZ, **{f"{n}|{k}": v for n, e in exs.items() for k, v in e.items()})
    return d.drop(columns="fkey")


def load_examples():
    z = np.load(OUT_NPZ)
    ex = {}
    for key in z.files:
        n, k = key.split("|", 1)
        ex.setdefault(n, {})[k] = z[key]
    return ex


# ------------------------------------------------------------------ figures
def example_roi(d, ex, name):
    """The example ROI of a recording: the one at the 95th percentile of std/mean of its trace
    (an active cell, not an outlier)."""
    F = ex[name]["F"]
    cv = F.std(1) / F.mean(1)
    return int(np.argsort(cv)[int(0.95 * (len(cv) - 1))])


def _chunk_lines(ax, N, T, f):
    for K, top in ((20, 0.33), (10, 0.66), (5, 1.0)):
        L = chunk_length(N, K, f, T)[0]
        for c in range(1, K + 1):
            ax.axvline(c * L * T / 60, ymin=0, ymax=top, color=C_K[K], lw=1.0 if K == 5 else 0.7,
                       zorder=3)


def fig_procedure(d, ex, out):
    fig, axes = plt.subplots(len(EXAMPLES), 1, figsize=(11, 1.7 * len(EXAMPLES)))
    for ax, name in zip(axes, EXAMPLES):
        e = ex[name]
        i = example_roi(d, ex, name)
        y, T = e["F"][i], float(e["T"])
        t = np.arange(len(y)) * T / 60
        ax.plot(t, y, color=C_INK, lw=0.4, zorder=2)
        _chunk_lines(ax, len(y), T, float(e["f"]))
        ax.set_xlim(0, t[-1])
        b = describe(name, 0.4 if "20220221" in name else (0.3 if "0914" in name else 0.1))[0]
        ax.set_title(f"{b}: {name}, ROI {int(e['roi'][i])}", fontsize=8, loc="left", color=C_INK)
        ax.set_ylabel("F (a.u.)", fontsize=7)
        _style(ax)
    axes[-1].set_xlabel("time (min)", fontsize=8)
    h = [plt.Line2D([], [], color=C_K[K], lw=1.5, label=f"K = {K} chunks") for K in KS]
    axes[0].legend(handles=h, fontsize=7, frameon=False, ncol=3, loc="upper right",
                   bbox_to_anchor=(1, 1.45))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_metrics(d, ex, name, out):
    e = ex[name]
    i = example_roi(d, ex, name)
    N, T = e["F"].shape[1], float(e["T"])
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(9, 6.2), sharex=True)
    for ax, (k, lab) in zip(axes, METRICS):
        for K in KS:
            v = e[f"series_{k}_{K}"][i]
            L = chunk_length(N, K, float(e["f"]), T)[0]
            edges = np.arange(K + 1) * L * T / 60
            ax.stairs(v, edges, color=C_K[K], lw=1.6 if K == 5 else 1.1, label=f"K = {K}",
                      baseline=None)
        ax.set_ylabel(lab, fontsize=8)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.25))
    axes[-1].set_xlabel("time (min)", fontsize=8)
    axes[-1].set_xlim(0, N * T / 60)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _p(p):
    return f"p ≤ {1 / (1 + N_REP):.3f}" if p <= 1 / (1 + N_REP) else f"p = {p:.3f}"


def fig_null(d, ex, name, out):
    e = ex[name]
    i = example_roi(d, ex, name)
    r = d[(d.experiment == name) & (d.roi == int(e["roi"][i])) & ~d.synthetic].iloc[0]
    fig, axes = plt.subplots(len(METRICS), len(KS), figsize=(11, 8))
    for a, (k, lab) in enumerate(METRICS):
        for b, K in enumerate(KS):
            ax = axes[a, b]
            o = r[f"obs_{k}_{K}"]
            for nl, c in (("full", C_FULL), ("block", C_K[K])):
                z = e[f"null_{nl}_{k}_{K}"][:, i]
                z = z[np.isfinite(z)]
                if len(z):
                    ax.hist(z, bins=40, color=c, alpha=0.8 if nl == "block" else 0.9, lw=0,
                            label=f"{nl} shuffle, {_p(r[f'p_{nl}_{k}_{K}'])}")
            if np.isfinite(o):
                ax.axvline(o, color=C_INK, lw=1.4, label="observed")
            ax.set_ylim(0, ax.get_ylim()[1] * 1.45)   # a band above the histograms for the legend
            ax.legend(fontsize=7, loc="upper left", frameon=True, facecolor="white",
                      edgecolor="none", framealpha=0.9, handlelength=1.2)
            if a == 0:
                ax.set_title(f"K = {K} chunks", fontsize=9, color=C_INK)
            _style(ax)
            if b == 0:
                ax.set_ylabel(f"{lab}\nreplicates", fontsize=8)
            if a == len(METRICS) - 1:
                ax.set_xlabel("std / mean across chunks", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def shares(d):
    """Share of ROIs with p < ALPHA per recording (real ROIs) and per batch (AR(1) control)."""
    cols = [f"p_{nl}_{k}_{K}" for nl in NULLS for k, _ in METRICS for K in KS]
    d = d.copy()
    d.loc[d.synthetic, "experiment"] = "AR(1) " + d.loc[d.synthetic, "batch"]   # pooled per batch
    d.loc[d.synthetic, ["coil", "visual"]] = False
    g = d.groupby(["experiment", "batch", "coil", "visual", "synthetic"])[cols]
    return g.apply(lambda x: (x < ALPHA).sum() / x.notna().sum()).reset_index()


def fig_population(d, out):
    s = shares(d)
    fig, axes = plt.subplots(len(NULLS), len(METRICS), figsize=(13, 6.5), sharey=True)
    rng = np.random.default_rng(0)
    groups = BATCHES + ["AR(1) control"]
    for a, nl in enumerate(NULLS):
        for b, (k, lab) in enumerate(METRICS):
            ax = axes[a, b]
            for gi, g in enumerate(groups):
                sub = s[s.synthetic] if g == groups[-1] else s[(s.batch == g) & ~s.synthetic]
                for ki, K in enumerate(KS):
                    x0 = gi + (ki - 1) * 0.25
                    v = sub[f"p_{nl}_{k}_{K}"].values
                    vis = sub.visual.values
                    jit = rng.uniform(-0.07, 0.07, len(v))
                    ax.scatter(x0 + jit[~vis], v[~vis], s=12, color=C_K[K], lw=0, zorder=3)
                    ax.scatter(x0 + jit[vis], v[vis], s=14, facecolor="none", edgecolor=C_K[K],
                               marker="^", lw=0.8, zorder=3)
            ax.axhline(ALPHA, color=C_INK, lw=0.8, ls="--", zorder=2)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([SHORT.get(g, "AR(1)\ncontrol") for g in groups], fontsize=7)
            ax.set_ylim(-0.02, 1.02)
            _style(ax)
            if a == 0:
                ax.set_title(lab, fontsize=9, color=C_INK)
            if b == 0:
                ax.set_ylabel(f"{nl} shuffle\nshare of ROIs with p < {ALPHA}", fontsize=8)
    h = [plt.Line2D([], [], marker="o", ls="", color=C_K[K], label=f"K = {K}") for K in KS]
    h += [plt.Line2D([], [], marker="o", ls="", color=C_MUTED, label="no visual stimulus"),
          plt.Line2D([], [], marker="^", ls="", mfc="none", mec=C_MUTED, label="visual stimulus"),
          plt.Line2D([], [], color=C_INK, ls="--", label=f"expected {ALPHA}")]
    fig.legend(handles=h, fontsize=7, frameon=False, ncol=6, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summary(d):
    s = shares(d)
    real = s[~s.synthetic]
    print("\nAR(1) control, share p < 0.05 (pooled ROIs):")
    syn = d[d.synthetic]
    for nl in NULLS:
        print(" ", nl, {f"{k}{K}": round(float((syn[f'p_{nl}_{k}_{K}'] < ALPHA).mean()), 3)
                        for k, _ in METRICS for K in KS})
    print("\nmedian share p < 0.05 per recording, block null:")
    cols = [f"p_block_{k}_{K}" for k, _ in METRICS for K in KS]
    print(real.groupby(["batch", "visual", "coil"])[cols].median().round(2).to_string())
    print("\nchunks (frames L, whole cycles, leftover delta, frames used), reference-window bins:")
    cols = [c for K in KS for c in (f"L_{K}", f"cycles_{K}", f"delta_{K}", f"nbins_band_{K}")]
    t = d[~d.synthetic].groupby(["batch", "N", "freq"])[cols].first()
    for K in KS:
        t[f"used_{K}"] = K * t[f"L_{K}"]
    print(t.round(3).to_string())
    print("\nsuspects (production p < 0.01 at the recording's analysis frequency) vs other ROIs,"
          " share with block p < 0.05:")
    r = d[~d.synthetic & d.p_1F.notna()]
    groups = [("magnetic frequency, no visual stimulus", (r.freq > 0.02) & ~r.visual),
              ("magnetic frequency, grating trials", (r.freq > 0.02) & r.visual),
              ("grating frequency 1/60 Hz (2022 Q1 visual)", r.freq < 0.02)]
    for gname, g in groups:
        x = r[g]
        sus = x.p_1F < 0.01
        print(f"  {gname}: {sus.sum()} suspects of {len(x)} ROIs")
        for k, _ in METRICS:
            for K in KS:
                c = f"p_block_{k}_{K}"
                ok = x[c].notna()
                n1 = ((x[c] < ALPHA) & sus & ok).sum(); m1 = (sus & ok).sum()
                n0 = ((x[c] < ALPHA) & ~sus & ok).sum(); m0 = (~sus & ok).sum()
                pf = stats.fisher_exact([[n1, m1 - n1], [n0, m0 - n0]])[1]
                print(f"    {k:5s} K={K:2d}: {n1}/{m1} suspects ({n1 / max(m1, 1):.3f}) vs "
                      f"{n0}/{m0} others ({n0 / max(m0, 1):.3f}), Fisher p = {pf:.2g}")


def figures(d):
    ex = load_examples()
    name = EXAMPLES[2]
    fig_procedure(d, ex, _HERE / "fig_ns_procedure.png")
    fig_metrics(d, ex, name, _HERE / "fig_ns_metrics.png")
    fig_null(d, ex, name, _HERE / "fig_ns_null.png")
    fig_population(d, _HERE / "fig_ns_population.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--only", help="run one experiment and print timing (no outputs)")
    a = ap.parse_args()
    if a.only:
        import time
        t0 = time.time()
        d, _ = run(a.only)
        print(f"{time.time() - t0:.1f} s")
        print(d[~d.synthetic].filter(like="p_block").lt(ALPHA).mean().round(3).to_string())
        print(d[d.synthetic].filter(like="p_").lt(ALPHA).mean().round(3).to_string())
        return
    d = pd.read_csv(OUT_CSV) if a.figures_only else compute(a.workers)
    summary(d)
    figures(d)


if __name__ == "__main__":
    main()
