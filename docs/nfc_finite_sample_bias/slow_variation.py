"""Does slow variation in the imaging traces explain the imaging p-value excess?

    python docs/nfc_finite_sample_bias/slow_variation.py                 # full run (~5 min)
    python docs/nfc_finite_sample_bias/slow_variation.py --figures-only  # replot from caches

Companion to imaging_surrogates.py. The hypothesis (advisor, 2026-10): slow
fluctuations in a cell's activity -- drift, bleaching, slow changes in
firing rate or in noise level -- make the trace nonstationary. NFC divides the
on-frequency coefficient by sigma-hat from 2M neighbouring bins, and the null
assumes those 2M+1 periodogram ordinates are independent. A slow change in
amplitude multiplies the whole band by a common factor, so neighbouring
ordinates move together, and sigma-hat has fewer effective degrees of freedom
than the null assumes.

What is computed, all at production thresholds and through the production
fit_Fourier / corrected_pvalues:

  raw views          example traces with their slow trend and 60 s envelope;
                     every ROI's periodogram across the analysis window,
                     sorted by p-value.
  bin coupling       within-ROI correlation of periodogram ordinates k and
                     k+lag across the window. Independent ordinates (the null)
                     give ~0 at every lag; measured against white noise of
                     the same N and M.
  per-ROI metrics    window tilt, lag-1 coupling, slow-power fraction,
                     envelope CV, kurtosis -- each against the ROI's p-value.
  surrogates         take slow variation out of the real traces (detrend,
                     then divide by the envelope) and put it into stationary
                     Gaussian noise (multiply by each ROI's own envelope).
  frequency scan     dev@0.5 at every analysis frequency with the window
                     width fixed in bins, for real and envelope-normalised
                     traces. Slow variation predicts an excess that is not
                     specific to the stimulus frequency.

Two populations are excluded or changed from Fig 2C, on purpose:
  - 20221002_fish1 magneto_2/_3 are byte-identical copies of magneto_1 on the
    NAS (CLAUDE.md); only magneto_1 is kept, so no ROI trace appears three times.
  - No neuron dedup: each (recording, ROI) trace is one row, because the
    question is about traces, not cells.

Outputs, next to this script: results_slow_variation_rois.csv,
results_slow_variation_scan.csv, results_slow_variation_coupling.csv,
fig_sv_*.png. slow_variation_cache.npz (window periodograms and example
traces for --figures-only) is gitignored.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "pipeline"))

from magpyneto2 import statistics as st  # noqa: E402
from magpyneto2.engert_helpers import fit_Fourier  # noqa: E402
from pipeline import body_outline, schema  # noqa: E402
from pipeline.analysis_stages import engert as engert_stage  # noqa: E402
from pipeline.analysis_stages import medaka as medaka_stage  # noqa: E402

PARQUET = _REPO / "data" / "manuscript" / "all_fourier_df.parquet"
CACHE = _HERE / "slow_variation_cache.npz"
DUPLICATE_RECS = {"engert_20221002_fish1_magneto_2", "engert_20221002_fish1_magneto_3"}

ENV_WIN_S = 60.0     # envelope: running RMS over this many seconds
HP_CUT_HZ = 0.02     # "slow" = below this; removed before the envelope is taken
N_DRAWS = 3          # stochastic surrogates are averaged over this many draws
MAX_LAG = 16         # bin-coupling lags shown
NEAR_BINS = 30       # bin offsets either side of the stimulus bin
P_GRID = np.linspace(0, 1, 201)
U_GRID = np.linspace(-1, 1, 41)   # window offset (bins / M), for the heatmaps

BATCHES = ["zebrafish 0.4 Hz (2022 Q1)", "zebrafish 0.3 Hz", "zebrafish 0.1 Hz",
           "medaka 0.1 Hz"]
SURROGATES = ["real", "detrended", "detrended + envelope-normalised",
              "Gaussian, smooth spectrum", "Gaussian, smooth spectrum x envelope",
              "white x envelope", "white"]

# Validated palette (same as simulate.py / imaging_surrogates.py).
C_REAL = "#e8590c"
C_SIM = "#1b6ac9"
C_INK = "#333333"
C_MUTED = "#8a8a8a"
# Categorical slots 1-4, fixed order: real, real-with-slow-variation-removed,
# Gaussian, Gaussian-with-slow-variation-added. White floors in gray.
C_SURR = {"real": "#e8590c", "detrended": "#f2a36b",
          "detrended + envelope-normalised": "#1baf7a",
          "Gaussian, smooth spectrum": "#2a78d6",
          "Gaussian, smooth spectrum x envelope": "#7b4fd6",
          "white x envelope": "#5a5a5a", "white": "#b0b0b0"}
QUART = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # ordinal blue ramp


# ------------------------------------------------------------------ data
def pool():
    """(experiment name, batch) for every imaging recording in the Fig 2C magnetic pool."""
    df = pd.read_parquet(PARQUET)
    neg, _, _ = st.get_poscontrols_negresults(df)
    im = neg[neg["species"].isin(["zebrafish", "medaka"])]
    out = []
    for (species, rec), g in im.groupby(["species", "rec"]):
        f = round(float(g["freq"].iloc[0]), 3)
        if species == "medaka":
            out.append(("medaka_" + rec.removesuffix(".tif"), "medaka 0.1 Hz", rec))
        elif rec not in DUPLICATE_RECS:
            b = {0.4: BATCHES[0], 0.3: BATCHES[1], 0.1: BATCHES[2]}[f]
            out.append((rec, b, rec))
    return out, im


def load(name, outline=False, series="suite2p"):
    """(cfg, F, ROI indices) of the ROIs passing the YAML's iscell/npix thresholds and flatline
    removal; with outline=True also the fish outline, as in production (but never the coverage
    threshold, which activity_coverage.py sweeps). `series`: suite2p's own traces by default,
    which these reports describe; "full" for the full-resolution traces production analyses
    since 2026-10-05 (docs/cv2.md)."""
    cfg = schema.load_experiment(str(_REPO / "experiments" / f"{name}.yml"))
    stage = medaka_stage if name.startswith("medaka") else engert_stage
    F, _, kept, _ = stage._load_from_nwb(
        cfg.nwb_path(), cfg.iscell_threshold, cfg.npix_threshold,
        outline=body_outline.params_for(cfg.body_outline) if outline else None, series=series)
    # fit_Fourier analyses the first N frames; everything here uses the same N.
    N = min(int(120 * (F.shape[1] // 60)), F.shape[1])
    return cfg, F[:, :N].astype(float), np.where(kept)[0]


# ------------------------------------------------------------------ helpers
def window_bins(N, T, f, Q_frac):
    xf = np.fft.fftfreq(N, T)[:N // 2]
    f0 = int(np.argmin(np.abs(f - xf)))
    M = st.bins_for_fraction(f, Q_frac, resolution=1.0 / (N * T), max_bins=min(f0, len(xf) - 1 - f0))
    return f0, M


def nfc_from_fft(Y, f0, M):
    off = np.concatenate([Y[:, f0 - M:f0], Y[:, f0 + 1:f0 + M + 1]], axis=1)
    return np.abs(Y[:, f0]) / np.sqrt(0.5 * np.mean(np.abs(off) ** 2, axis=1))


def pvals(NFC, M):
    # One shared `upper` per call, rounded up, so corrected_null_grid's
    # memoisation hits across surrogates instead of re-convolving each time.
    finite = NFC[np.isfinite(NFC)]
    upper = float(np.ceil(max(6.0, finite.max() * 1.05))) if finite.size else 6.0
    return st.corrected_pvalues(NFC, M, upper=upper)


def ecdf_dev(p):
    p = np.sort(p[np.isfinite(p)])
    return np.searchsorted(p, P_GRID, side="right") / len(p) - P_GRID


def dev_half(p):
    p = p[np.isfinite(p)]
    return float(np.mean(p <= 0.5) - 0.5)


def highpass(F, T, cut=HP_CUT_HZ):
    Y = np.fft.rfft(F - F.mean(axis=1, keepdims=True), axis=1)
    Y[:, np.fft.rfftfreq(F.shape[1], T) < cut] = 0
    return np.fft.irfft(Y, n=F.shape[1], axis=1)


def envelope(F, T):
    """Slow amplitude envelope: running RMS (ENV_WIN_S) of the high-passed trace."""
    w = max(3, int(round(ENV_WIN_S / T)))
    a = np.sqrt(uniform_filter1d(highpass(F, T) ** 2, w, axis=1, mode="reflect"))
    return a / np.sqrt(np.mean(a ** 2, axis=1, keepdims=True))


def detrend(F):
    """Subtract a per-trace cubic: removes bleaching/drift and the end-point
    mismatch, which is how purely ADDITIVE slow variation leaks into the window."""
    t = np.linspace(-1, 1, F.shape[1])
    V = np.vander(t, 4)
    coef, *_ = np.linalg.lstsq(V, F.T, rcond=None)
    return F - (V @ coef).T


def smooth_spectrum_gaussian(F, M, rng):
    """Stationary Gaussian noise whose EXPECTED spectrum is the ROI's periodogram
    smoothed over ~M/4 bins: keeps the window's tilt and curvature, discards
    bin-to-bin fluctuation and any dependence between ordinates.

    imaging_surrogates.py's `gaussian_psd` instead multiplied the REALISED
    periodogram by a fresh exponential draw, so each ordinate became a product
    of two exponentials; that surrogate gives dev@0.5 = -0.13 on pure white
    noise (see fig_sv_surrogate_validation.png).
    """
    N = F.shape[1]
    S = np.abs(np.fft.rfft(F - F.mean(axis=1, keepdims=True), axis=1)) ** 2
    S = gaussian_filter1d(S, max(2.0, M / 4), axis=1, mode="reflect")
    S[:, 0] = 0
    g = (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape)) / np.sqrt(2)
    Z = np.sqrt(S) * g
    Z[:, 0] = 0
    if N % 2 == 0:
        Z[:, -1] = Z[:, -1].real
    return np.fft.irfft(Z, n=N, axis=1)


def make_surrogate(kind, F, T, M, env, rng):
    if kind == "real":
        return F
    if kind == "detrended":
        return detrend(F)
    if kind == "detrended + envelope-normalised":
        return highpass(detrend(F), T) / env
    if kind == "Gaussian, smooth spectrum":
        return smooth_spectrum_gaussian(F, M, rng)
    if kind == "Gaussian, smooth spectrum x envelope":
        return smooth_spectrum_gaussian(F, M, rng) * env
    if kind == "white x envelope":
        return rng.standard_normal(F.shape) * env
    if kind == "white":
        return rng.standard_normal(F.shape)
    raise ValueError(kind)


def lag_corr(I, max_lag=MAX_LAG):
    """Within-ROI Pearson correlation of window ordinates k and k+lag; (ROIs, lags)."""
    out = np.full((len(I), max_lag), np.nan)
    for L in range(1, max_lag + 1):
        a, b = I[:, :-L], I[:, L:]
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
        out[:, L - 1] = (a * b).sum(axis=1) / np.where(den > 0, den, np.nan)
    return out


def window_periodogram(F, f0, M):
    Y = np.fft.rfft(F - F.mean(axis=1, keepdims=True), axis=1)
    return np.abs(Y[:, f0 - M:f0 + M + 1]) ** 2


# ------------------------------------------------------------------ per recording
def analyse(name, batch, rec, rng):
    cfg, F, roi_idx = load(name)
    if name.startswith("medaka"):
        roi_idx = np.arange(len(F))     # medaka's parquet id is still positional
    T, f, Q_frac = cfg.sample_period, cfg.analysis.f, cfg.analysis.Q_frac
    N = F.shape[1]
    f0, M = window_bins(N, T, f, Q_frac)

    # Production p-values via the production function (self-test target).
    NFC_l, *_ , M_prod, _ = fit_Fourier(F, T=T, f=f, Q_frac=Q_frac)
    assert M_prod == M
    p_real = st.corrected_pvalues(np.asarray(NFC_l), M)

    env = envelope(F, T)
    I = window_periodogram(F, f0, M)
    sigma2 = 0.5 * np.mean(np.delete(I, M, axis=1), axis=1)
    u = (np.arange(-M, M + 1)) / M
    I_norm_u = np.array([np.interp(U_GRID, u, row) for row in I / (2 * sigma2[:, None])])

    # ---- per-ROI metrics
    off = np.delete(I, M, axis=1)
    u_off = np.delete(u, M)
    slope = np.polyfit(u_off, np.log10(off.T), 1)[0]               # log10 power per half-window
    rho1 = lag_corr(I, 1)[:, 0]
    S_all = np.abs(np.fft.rfft(F - F.mean(axis=1, keepdims=True), axis=1)) ** 2
    fr = np.fft.rfftfreq(N, T)
    slow_frac = S_all[:, (fr > 0) & (fr < HP_CUT_HZ)].sum(1) / S_all[:, fr > 0].sum(1)
    env_cv = np.std(env ** 2, axis=1) / np.mean(env ** 2, axis=1)
    kurt = stats.kurtosis(highpass(F, T), axis=1)
    lo = F.min(axis=1, keepdims=True)
    floor_frac = np.mean(F <= lo + 1e-6 * np.abs(lo), axis=1)      # frames at the trace's minimum

    rows = pd.DataFrame(dict(experiment=name, rec=rec, batch=batch, id=roi_idx, freq=f, M=M,
                             N=N, p_value=p_real, window_tilt=slope, rho1=rho1,
                             slow_frac=slow_frac, env_cv=env_cv, kurtosis=kurt,
                             floor_frac=floor_frac))

    # ---- surrogates: p-value per ROI per kind (stochastic ones averaged over draws
    # at the curve level, so keep every draw's p-values)
    surr_p, coupling = {}, {}
    for kind in SURROGATES:
        draws = 1 if kind in ("real", "detrended", "detrended + envelope-normalised") else N_DRAWS
        ps, lc = [], []
        for _ in range(draws):
            G = make_surrogate(kind, F, T, M, env, rng)
            Y = np.fft.rfft(G - G.mean(axis=1, keepdims=True), axis=1)
            ps.append(pvals(nfc_from_fft(Y, f0, M), M))
            lc.append(np.nanmean(lag_corr(np.abs(Y[:, f0 - M:f0 + M + 1]) ** 2), axis=0))
        surr_p[kind] = np.stack(ps)
        coupling[kind] = np.mean(lc, axis=0)
    rows["p_detrended"] = surr_p["detrended"][0]
    rows["p_envnorm"] = surr_p["detrended + envelope-normalised"][0]
    np.testing.assert_allclose(surr_p["real"][0], p_real, atol=1e-6)

    # ---- frequency scan, window width fixed at the production M (bins)
    scan = []
    variants = {"real": F, "detrended + envelope-normalised": highpass(detrend(F), T) / env,
                "white": rng.standard_normal(F.shape)}
    Ys = {k: np.fft.rfft(v - v.mean(axis=1, keepdims=True), axis=1) for k, v in variants.items()}
    lo = int(np.ceil((HP_CUT_HZ * 2) * N * T)) + M      # window clear of the slow band
    hi = N // 2 - M - 2
    for k0 in np.unique(np.geomspace(lo, hi, 60).astype(int)):
        for kind, Y in Ys.items():
            p = pvals(nfc_from_fft(Y, k0, M), M)
            scan.append(dict(experiment=name, batch=batch, kind=kind, freq=k0 / (N * T),
                             f_stim=f, n=len(p), n_le_half=int(np.sum(p <= 0.5))))

    Yr = Ys["real"]
    for off in range(-NEAR_BINS, NEAR_BINS + 1):
        p = p_real if off == 0 else pvals(nfc_from_fft(Yr, f0 + off, M), M)
        scan.append(dict(experiment=name, batch=batch, kind="near stimulus", offset=off,
                         freq=(f0 + off) / (N * T), f_stim=f, n=len(p),
                         n_le_half=int(np.sum(p <= 0.5))))

    examples = dict(F=F[:3], env=env[:3], trend=(F - detrend(F))[:3], T=T,
                    p=p_real[:3])
    return rows, surr_p, coupling, scan, I_norm_u, examples


# ------------------------------------------------------------------ figures
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_INK, labelsize=7)


def _band(ax, n):
    half = 1.96 * np.sqrt(P_GRID * (1 - P_GRID) / n)
    ax.fill_between(P_GRID, -half, half, color="#d9d9d9", lw=0, zorder=0)


def fig_traces(ex, out):
    """Raw traces with the slow envelope that envelope normalisation divides out."""
    fig, axes = plt.subplots(len(BATCHES), 3, figsize=(12, 1.9 * len(BATCHES)), squeeze=False)
    for r, b in enumerate(BATCHES):
        e = ex[b]
        t = np.arange(e["F"].shape[1]) * e["T"]
        for c in range(3):
            ax = axes[r, c]
            y = e["F"][c]
            ax.plot(t, y, color=C_INK, lw=0.4)
            sd = np.std(highpass(y[None], e["T"]))
            ax.plot(t, np.mean(y) + 2 * sd * e["env"][c], color=C_SIM, lw=1.2,
                    label=f"mean + 2 SD x envelope ({ENV_WIN_S:.0f} s RMS)")
            ax.set_title(f"{b}  {e['name']}  ROI {e['ids'][c]}  p = {e['p'][c]:.2f}",
                         fontsize=7, color=C_INK)
            _style(ax)
            if r == len(BATCHES) - 1:
                ax.set_xlabel("time (s)", fontsize=7)
            if c == 0:
                ax.set_ylabel("F (a.u.)", fontsize=7)
    axes[0, 0].legend(fontsize=6, frameon=False, loc="upper right")
    fig.suptitle("Raw traces (first three ROIs of one recording from each set of recordings, no selection on p)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_window_spectra(rois, Inorm, out):
    """Every ROI's window periodogram, sorted by p; and its median across ROIs."""
    fig, axes = plt.subplots(2, len(BATCHES), figsize=(3.3 * len(BATCHES), 7.2),
                             gridspec_kw=dict(height_ratios=[3, 1.3]), squeeze=False)
    for c, b in enumerate(BATCHES):
        sel = (rois["batch"] == b).values
        X, p = Inorm[sel], rois.loc[sel, "p_value"].values
        order = np.argsort(p)
        ax = axes[0, c]
        im = ax.imshow(np.log10(X[order]), aspect="auto", cmap="magma", vmin=-2, vmax=1,
                       extent=(-1, 1, 1, 0), interpolation="nearest")
        ax.set_title(f"{b}\n{sel.sum()} ROI traces", fontsize=8)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(["0", ".25", ".5", ".75", "1"], fontsize=7)
        if c == 0:
            ax.set_ylabel("ROIs sorted by p-value (p quantile)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax2 = axes[1, c]
        med = np.median(np.log10(X), axis=0)
        q1, q3 = np.percentile(np.log10(X), [25, 75], axis=0)
        ax2.fill_between(U_GRID, q1, q3, color=C_REAL, alpha=0.2, lw=0)
        ax2.plot(U_GRID, med, color=C_REAL, lw=1.4, label="real: median, IQR")
        # Exp(1) ordinates: median log10 = log10(ln 2), IQR log10(-ln .75)..log10(ln 4)
        ax2.axhline(np.log10(np.log(2)), color=C_MUTED, ls="--", lw=1,
                    label="independent ordinates (Exp(1)) median")
        for v in (np.log10(-np.log(0.75)), np.log10(np.log(4))):
            ax2.axhline(v, color=C_MUTED, ls=":", lw=0.8)
        ax2.set_xlabel("window offset (bins / M); 0 = analysis frequency", fontsize=7)
        if c == 0:
            ax2.set_ylabel("log10 I_k / (2 sigma-hat^2)", fontsize=7)
        _style(ax2)
    axes[1, 0].legend(fontsize=6, frameon=False)
    fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01,
                 label="log10 ordinate / (2 sigma-hat^2)")
    fig.suptitle("Periodogram across the analysis window, every ROI (white column at 0 = the on-frequency bin)",
                 fontsize=9)
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fig_coupling(coup, out):
    fig, axes = plt.subplots(1, len(BATCHES), figsize=(3.3 * len(BATCHES), 3.2), squeeze=False)
    lags = np.arange(1, MAX_LAG + 1)
    for c, b in enumerate(BATCHES):
        ax = axes[0, c]
        d = coup[coup["batch"] == b]
        for kind in SURROGATES:
            g = d[d["kind"] == kind]
            if g.empty:
                continue
            # weight each recording by its ROI count
            w = g["n"].values[:, None]
            y = (g[[f"lag{L}" for L in lags]].values * w).sum(0) / w.sum()
            ax.plot(lags, y, color=C_SURR[kind], lw=2 if kind == "real" else 1.2,
                    marker="o", ms=3, label=kind)
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_title(b, fontsize=8)
        ax.set_xlabel("lag (bins)", fontsize=7)
        if c == 0:
            ax.set_ylabel("within-ROI corr(I_k, I_k+lag)\nmean over ROIs", fontsize=7)
        _style(ax)
    axes[0, -1].legend(fontsize=6, frameon=False, loc="upper right")
    fig.suptitle("Do neighbouring periodogram ordinates move together? (null: independent, ~0)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


METRICS = [("window_tilt", "window tilt (log10 power, -M to +M)"),
           ("rho1", "lag-1 ordinate coupling"),
           ("slow_frac", f"slow-power fraction (< {HP_CUT_HZ} Hz)"),
           ("env_cv", f"envelope CV ({ENV_WIN_S:.0f} s)"),
           ("kurtosis", "excess kurtosis"),
           ("floor_frac", "fraction of frames at the trace minimum")]


def fig_metric_scatter(rois, out):
    fig, axes = plt.subplots(len(METRICS), len(BATCHES), figsize=(3.2 * len(BATCHES), 2.3 * len(METRICS)),
                             squeeze=False)
    for r, (m, lab) in enumerate(METRICS):
        for c, b in enumerate(BATCHES):
            ax = axes[r, c]
            d = rois[rois["batch"] == b]
            x = d[m].values
            if m in ("slow_frac", "env_cv", "kurtosis"):
                x = np.log10(np.clip(x, 1e-4, None))
            ax.scatter(x, d["p_value"], s=2, color=C_INK, alpha=0.25, lw=0, rasterized=True)
            rho, pp = stats.spearmanr(x, d["p_value"], nan_policy="omit")
            ax.set_title(f"{b}\nSpearman {rho:+.3f} (p={pp:.1g})" if r == 0
                         else f"Spearman {rho:+.3f} (p={pp:.1g})", fontsize=7)
            ax.set_xlabel(("log10 " if m in ("slow_frac", "env_cv", "kurtosis") else "") + lab,
                          fontsize=7)
            if c == 0:
                ax.set_ylabel("p-value (magnetic f)", fontsize=7)
            _style(ax)
    fig.suptitle("Per-ROI slow-variation metrics against p-value, one dot per ROI trace", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_metric_quartiles(rois, out):
    fig, axes = plt.subplots(len(METRICS), len(BATCHES), figsize=(3.2 * len(BATCHES), 2.3 * len(METRICS)),
                             squeeze=False, sharey=True)
    for r, (m, lab) in enumerate(METRICS):
        for c, b in enumerate(BATCHES):
            ax = axes[r, c]
            d = rois[rois["batch"] == b]
            q = pd.qcut(d[m].rank(method="first"), 4, labels=False)
            _band(ax, len(d) / 4)
            for k in range(4):
                p = d.loc[q == k, "p_value"].values
                ax.plot(P_GRID, ecdf_dev(p), color=QUART[k], lw=1.3,
                        label=f"Q{k + 1} (dev@0.5 {dev_half(p):+.3f})")
            ax.axhline(0, color=C_INK, lw=0.6)
            ax.set_title(f"{b}\n{lab}" if r == 0 else lab, fontsize=7)
            ax.legend(fontsize=5.5, frameon=False, loc="lower center")
            if c == 0:
                ax.set_ylabel("ECDF(p) - p", fontsize=7)
            if r == len(METRICS) - 1:
                ax.set_xlabel("p", fontsize=7)
            _style(ax)
    fig.suptitle("ECDF deviation by metric quartile (Q1 = lowest; gray = 95% binomial band for one quartile)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_surrogates(curves, out):
    fig, axes = plt.subplots(1, len(BATCHES), figsize=(3.4 * len(BATCHES), 5.0), squeeze=False,
                             sharey=True)
    for c, b in enumerate(BATCHES):
        ax = axes[0, c]
        d = curves[curves["batch"] == b]
        _band(ax, int(d["n"].iloc[0]))
        for kind in SURROGATES:
            g = d[d["kind"] == kind]
            y = g[[f"c{i}" for i in range(len(P_GRID))]].values[0]
            ax.plot(P_GRID, y, color=C_SURR[kind], lw=2 if kind == "real" else 1.2,
                    label=f"{kind} ({float(g['dev_at_half'].iloc[0]):+.3f})")
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_title(f"{b}  (N = {int(d['n'].iloc[0])} ROI traces)", fontsize=8)
        ax.set_xlabel("p", fontsize=7)
        if c == 0:
            ax.set_ylabel("ECDF(p) - p", fontsize=7)
        ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
        _style(ax)
    fig.suptitle("Surrogates at the magnetic frequency; legend = dev@0.5; gray = 95% binomial band",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_scan(scan, out):
    fig, axes = plt.subplots(1, len(BATCHES), figsize=(3.4 * len(BATCHES), 3.6), squeeze=False,
                             sharey=True)
    colors = {"real": C_REAL, "detrended + envelope-normalised": "#1baf7a", "white": "#b0b0b0"}
    for c, b in enumerate(BATCHES):
        ax = axes[0, c]
        d = scan[(scan["batch"] == b) & (scan["kind"] != "near stimulus")].copy()
        # pool recordings onto a common log-frequency grid
        edges = np.geomspace(d["freq"].min(), d["freq"].max() * 1.0001, 31)
        d["fb"] = np.digitize(d["freq"], edges)
        for kind, col in colors.items():
            g = d[d["kind"] == kind].groupby("fb").agg(n=("n", "sum"), k=("n_le_half", "sum"),
                                                        f=("freq", "median"))
            dev = g["k"] / g["n"] - 0.5
            se = np.sqrt(0.25 / g["n"])
            ax.fill_between(g["f"], dev - 1.96 * se, dev + 1.96 * se, color=col, alpha=0.18, lw=0)
            ax.plot(g["f"], dev, color=col, lw=1.4, marker="o", ms=2.5, label=kind)
        fs = d["f_stim"].iloc[0]
        for h, ls in ((1, "-"), (2, "--")):
            if fs * h < d["freq"].max():
                ax.axvline(fs * h, color=C_INK, lw=0.8, ls=ls)
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_xscale("log")
        ax.set_xticks([0.07, 0.1, 0.2, 0.3, 0.4])
        ax.set_xticklabels(["0.07", "0.1", "0.2", "0.3", "0.4"])
        ax.minorticks_off()
        ax.set_title(f"{b}\n(vertical: stimulus f solid, 2f dashed)", fontsize=8)
        ax.set_xlabel("analysis frequency (Hz)", fontsize=7)
        if c == 0:
            ax.set_ylabel("dev@0.5 (window fixed at production M bins)", fontsize=7)
        _style(ax)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle("Is the excess specific to the stimulus frequency? dev@0.5 at every analysis frequency, "
                 "band = 95% binomial", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_near_stimulus(scan, out):
    """dev@0.5 at the stimulus bin and at every bin within NEAR_BINS of it, per recording."""
    d = scan[scan["kind"] == "near stimulus"]
    fig, axes = plt.subplots(1, len(BATCHES), figsize=(3.4 * len(BATCHES), 3.4), squeeze=False,
                             sharey=True)
    for c, b in enumerate(BATCHES):
        ax = axes[0, c]
        g = d[d["batch"] == b]
        for e, ge in g.groupby("experiment"):
            ge = ge.sort_values("offset")
            ax.plot(ge["offset"], ge["n_le_half"] / ge["n"] - 0.5, color=C_MUTED, lw=0.6, alpha=0.6)
        pooled = g.groupby("offset").agg(n=("n", "sum"), k=("n_le_half", "sum"))
        dev = pooled["k"] / pooled["n"] - 0.5
        se = np.sqrt(0.25 / pooled["n"])
        ax.fill_between(pooled.index, dev - 1.96 * se, dev + 1.96 * se, color=C_REAL, alpha=0.2, lw=0)
        ax.plot(pooled.index, dev, color=C_REAL, lw=1.5, marker="o", ms=2.5, label="pooled")
        ax.plot([0], [dev.loc[0]], marker="o", ms=7, color=C_REAL, mec=C_INK)
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.axvline(0, color=C_INK, lw=0.6, ls=":")
        ax.set_title(f"{b}\nstimulus bin dev@0.5 = {dev.loc[0]:+.3f}", fontsize=8)
        ax.set_xlabel("analysis bin - stimulus bin", fontsize=7)
        if c == 0:
            ax.set_ylabel("dev@0.5", fontsize=7)
        _style(ax)
    axes[0, 0].plot([], [], color=C_MUTED, lw=0.6, label="one recording")
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle("Is the stimulus bin special? dev@0.5 at the stimulus bin (big dot) and its neighbours "
                 "(window fixed at M bins); band = 95% binomial for the pool", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def surrogate_validation(out_csv, out_png):
    """Each Gaussian surrogate generator applied to white noise must give the
    white-noise floor; the old realised-periodogram x exponential one does not."""
    rng = np.random.default_rng(7)
    N, T, f, Q_frac = 1080, 1.0, 0.1, 0.3
    f0, M = window_bins(N, T, f, Q_frac)
    W = rng.standard_normal((20000, N))
    Y = np.fft.rfft(W, axis=1)
    g = (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape)) / np.sqrt(2)
    old = np.abs(Y) * g
    res = {"white noise itself": pvals(nfc_from_fft(Y, f0, M), M),
           "smooth-spectrum Gaussian of white noise (this report)":
               pvals(nfc_from_fft(np.fft.rfft(smooth_spectrum_gaussian(W, M, rng), axis=1), f0, M), M),
           "|Y| x complex Gaussian of white noise (imaging_surrogates.py gaussian_psd)":
               pvals(nfc_from_fft(old, f0, M), M)}
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _band(ax, W.shape[0])
    cols = ["#b0b0b0", "#2a78d6", "#e8590c"]
    rows = []
    for (k, p), col in zip(res.items(), cols):
        ax.plot(P_GRID, ecdf_dev(p), color=col, lw=1.5, label=f"{k}  ({dev_half(p):+.3f})")
        rows.append(dict(generator=k, dev_at_half=dev_half(p), p_below_01=float(np.mean(p < 0.01))))
    ax.axhline(0, color=C_INK, lw=0.6)
    ax.set_xlabel("p", fontsize=8)
    ax.set_ylabel("ECDF(p) - p", fontsize=8)
    ax.set_title(f"Surrogate generators applied to white noise (N={N}, M={M}, 20000 traces)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


# ------------------------------------------------------------------ driver
def figures(rois, curves, coup, scan, Inorm, ex):
    fig_traces(ex, _HERE / "fig_sv_traces.png")
    fig_window_spectra(rois, Inorm, _HERE / "fig_sv_window_spectra.png")
    fig_coupling(coup, _HERE / "fig_sv_bin_coupling.png")
    fig_metric_scatter(rois, _HERE / "fig_sv_metrics_scatter.png")
    fig_metric_quartiles(rois, _HERE / "fig_sv_metrics_quartiles.png")
    fig_surrogates(curves, _HERE / "fig_sv_surrogates.png")
    fig_scan(scan, _HERE / "fig_sv_freq_scan.png")
    fig_near_stimulus(scan, _HERE / "fig_sv_near_stimulus.png")
    print("  wrote fig_sv_*.png")


def _save_cache(Inorm, ex):
    flat = {"Inorm": Inorm}
    for i, (b, e) in enumerate(ex.items()):
        for k in ("F", "env", "trend", "p", "ids"):
            flat[f"ex{i}_{k}"] = e[k]
        flat[f"ex{i}_T"] = np.array(e["T"])
        flat[f"ex{i}_name"] = np.array(e["name"])
    np.savez_compressed(CACHE, **flat)


def _load_cache(rois):
    """Examples are keyed by their recording name, NOT by position: they are saved in
    recording order, which is not BATCHES order (keying by position once mislabelled
    every row of fig_sv_traces.png)."""
    z = np.load(CACHE)
    batch_of = rois.groupby("experiment")["batch"].first()
    ex = {}
    i = 0
    while f"ex{i}_name" in z:
        name = str(z[f"ex{i}_name"])
        e = {k: z[f"ex{i}_{k}"] for k in ("F", "env", "trend", "p", "ids")}
        e["T"] = float(z[f"ex{i}_T"])
        e["name"] = name
        ex[batch_of[name]] = e
        i += 1
    return z["Inorm"], ex


def main():
    if "--figures-only" in sys.argv:
        rois = pd.read_csv(_HERE / "results_slow_variation_rois.csv")
        curves = pd.read_csv(_HERE / "results_slow_variation_curves.csv")
        coup = pd.read_csv(_HERE / "results_slow_variation_coupling.csv")
        scan = pd.read_csv(_HERE / "results_slow_variation_scan.csv")
        Inorm, ex = _load_cache(rois)
        figures(rois, curves, coup, scan, Inorm, ex)
        return

    surrogate_validation(_HERE / "results_surrogate_validation.csv",
                         _HERE / "fig_sv_surrogate_validation.png")

    recs, im = pool()
    rng = np.random.default_rng(0)
    rois_l, coup_l, scan_l, Inorm_l = [], [], [], []
    surr_by_batch = {b: {k: [] for k in SURROGATES} for b in BATCHES}
    ex = {}
    for name, batch, rec in recs:
        print(f"  {name}  [{batch}]")
        rows, surr_p, coupling, scan, Inorm, examples = analyse(name, batch, rec, rng)
        rois_l.append(rows)
        Inorm_l.append(Inorm)
        scan_l += scan
        for kind in SURROGATES:
            # draws stay separate (columns) so curves average over draws, not ROIs
            surr_by_batch[batch][kind].append(surr_p[kind])
            coup_l.append(dict(experiment=name, batch=batch, kind=kind, n=len(rows),
                               **{f"lag{L + 1}": v for L, v in enumerate(coupling[kind])}))
        if batch not in ex:
            ex[batch] = dict(examples, name=name, ids=rows["id"].values[:3])

    rois = pd.concat(rois_l, ignore_index=True)
    Inorm = np.concatenate(Inorm_l)

    # Self-test: the production p-values recomputed here must match the parquet.
    j = rois.merge(im[["rec", "id", "p_value"]], on=["rec", "id"],
                   how="left", suffixes=("", "_parquet"))
    assert j["p_value_parquet"].notna().all(), "ROI rows missing from the parquet"
    err = float(np.max(np.abs(j["p_value"] - j["p_value_parquet"])))
    print(f"  self-test: {len(j)} ROI rows matched to the parquet, max |dp| = {err:.1e}")
    assert err < 1e-3

    curves = []
    for b in BATCHES:
        for kind in SURROGATES:
            P = np.concatenate(surr_by_batch[b][kind], axis=1)     # (draws, ROIs)
            c = np.mean([ecdf_dev(row) for row in P], axis=0)
            curves.append(dict(batch=b, kind=kind, n=P.shape[1],
                               dev_at_half=float(np.mean([dev_half(row) for row in P])),
                               p_below_01=float(np.mean(P < 0.01)),
                               **{f"c{i}": v for i, v in enumerate(c)}))
    curves = pd.DataFrame(curves)
    coup = pd.DataFrame(coup_l)
    scan = pd.DataFrame(scan_l)

    rois.to_csv(_HERE / "results_slow_variation_rois.csv", index=False)
    curves.to_csv(_HERE / "results_slow_variation_curves.csv", index=False)
    coup.to_csv(_HERE / "results_slow_variation_coupling.csv", index=False)
    scan.to_csv(_HERE / "results_slow_variation_scan.csv", index=False)
    _save_cache(Inorm, ex)

    print("\ndev@0.5 by batch and surrogate:")
    print(curves.pivot(index="kind", columns="batch", values="dev_at_half")
          .loc[SURROGATES, BATCHES].round(4).to_string())
    figures(rois, curves, coup, scan, Inorm, ex)


if __name__ == "__main__":
    main()
