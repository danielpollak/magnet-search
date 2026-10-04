"""How a trace with few events produces a rippled spectrum, correlated ordinates and p-values
that pile up in the middle of the range.

    python docs/guard_band_null/ripples.py

Write-up: docs/few_event_spectra.md. Three figures:

  fig_gb_ripples_toy.png   synthetic traces (1 event, 2 events, an epoch of events, continuous
                           activity), one per column, followed through every step: trace,
                           spectrum, ordinate correlation, analysis power vs noise power, and
                           the power ratio against its null.
  fig_gb_ripples_real.png  the same steps for ten real ROIs of one 0.1 Hz zebrafish recording:
                           five with high activity coverage, five with low-to-medium coverage
                           (all of them pass production's thresholds). ROIs are drawn at random
                           within each coverage group, never selected on p.
  fig_gb_ripples_pooled.png  the same steps pooled over every ROI of that recording, in three
                           coverage groups (including the ROIs production drops).

Notation. I_k = |Y_k|^2 is the periodogram ordinate at bin k. At an analysis bin k the power
ratio is R_k = I_k / mean(I over the 2M noise bins), so NFC^2 / 2 = R. For Gaussian noise R
follows F(2, 4M) (close to Exp(1)): it is often far below 1 and sometimes far above, and the
p-value p = P(F(2, 4M) > R) is uniform. R = 1 gives p ~ e^-1 = 0.37.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter1d

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE.parent / "nfc_finite_sample_bias"))
sys.path.insert(0, str(_REPO))
import slow_variation as sv  # noqa: E402
import slow_variation_sim as svs  # noqa: E402
from slow_variation import C_INK, C_MUTED, _style  # noqa: E402
from pipeline.roi_coverage import activity  # noqa: E402

REC = "engert_20221001_fish2_magneto_0"   # 0.1 Hz zebrafish
GOOD_MIN, BAD_RANGE = 0.8, (0.1, 0.4)     # coverage groups; production keeps coverage >= 0.1
N_EACH = 5
F_MIN = 0.05                               # lowest analysis frequency used (Hz)
SHOW_BINS = 60                             # spectrum shown within +-SHOW_BINS of the stimulus bin
MAX_LAG = 30
MED_WIN = 51
LEVEL_WIN = 201                            # bins; local spectrum level for panel D
EVENT_AMP = 5.0                            # spikes per synthetic event
R_MAX = 6.0
C_GOOD, C_BAD = "#2a78d6", "#eb6834"


# ------------------------------------------------------------------ per-trace quantities
def spectrum(x):
    return np.abs(np.fft.rfft(x - x.mean())) ** 2


def analysis_bins(n_bins, res, M, f0):
    k = np.arange(max(int(np.ceil(F_MIN / res)), M + 1), n_bins - M - 1)
    return k[np.abs(k - f0) > M]


def ratios(I, ks, M):
    """Analysis power, noise power and their ratio R at each analysis bin. The two powers are
    returned relative to the spectrum's local level (running mean over LEVEL_WIN bins, wide
    enough to average over the ripples), so that the spectrum's overall fall-off with
    frequency does not spread the points; R is computed from the raw powers."""
    num = I[ks]
    den = np.array([np.r_[I[k - M:k], I[k + 1:k + M + 1]].mean() for k in ks])
    level = uniform_filter1d(I, LEVEL_WIN, mode="nearest")[ks]
    return num / level, den / level, num / den


def ordinate_corr(I, lo):
    """Correlation of ordinates L bins apart, each divided by the running mean of its MED_WIN
    neighbours so the spectrum's overall slope does not count as correlation."""
    J = I[lo:] / np.maximum(uniform_filter1d(I[lo:], MED_WIN, mode="nearest"), 1e-30)
    J = J - J.mean()
    v = max((J * J).mean(), 1e-30)
    return np.array([(J[:-L] * J[L:]).mean() / v for L in range(1, MAX_LAG + 1)])


def cv2(I, lo):
    J = I[lo:] / np.maximum(uniform_filter1d(I[lo:], MED_WIN, mode="nearest"), 1e-30)
    return J.var() / J.mean() ** 2


# ------------------------------------------------------------------ synthetic traces
def toy_traces(N, T, rng):
    """slow_variation_sim's GCaMP model (2 s decay, Gaussian noise of SD NOISE_SD) with
    events of EVENT_AMP spikes each and the baseline clipped at 4 noise SDs, so that, as in the
    real floor-clipped traces, almost nothing but the events rises above the floor. The last
    trace is unclipped single spikes at svs.RATE."""
    kern = np.exp(-np.arange(0, 10 * svs.TAU_CA, T) / svs.TAU_CA)

    def make(times, amp=EVENT_AMP, clip=True):
        s = np.zeros(N)
        np.add.at(s, (np.asarray(times) / T).astype(int), amp)
        x = np.convolve(s, kern)[:N] + svs.NOISE_SD * rng.standard_normal(N)
        return np.maximum(x - 4 * svs.NOISE_SD, 0.0) if clip else x

    epoch = np.sort(rng.uniform(300, 600, 8))
    steady = np.flatnonzero(rng.random(N) < svs.RATE * T) * T
    return [("1 event", make([500])),
            ("2 events, 100 s apart", make([450, 550])),
            ("8 events in one 300 s epoch", make(epoch)),
            ("continuous activity, no floor\n(like the 2022 Q1 traces)",
             make(steady, amp=1.0, clip=False))]


# ------------------------------------------------------------------ plotting
def null_density(r, M):
    return stats.f.pdf(r, 2, 4 * M)


C_RE, C_IM = "#1b1b1b", "#a0a0a0"
HEADS = ["trace",
         "Fourier coefficients near the stimulus frequency\n(black: real, gray: imaginary; "
         "shading: noise bins)",
         "complex plane" + chr(10) + "(gray: noise bins; dot: analysis bin)",
         "correlation of ordinates L bins apart",
         "power at the analysis bin vs\nmean power of its noise bins",
         "power ratio R at every analysis bin\n(black: null, F(2, 4M))"]
N_PANELS = len(HEADS)


def coefficients(x):
    return np.fft.rfft(x - x.mean())


def draw_row(axes, x, T, f0, M, res, color, title, show_titles):
    """trace | coefficient spectrum | complex plane | ordinate correlation |
    analysis vs noise power | R histogram."""
    N = len(x)
    Y = coefficients(x)
    I = np.abs(Y) ** 2
    lo = max(int(np.ceil(F_MIN / res)) - MED_WIN, 1)
    ks = analysis_bins(len(I), res, M, f0)
    num, den, R = ratios(I, ks, M)
    rho = ordinate_corr(I, lo)
    p = stats.f.sf(R, 2, 4 * M)
    noise = np.r_[f0 - M:f0, f0 + 1:f0 + M + 1]
    # Coefficients divided by sigma-hat at the stimulus bin, the same normalisation as NFC:
    # under the null each component is N(0, 1) and |c| at the stimulus bin is its NFC.
    c = Y / np.sqrt(0.5 * I[noise].mean())

    ax = axes[0]
    ax.plot(np.arange(N) * T, x, color=color, lw=0.6)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlim(0, N * T)
    ax.set_ylabel("F", fontsize=7)

    ax = axes[1]
    k = np.arange(f0 - SHOW_BINS, f0 + SHOW_BINS + 1)
    fz = k * res
    ax.axvspan((f0 - M) * res, (f0 + M) * res, color="#ececec", lw=0)
    ax.axhspan(-1.96, 1.96, color="#dcdcdc", alpha=0.5, lw=0)
    for part, col in ((np.real, C_RE), (np.imag, C_IM)):
        ax.plot(fz, part(c[k]), color=col, lw=0.4, alpha=0.6)
        ax.plot(fz, part(c[k]), ".", color=col, ms=2.5)
    ax.axvline(f0 * res, color=color, lw=0.8, ls=":")
    ax.axhline(0, color=C_INK, lw=0.4)
    lim = max(3.5, 1.1 * np.abs(np.r_[c[k].real, c[k].imag]).max())
    ax.set_ylim(-lim, lim)
    ax.set_xlim(fz[0], fz[-1])
    ax.text(0.98, 0.95, f"CV$^2$ = {cv2(I, lo):.2f}", transform=ax.transAxes, ha="right",
            va="top", fontsize=7)

    ax = axes[2]
    t = np.linspace(0, 2 * np.pi, 200)
    r05 = np.sqrt(2 * stats.f.isf(0.05, 2, 4 * M))     # NFC at p = 0.05
    ax.plot(np.sqrt(2) * np.cos(t), np.sqrt(2) * np.sin(t), color=C_MUTED, lw=0.6, ls=":")
    ax.plot(r05 * np.cos(t), r05 * np.sin(t), color=C_MUTED, lw=0.6, ls="--")
    ax.plot(c[noise].real, c[noise].imag, ".", color=C_IM, ms=3)
    ax.plot(c[f0].real, c[f0].imag, "o", color=color, ms=5, mec=C_INK, mew=0.6)
    lim = max(3.5, 1.1 * np.abs(np.r_[c[noise].real, c[noise].imag]).max())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    ax = axes[3]
    ax.bar(np.arange(1, MAX_LAG + 1), rho, color=color, width=0.8)
    ax.axhline(0, color=C_INK, lw=0.6)
    ax.set_ylim(-1, 1)

    if len(axes) > 4:     # the per-bin panels; fig_real leaves them out (see its docstring)
        ax = axes[4]
        ax.scatter(den, num, s=4, color=color, alpha=0.5, lw=0)
        ax.plot([0, 5], [0, 5], color=C_INK, lw=0.6)
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 5)

        ax = axes[5]
        edges = np.linspace(0, R_MAX, 49)
        ax.hist(np.clip(R, 0, R_MAX - 1e-9), bins=edges, density=True, color=color, alpha=0.8)
        r = np.linspace(0, R_MAX, 300)
        ax.plot(r, null_density(r, M), color=C_INK, lw=1.2)
        ax.set_xlim(0, R_MAX)
        ax.text(0.98, 0.95, f"p < 0.05: {np.mean(p < 0.05):.1%}\n0.2 < p < 0.8: "
                f"{np.mean((p > 0.2) & (p < 0.8)):.0%} (null 60%)",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5)

    for ax in axes:
        _style(ax)
        ax.tick_params(labelsize=6.5)
    if show_titles:
        for ax, h in zip(axes, HEADS):
            old = ax.get_title(loc="left")
            ax.set_title(h + ("\n\n" + old if old else ""), fontsize=8, loc="left")


def label_axes(axes, f_label):
    axes[0].set_xlabel("time (s)", fontsize=7)
    axes[1].set_xlabel(f"frequency (Hz); dotted: {f_label}", fontsize=7)
    axes[1].set_ylabel("coefficient / noise SD", fontsize=7)
    axes[2].set_xlabel("real / noise SD", fontsize=7)
    axes[2].set_ylabel("imaginary / noise SD", fontsize=7)
    axes[3].set_xlabel("L (bins)", fontsize=7)
    if len(axes) <= 4:
        return
    axes[4].set_xlabel("noise power (/ local level)", fontsize=7)
    axes[4].set_ylabel("analysis power (/ local level)", fontsize=7)
    axes[5].set_xlabel("R = analysis power / noise power", fontsize=7)


def fig_toy(out):
    rng = np.random.default_rng(3)
    N, T, f, Q_frac = 1080, 1.0, 0.1, 0.30
    f0, M = sv.window_bins(N, T, f, Q_frac)
    res = 1.0 / (N * T)
    traces = toy_traces(N, T, rng)
    fig, axes = plt.subplots(N_PANELS, len(traces), figsize=(4.0 * len(traces), 2.9 * N_PANELS))
    for j, (name, x) in enumerate(traces):
        draw_row(axes[:, j], x, T, f0, M, res, C_BAD if j < 3 else C_GOOD, name,
                 show_titles=False)
        label_axes(axes[:, j], "0.1 Hz")
    for i, h in enumerate(HEADS):
        axes[i, 0].annotate(f"{'ABCDEF'[i]}. " + h.replace("\n", " "),
                            (0, 1.12 if i else 1.25), xycoords="axes fraction",
                            fontsize=8.5, weight="bold").set_in_layout(False)
    fig.tight_layout(h_pad=3.5, rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_real(out):
    """Ten ROIs, trace to bin correlation (Figure 1 rows A-D). Figure 1's per-bin panels E-F
    are left out: one ROI has only ~420 analysis bins, so its share of p-values in any range
    is uncertain by about as much as the whole effect (~2.5%), and its R histogram looks like
    the null whether or not the ROI is miscalibrated. fig_pooled shows them pooled instead."""
    cfg, F, _ = sv.load(REC, outline=True)
    T, N = cfg.sample_period, F.shape[1]
    f0, M = sv.window_bins(N, T, cfg.analysis.f, cfg.analysis.Q_frac)
    res = 1.0 / (N * T)
    cov = activity(F, T)["coverage"]
    rng = np.random.default_rng(0)
    good = np.flatnonzero(cov >= GOOD_MIN)
    bad = np.flatnonzero((cov >= BAD_RANGE[0]) & (cov < BAD_RANGE[1]))
    print(f"  {REC}: {len(F)} ROIs, M = {M}; {len(good)} with coverage >= {GOOD_MIN}, "
          f"{len(bad)} with {BAD_RANGE[0]} <= coverage < {BAD_RANGE[1]}")
    pick = [(i, C_GOOD) for i in np.sort(rng.choice(good, N_EACH, replace=False))] + \
           [(i, C_BAD) for i in np.sort(rng.choice(bad, N_EACH, replace=False))]
    fig, axes = plt.subplots(len(pick), 4, figsize=(17, 2.3 * len(pick)),
                             gridspec_kw=dict(width_ratios=[2.2, 1.6, 0.9, 1.1]))
    for r, (i, c) in enumerate(pick):
        group = "high coverage" if c == C_GOOD else "low-medium coverage"
        p0 = stats.f.sf(ratios(spectrum(F[i]), np.array([f0]), M)[2], 2, 4 * M)[0]
        draw_row(axes[r], F[i], T, f0, M, res, c,
                 f"ROI {i}, {group}: coverage {cov[i]:.2f}; p at 0.1 Hz = {p0:.2f}",
                 show_titles=(r == 0))
    label_axes(axes[-1], "0.1 Hz")
    for r in range(len(pick)):
        axes[r, 1].set_ylabel("coefficient / noise SD", fontsize=7)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


GROUPS = [("coverage >= 0.8", lambda c: c >= GOOD_MIN, C_GOOD),
          ("0.1 <= coverage < 0.4", lambda c: (c >= BAD_RANGE[0]) & (c < BAD_RANGE[1]), C_BAD),
          ("coverage < 0.1 (dropped in production)", lambda c: c < BAD_RANGE[0], "#8a3a10")]


def fig_pooled(out):
    """The steps of fig_real, pooled over every ROI of the recording in each coverage group:
    one ROI's few hundred analysis bins cannot show a shift of a few percent."""
    cfg, F, _ = sv.load(REC, outline=True)
    T, N = cfg.sample_period, F.shape[1]
    f0, M = sv.window_bins(N, T, cfg.analysis.f, cfg.analysis.Q_frac)
    res = 1.0 / (N * T)
    lo = max(int(np.ceil(F_MIN / res)) - MED_WIN, 1)
    cov = activity(F, T)["coverage"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.9))
    edges = np.linspace(0, 8, 41)
    mid = (edges[:-1] + edges[1:]) / 2
    for name, sel, c in GROUPS:
        idx = np.flatnonzero(sel(cov))
        rho, cv, R = [], [], []
        for i in idx:
            I = spectrum(F[i])
            rho.append(ordinate_corr(I, lo))
            cv.append(cv2(I, lo))
            R.append(ratios(I, analysis_bins(len(I), res, M, f0), M)[2])
        rho, cv, R = np.array(rho), np.array(cv), np.concatenate(R)
        p = stats.f.sf(R, 2, 4 * M)
        lab = f"{name}: {len(idx)} ROIs"
        axes[0].plot(np.arange(1, MAX_LAG + 1), rho.mean(0), color=c, lw=2, label=lab)
        x = np.sort(cv)
        axes[1].plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=2)
        h, _ = np.histogram(R, bins=edges, density=True)
        null = np.diff(stats.f.cdf(edges, 2, 4 * M)) / np.diff(edges)
        axes[2].plot(mid, h / null, color=c, lw=2, marker="o", ms=3)
        axes[3].plot(sv.P_GRID, sv.ecdf_dev(p), color=c, lw=2,
                     label=f"dev@0.5 {np.mean(p <= 0.5) - 0.5:+.3f}")
    axes[0].axhline(0, color=C_INK, lw=0.6)
    axes[0].set(xlabel="L (bins)", ylabel="mean correlation of ordinates L bins apart",
                title="A. Ordinate correlation")
    axes[1].axvline(1, color=C_MUTED, lw=0.6, ls=":")
    axes[1].set(xlim=(0, 2), xlabel="CV$^2$ of the ROI's ordinates (1 = Gaussian noise)",
                ylabel="ECDF over ROIs", title="B. Spread of the ordinates")
    axes[2].axhline(1, color=C_INK, lw=0.6)
    axes[2].set(yscale="log", xlabel="R = analysis power / noise power",
                ylabel="observed density / null density", title="C. Power ratio vs its null")
    axes[3].axhline(0, color=C_INK, lw=0.6)
    axes[3].set(xlabel="p", ylabel="ECDF(p) - p", title="D. p-values at every analysis bin")
    axes[0].legend(fontsize=7, frameon=False)
    axes[3].legend(fontsize=7, frameon=False)
    for ax in axes:
        _style(ax)
        ax.title.set_fontsize(9)
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    fig_toy(_HERE / "fig_gb_ripples_toy.png")
    fig_real(_HERE / "fig_gb_ripples_real.png")
    fig_pooled(_HERE / "fig_gb_ripples_pooled.png")
    print("  wrote fig_gb_ripples_*.png")


if __name__ == "__main__":
    main()
