"""
Analysis-stage diagnostic plots.

plot_analysis_diagnostics  — single multi-page PDF, three pages per (rec, freq) group:
  Page 1 (2×2): [power spectrum]  [c-hat histogram]
                [coefficient CDF] [magnitude PDF  ]
  Page 2 (2×1): Fourier coefficient spectrum (real + imaginary vs frequency)
  Page 3 (4×1): Moments of off-frequency coefficients vs firing rate
All axes content is rasterized.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from magpyneto2.statistics import (
    draw_hist,
    inset_hist,
    plot_coefficient_cdf,
    plot_magnitude_pdf,
    plot_power_by_freq,
    Moments_vs_FR,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_analysis_diagnostics(cfg, modulation_df, fourier_df, log_dict, save_dir):
    """Write one multi-page PDF of analysis diagnostics for an experiment.

    Parameters
    ----------
    cfg          : ExperimentConfig
    modulation_df: pd.DataFrame  columns [period, spk, phase, freq, id, rec]
    fourier_df   : pd.DataFrame  columns include [rr, rec, freq]
    log_dict     : dict  {(rec, freq): {"ff_alt": ..., "fou_alt": ..., ...}}
    save_dir     : Path-like
    """
    save_dir = Path(save_dir)
    out_path = save_dir / f"{cfg.name}_analysis_diagnostics.pdf"

    groups = list(fourier_df.groupby(["rec", "freq"]))

    with PdfPages(out_path) as pdf:
        for (rec, freq), fdf_group in groups:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle(f"{cfg.name}  |  {rec}  |  {freq} Hz", fontsize=9)

            ax_psd, ax_hist, ax_cdf, ax_pdf = axes.flat

            # ── Power spectrum ───────────────────────────────────────────────
            mod_mask = (modulation_df.rec == rec) & (modulation_df.freq == freq)
            spks = [g["spk"].values
                    for _, g in modulation_df.loc[mod_mask].groupby("id")
                    if len(g) > 0]
            _fill_power_spectra(ax_psd, spks, freq)
            ax_psd.set_title("Power spectrum")

            # ── c-hat histogram ──────────────────────────────────────────────
            rr = fdf_group["rr"].values
            _fill_chat_hist(ax_hist, rr)
            ax_hist.set_title(f"c-hat distribution  (N={len(rr)})")

            # ── Coefficient CDF and magnitude PDF ────────────────────────────
            entry = log_dict.get((rec, freq), {})
            ff_alt = entry.get("ff_alt")
            fou_alt = entry.get("fou_alt")

            if ff_alt is not None and fou_alt is not None and len(fou_alt) > 0:
                kk = _subsample_units(len(fou_alt), max_units=20)
                _fill_coeff_cdf(ax_cdf, ff_alt, fou_alt, kk=kk)
                _fill_mag_pdf(ax_pdf, ff_alt, fou_alt, kk=kk)
            else:
                for ax in (ax_cdf, ax_pdf):
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes)

            ax_cdf.set_title("Coefficient CDF")
            ax_pdf.set_title("Magnitude PDF")

            # ── Rasterize all axes content ───────────────────────────────────
            for ax in axes.flat:
                _rasterize_ax(ax)

            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            # ── Page 2: Fourier coefficient spectrum (power_by_freq) ─────────
            entry = log_dict.get((rec, freq), {})
            if all(k in entry for k in ("ff_alt", "fou_alt", "fou0")):
                fig2, axes2 = plt.subplots(2, 1, figsize=(10, 6))
                fig2.suptitle(
                    f"{cfg.name}  |  {rec}  |  {freq} Hz  —  Fourier spectrum",
                    fontsize=9)
                kk2 = _subsample_units(len(entry["fou_alt"]), max_units=30)
                try:
                    plot_power_by_freq(
                        entry["ff_alt"], entry["fou_alt"], entry["fou0"],
                        freq, kk=kk2, axes=axes2)
                except Exception as exc:
                    axes2[0].text(0.5, 0.5, f"error:\n{exc}", ha="center",
                                  va="center", transform=axes2[0].transAxes,
                                  fontsize=7)
                for ax in axes2:
                    _rasterize_ax(ax)
                fig2.tight_layout()
                pdf.savefig(fig2, dpi=150)
                plt.close(fig2)

            # ── Page 3: Moments vs firing rate ───────────────────────────────
            if all(k in entry for k in ("nn", "fou_alt_c", "T")):
                try:
                    fig3_axes = Moments_vs_FR(
                        entry["nn"], entry["fou_alt_c"], entry["T"])
                    fig3 = fig3_axes[0].get_figure()
                    fig3.suptitle(
                        f"{cfg.name}  |  {rec}  |  {freq} Hz  —  Moments vs FR",
                        fontsize=9)
                    for ax in fig3_axes:
                        _rasterize_ax(ax)
                    fig3.tight_layout()
                    pdf.savefig(fig3, dpi=150)
                    plt.close(fig3)
                except Exception as exc:
                    fig3, ax3 = plt.subplots(figsize=(5, 8))
                    ax3.text(0.5, 0.5, f"Moments_vs_FR error:\n{exc}",
                             ha="center", va="center",
                             transform=ax3.transAxes, fontsize=7)
                    pdf.savefig(fig3, dpi=150)
                    plt.close(fig3)


# ---------------------------------------------------------------------------
# Per-axes fill helpers
# ---------------------------------------------------------------------------

def _fill_power_spectra(ax, spks, freq, f_lo=0.3, f_hi=20, df=0.3):
    dt = 0.02
    yyy = []
    if spks:
        t_all = np.concatenate(spks)
        T = t_all.max() - t_all.min()
        xx = np.arange(0, T + 0.0001, dt)
        for tt in spks:
            if len(tt) > 1000:
                yy, _ = np.histogram(tt, xx)
                yyy.append(yy - np.mean(yy))
            if len(yyy) > 10:
                break

    if not yyy:
        ax.text(0.5, 0.5, "insufficient spikes\n(need >1000 per unit)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (a.u.)")
        return

    N = len(yyy[0])
    ff = np.fft.rfftfreq(N, d=dt)
    pxx = np.array([np.abs(np.fft.rfft(y)) ** 2 / N for y in yyy]).T

    use = (ff >= f_lo) & (ff <= f_hi)
    yvals = pxx[use]
    if yvals.size > 0:
        ymin, ymax = yvals.min(), yvals.max()
        ax.fill_between([freq - df, freq + df], [ymin, ymin], [ymax, ymax],
                        facecolor=(0.7, 0.8, 1), zorder=0,
                        label=f"{freq} Hz", rasterized=True)

    for k in range(pxx.shape[1]):
        ax.loglog(ff[use], pxx[use, k], alpha=0.6, linewidth=0.8,
                  rasterized=True)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (a.u.)")


def _fill_chat_hist(ax, rr):
    if len(rr) < 2:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return
    vals, bins = draw_hist(rr, ax, xlim=9, inset=False)
    inset_hist(ax, vals, bins)


def _subsample_units(n_units, max_units=20, seed=0):
    """Return indices for up to max_units evenly-spaced units."""
    if n_units <= max_units:
        return np.arange(n_units)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_units, size=max_units, replace=False))


def _fill_coeff_cdf(ax, ff_alt, fou_alt, kk=None):
    try:
        plot_coefficient_cdf(ff_alt, fou_alt, kk=kk, ax=ax)
    except Exception as exc:
        ax.text(0.5, 0.5, f"error:\n{exc}", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, wrap=True)


def _fill_mag_pdf(ax, ff_alt, fou_alt, kk=None):
    # plot_magnitude_pdf uses plt.plot/plt.bar (implicit current-axes),
    # so set the current axes first.
    plt.sca(ax)
    try:
        plot_magnitude_pdf(ff_alt, fou_alt, kk=kk, ax=ax)
    except Exception as exc:
        ax.text(0.5, 0.5, f"error:\n{exc}", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, wrap=True)


def _rasterize_ax(ax):
    """Rasterize every artist in an axes."""
    for artist in ax.get_children():
        try:
            artist.set_rasterized(True)
        except AttributeError:
            pass
