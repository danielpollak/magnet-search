"""
Analysis-stage diagnostic plots.

plot_analysis_diagnostics  — single multi-page PDF, up to six pages per (rec, freq) group:
  Page 1 (2×2): [power spectrum]  [NFC histogram]
                [coefficient CDF] [magnitude PDF  ]
  Page 2 (1×1): Spike raster (units sorted by NFC, period boundaries marked)
  Page 3 (2×1): Fourier coefficient spectrum (real + imaginary vs frequency)
  Page 4 (4×1): Moments of off-frequency coefficients vs firing rate
  Page 5 (1×1): 2F NFC histogram
  Page 6 (2×1): 2F Fourier coefficient spectrum (only when per-unit 2F
                coefficients are available -- see that page's own comment)
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
    get_epsilon,
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
    fourier_df   : pd.DataFrame  columns include [NFC, rec, freq]
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

            # ── NFC histogram ─────────────────────────────────────────────────
            NFC = fdf_group["NFC"].values
            Q = fdf_group["Q"].iloc[0] if "Q" in fdf_group.columns and len(fdf_group) else None
            _fill_NFC_hist(ax_hist, NFC, Q=Q)
            ax_hist.set_title(f"NFC distribution  (N={len(NFC)})")

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

            # ── Page 2: Spike raster (sorted by NFC, period boundaries) ──────
            NFC_by_id = dict(zip(fdf_group["id"], fdf_group["NFC"]))
            id_spk_pairs = [(id_, g["spk"].values)
                            for id_, g in modulation_df.loc[mod_mask].groupby("id")
                            if len(g) > 0]
            id_spk_pairs.sort(key=lambda p: NFC_by_id.get(p[0], -1), reverse=True)

            fig_r, ax_r = plt.subplots(figsize=(10, 6))
            fig_r.suptitle(f"{cfg.name}  |  {rec}  |  {freq} Hz  —  spike raster "
                           f"(sorted by NFC, high → low)", fontsize=9)
            _fill_spike_raster(ax_r, id_spk_pairs, freq)
            _rasterize_ax(ax_r)
            fig_r.tight_layout()
            pdf.savefig(fig_r, dpi=150)
            plt.close(fig_r)

            # ── Page 3: Fourier coefficient spectrum (power_by_freq) ─────────
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

            # ── Page 4: Moments vs firing rate ───────────────────────────────
            if all(k in entry for k in ("spk_count", "fou_alt_c", "T")):
                try:
                    fig3_axes = Moments_vs_FR(
                        entry["spk_count"], entry["fou_alt_c"], entry["T"])
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

            # ── Page 5: 2F NFC histogram ──────────────────────────────────────
            # Mirrors Page 1's NFC histogram but for the 2nd harmonic. The
            # 2F group has its own bin count M_2f (~= 2*M_1f under the
            # Q_frac policy -- see fit_fourier_sig), looked up from the twoF_
            # log_dict entry rather than fourier_df's "Q" column (which only
            # ever holds the 1F bin count, matching what it already meant
            # pre-Q_frac -- see .claude/plans).
            twoF_key = ("twoF_" + rec, "twoF_" + str(freq))
            twoF_entry = log_dict.get(twoF_key, {})
            NFC_2f_all = fdf_group["2f_NFC"].values if "2f_NFC" in fdf_group.columns else np.array([])
            NFC_2f = NFC_2f_all[~np.isnan(NFC_2f_all)] if len(NFC_2f_all) else NFC_2f_all

            if len(NFC_2f) >= 2:
                fig5, ax5 = plt.subplots(figsize=(7, 5))
                fig5.suptitle(f"{cfg.name}  |  {rec}  |  {freq * 2} Hz (2F)  —  "
                              f"NFC distribution", fontsize=9)
                _fill_NFC_hist(ax5, NFC_2f, Q=twoF_entry.get("M"))
                ax5.set_title(f"N = {len(NFC_2f)} units")
                _rasterize_ax(ax5)
                fig5.tight_layout()
                pdf.savefig(fig5, dpi=150)
                plt.close(fig5)

                # ── Page 6: 2F Fourier coefficient spectrum ──────────────────
                # Per-unit 2F Fourier coefficients (fou_alt/fou0) would need
                # log_dict straight from fit_fourier_sig() -- but BOTH
                # simple.py and multistim.py rebuild their diagnostics input
                # from the just-written NWB file (nwb_io.
                # read_log_dict_equivalent), which only persists 1F per-unit
                # coefficients (see that function's docstring), so this page
                # never renders under the current NWB schema for any NPIX
                # call site. Kept as a graceful skip (matching how Pages 3/4
                # already skip when their own data is missing) rather than
                # deleted outright, in case a future caller passes a
                # fit_fourier_sig()-sourced log_dict directly.
                if all(k in twoF_entry for k in ("ff_alt", "fou_alt", "fou0")):
                    fig6, axes6 = plt.subplots(2, 1, figsize=(10, 6))
                    fig6.suptitle(
                        f"{cfg.name}  |  {rec}  |  {freq * 2} Hz (2F)  —  "
                        f"Fourier spectrum", fontsize=9)
                    kk6 = _subsample_units(len(twoF_entry["fou_alt"]), max_units=30)
                    try:
                        plot_power_by_freq(
                            twoF_entry["ff_alt"], twoF_entry["fou_alt"],
                            twoF_entry["fou0"], freq * 2, kk=kk6, axes=axes6)
                    except Exception as exc:
                        axes6[0].text(0.5, 0.5, f"error:\n{exc}", ha="center",
                                      va="center", transform=axes6[0].transAxes,
                                      fontsize=7)
                    for ax in axes6:
                        _rasterize_ax(ax)
                    fig6.tight_layout()
                    pdf.savefig(fig6, dpi=150)
                    plt.close(fig6)


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


def _fill_spike_raster(ax, id_spk_pairs, freq, max_units=80, max_periods=20):
    """Rasterized spike-raster plot: one row per unit (in the order given by
    the caller -- e.g. sorted by NFC descending), spike times on the
    x-axis, with vertical dashed lines marking stimulus period boundaries so
    stimulus-locked (or not) structure in the raw spikes can be compared by
    eye against the reported Fourier result. Restricted to `max_periods`
    periods (from the first spike) so long recordings still render as a
    legible raster instead of a solid smear.
    """
    if not id_spk_pairs:
        ax.text(0.5, 0.5, "no spikes", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return

    period_s = 1.0 / freq
    pairs = id_spk_pairs[:max_units]
    nonempty = [s for _, s in pairs if len(s) > 0]
    if not nonempty:
        ax.text(0.5, 0.5, "no spikes", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return
    t0 = min(s.min() for s in nonempty)
    t_window_end = t0 + max_periods * period_s

    spk_list = [s[(s >= t0) & (s <= t_window_end)] - t0 for _, s in pairs]

    ax.eventplot(spk_list, lineoffsets=np.arange(len(spk_list)), linelengths=0.8,
                 colors="k", linewidths=0.5, rasterized=True)

    n_periods_drawn = int(np.ceil((t_window_end - t0) / period_s))
    for k in range(n_periods_drawn + 1):
        ax.axvline(k * period_s, color="crimson", lw=0.6, alpha=0.6,
                   linestyle="--", zorder=0, rasterized=True)

    ax.set_xlim(0, t_window_end - t0)
    ax.set_ylim(-1, len(spk_list))
    ax.set_xlabel(f"Time (s), relative to first spike -- dashed lines every "
                  f"period ({period_s:.2f} s)")
    ax.set_ylabel(f"Unit (N={len(spk_list)} of {len(id_spk_pairs)} shown)")


def _fill_NFC_hist(ax, NFC, Q=None):
    if len(NFC) < 2:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return
    eps = get_epsilon(Q) if Q else None
    vals, bins = draw_hist(NFC, ax, xlim=9, inset=False, eps=eps)
    inset_hist(ax, vals, bins, eps=eps)


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
