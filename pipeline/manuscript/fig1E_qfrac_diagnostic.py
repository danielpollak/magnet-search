"""Fig1E diagnostic supplement -- does Q_frac explain the exemplar mag
population's ECDF-deviation dip?

Fig1E (pipeline/manuscript/fig1.py's mag_dist_ax) overlays the null NFC PDF
on the exemplar mag population's (20230413_secondsite, mag3, N=204) NFC
histogram, plus an inset ECDF(p)-p deviation curve. That curve dips to about
-0.09 across p in (0.1, 0.7) instead of hugging 0 -- fewer small/mid p-values
than a uniform null would predict, i.e. this population's NFC values run
systematically a bit low relative to the fitted null.

The working hypothesis going in was that the off-frequency sigma estimate
(mean |c_n|^2 over the Q_frac-determined window around the stimulus
frequency) is inflated -- e.g. by a spectral bump or leakage from the
stimulus frequency into its nearest neighbor bins -- which would deflate NFC
= |c_s|/sigma for every unit and produce exactly this kind of systematic p
-> 1 skew. This script tests that hypothesis two ways, both using the SAME
computation Fig1C/D use (statistics.fourier_analysis + statistics.
plot_spectrum) so the swept configurations are visualized the same way as
the manuscript figure itself:

  1. Q_frac sweep (0.05-0.50, i.e. Q=9-87 off-frequency bins/side): does
     widening/narrowing the window change the dip?
  2. Guard-band sweep (fixed Q_frac=0.30, excluding the innermost 0-8 bins
     on each side of the stimulus frequency): does excluding bins closest
     to (and most likely to leak from) the stimulus frequency change it?

Neither does (page 1: max|ECDF dev| stays ~0.08-0.11 across the whole Q_frac
range; page 2: ~0.09-0.10 across every guard width) -- ruling out "Q_frac is
picking up a nearby noise feature" as the mechanism. Page 3 then checks
whether this dip is even unusual: computed the same signed mean-ECDF-
deviation statistic for every other NPIX null (magnetic) recording in
data/manuscript/all_fourier_df.parquet (72 total, via
statistics.get_poscontrols_negresults's all_mag_exp, restricted to
Pigeon/Quail/zebra finch -- the ephys/NPIX species). The population of
mean-deviations spans -0.052 to +0.072 and is roughly symmetric around 0
(mean +0.004) -- this exemplar's -0.049 IS near the extreme negative end
(only 2/72 = 3% of recordings dip as hard or harder), but it's still well
inside the empirically observed range, not off in some qualitatively
different regime, and recordings of comparable or larger magnitude occur
in the *positive* direction too (up to +0.072). Combined with page 1/2's
Q_frac-insensitivity, this reads as an ordinary (if somewhat unlucky) draw
from finite-N/non-independent-unit sampling noise in the K-S statistic for
a single recording -- not a bug in the null model, and not something any
Q_frac choice can specifically correct. Fig1E's fit doesn't need "fixing";
if anything it's a reminder that a single exemplar recording's own p-value
uniformity is a noisy diagnostic, and the population-level story (Fig2's
aggregate excess-count test) is the one that should carry statistical
weight, not this one panel.

Requires:
  data/20230413_secondsite.nwb            (Fig1's own exemplar experiment)
  data/manuscript/all_fourier_df.parquet  (page 3's cross-recording check;
                                            run pipeline/aggregate.py first)

Usage:
    python pipeline/manuscript/fig1E_qfrac_diagnostic.py
    python pipeline/manuscript/fig1E_qfrac_diagnostic.py --out-dir figs/paper
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from ecdfbounds import bootstrap_ecdf_band
from magpyneto2 import statistics
from magpyneto2.statistics import (
    fourier_analysis, fit_fourier_sig, corrected_pvalues, get_epsilon,
    normalized_Fourier_PDF, normalized_Fourier_PDF_corrected,
    bins_for_fraction, get_poscontrols_negresults, plot_spectrum,
)
from pipeline import nwb_io
from pipeline.schema import load_experiment

import format_parameters as FP

EXPERIMENT = "20230413_secondsite"
MAG_CONTINGENCY = "2023-04-13_17-06-27_W25R_second_site_mag3_inclined"  # see fig1.py's docstring (2026-09-02 swap)
MAG_FREQ = 3.0
CLUSTER_ID = 540  # same exemplar unit as fig1.py

Q_FRACS = [0.05, 0.10, 0.15, 0.30, 0.50]
GUARD_Q_FRAC = 0.30
GUARD_BANDS = [0, 2, 5, 8]

HIST_BINS = np.arange(0, 12, 0.2)


def load_mag_population():
    cfg = load_experiment(Path(__file__).parent.parent.parent / "experiments" / f"{EXPERIMENT}.yml")
    io_r, nwbfile = nwb_io.read_nwbfile(
        str(Path(FP.DATA_DIR) / f"{EXPERIMENT}.nwb"))
    modulation_df = nwb_io.build_modulation_frame(nwbfile, good_only=cfg.good)
    io_r.close()
    return modulation_df.loc[modulation_df.rec == MAG_CONTINGENCY].copy()


def _spectrum_panel(ax, ff_alt, fou_alt, keep_mask, freq, fou0, title):
    """Fig1C/D's own plot_spectrum, restricted to the kept off-frequency
    bins -- excluded bins are drawn first, in light gray, underneath, so the
    swept window shows up as "which dots turned gray", not a different plot.
    """
    if np.any(~keep_mask):
        ax.plot(ff_alt[~keep_mask], np.abs(fou_alt[~keep_mask]), ".",
                color="lightgray", alpha=0.6, markersize=1, zorder=0)
    plot_spectrum(ax, fou_alt[keep_mask], ff_alt[keep_mask], freq, fou0,
                  dot_color=FP.COLOR_MAG, stem_color="black", sigma_color="tab:orange")
    ax.set_title(title, fontsize=FP.FS_TITLE)
    ax.set_xlabel("Freq (Hz)", fontsize=FP.FS_LEGEND)
    ax.set_ylabel("Magnitude", fontsize=FP.FS_LEGEND)
    ax.tick_params(labelsize=FP.FS_LEGEND)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _hist_pdf_panel(ax, NFC, Q, title):
    vals, bins = np.histogram(NFC, bins=HIST_BINS, density=True)
    ax.bar(bins[:-1], vals, width=np.diff(bins)[0], align="edge", color=FP.COLOR_MAG, zorder=2)
    XX, YY = normalized_Fourier_PDF()
    ax.plot(XX, YY, color="k", linewidth=1, label="uncorrected null")
    eps = get_epsilon(Q)
    if eps > 0:
        YYc = normalized_Fourier_PDF_corrected(XX, XX, YY, eps)
        ax.plot(XX, YYc, color="tab:orange", linewidth=1, linestyle="--", label="corrected null")
    ax.set_xlim(0, 9)
    ax.set_title(title, fontsize=FP.FS_TITLE)
    ax.set_xlabel("NFC", fontsize=FP.FS_LEGEND)
    ax.set_ylabel("PDF", fontsize=FP.FS_LEGEND)
    ax.tick_params(labelsize=FP.FS_LEGEND)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _ecdf_panel(ax, pvals, title):
    """Same construction as fig1.py's ecdf_mag_ax (see plot_fig1_composite)."""
    x, lower, upper = bootstrap_ecdf_band(pvals, alpha=0.05)
    ecdf = (np.arange(1, len(pvals) + 1)) / len(pvals)
    ks_dev = ecdf - x
    ks_lower = lower - x
    ks_upper = upper - x
    ax.plot(x, ks_dev, color=FP.COLOR_MAG, linewidth=FP.LW_TRACE)
    ax.fill_between(x, ks_lower, ks_upper, color=FP.COLOR_MAG, alpha=FP.ALPHA_CONFIDENCE)
    ax.axhline(0, color=FP.COLOR_NULL, linestyle="--", linewidth=FP.LW_REFERENCE, alpha=0.6)
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlim(0, 1)
    ax.set_title(f"{title}\nmax|dev|={np.max(np.abs(ks_dev)):.3f}", fontsize=FP.FS_TITLE)
    ax.set_xlabel("p-value", fontsize=FP.FS_LEGEND)
    ax.set_ylabel("ECDF dev.", fontsize=FP.FS_LEGEND)
    ax.tick_params(labelsize=FP.FS_LEGEND)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_qfrac_sweep_page(pdf, mag_df, spk_540, group_T):
    """Page 1: sweep Q_frac itself."""
    fig, axes = plt.subplots(3, len(Q_FRACS), figsize=(3.2 * len(Q_FRACS), 8.5))
    resolution = 1 / group_T

    # Widest window drives the exemplar spectrum panels -- every smaller
    # Q_frac's window is a contiguous inner subset of it (same i0, same T),
    # so one fourier_analysis call on the exemplar unit covers every column.
    Q_widest = bins_for_fraction(MAG_FREQ, max(Q_FRACS), resolution, context="[diagnostic] ")
    (_, _, _, _, i0, ff_alt_wide, fou0_540, fou_alt_wide, _, _) = fourier_analysis(
        [spk_540], MAG_FREQ, Q=Q_widest, sr=30_000, T=group_T)
    fou_alt_wide = fou_alt_wide.flatten()
    mid = Q_widest  # ff_alt_wide has Q_widest entries below i0, then Q_widest above

    for j, qf in enumerate(Q_FRACS):
        M = bins_for_fraction(MAG_FREQ, qf, resolution, context="[diagnostic] ")
        keep = np.zeros(len(ff_alt_wide), dtype=bool)
        keep[mid - M:mid + M] = True

        _spectrum_panel(axes[0, j], ff_alt_wide, fou_alt_wide, keep, MAG_FREQ, fou0_540,
                         f"Q_frac={qf} (Q={M})")

        fdf, _ = fit_fourier_sig(mag_df, Q_frac=qf, diagnostics=False)
        _hist_pdf_panel(axes[1, j], fdf["NFC"].values, fdf["Q"].iloc[0], f"N={len(fdf)}")
        _ecdf_panel(axes[2, j], fdf["p_value"].values, "")

    fig.suptitle("Fig1E diagnostic (1/3) -- Q_frac sweep, exemplar unit %d, %s"
                 % (CLUSTER_ID, MAG_CONTINGENCY), fontsize=FP.FS_TITLE + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def plot_guardband_sweep_page(pdf, mag_df, spk_540, group_T):
    """Page 2: fixed Q_frac, exclude an inner guard band around the stimulus
    frequency (tests spectral-leakage-into-nearest-bins as the mechanism)."""
    fig, axes = plt.subplots(3, len(GUARD_BANDS), figsize=(3.2 * len(GUARD_BANDS), 8.5))
    resolution = 1 / group_T
    M = bins_for_fraction(MAG_FREQ, GUARD_Q_FRAC, resolution, context="[diagnostic] ")

    (_, _, _, _, i0, ff_alt, fou0_540, fou_alt, _, _) = fourier_analysis(
        [spk_540], MAG_FREQ, Q=M, sr=30_000, T=group_T)
    fou_alt = fou_alt.flatten()
    mid = len(ff_alt) // 2

    fdf_full, log_dict = fit_fourier_sig(mag_df, Q_frac=GUARD_Q_FRAC, diagnostics=False)
    key = (MAG_CONTINGENCY, MAG_FREQ)
    fou_alt_c_pop = log_dict[key]["fou_alt_c"]   # (n_units, 2M, 2)
    fou0_pop = log_dict[key]["fou0"]
    n_alt_pop = fou_alt_c_pop.shape[1]
    mid_pop = n_alt_pop // 2

    for j, g in enumerate(GUARD_BANDS):
        keep = np.ones(len(ff_alt), dtype=bool)
        keep[mid - g:mid + g] = False
        _spectrum_panel(axes[0, j], ff_alt, fou_alt, keep, MAG_FREQ, fou0_540,
                         f"guard={g} bins/side")

        keep_pop = np.ones(n_alt_pop, dtype=bool)
        keep_pop[mid_pop - g:mid_pop + g] = False
        fac = fou_alt_c_pop[:, keep_pop, :]
        complex_fac = fac[:, :, 0] + fac[:, :, 1] * 1j
        sgm = np.sqrt(0.5 * np.mean(np.abs(complex_fac) ** 2, axis=1))
        NFC = np.abs(fou0_pop.flatten()) / sgm
        Q_eff = keep_pop.sum() // 2
        pvals = corrected_pvalues(NFC, Q_eff)

        _hist_pdf_panel(axes[1, j], NFC, Q_eff, f"N={len(NFC)}")
        _ecdf_panel(axes[2, j], pvals, "")

    fig.suptitle("Fig1E diagnostic (2/3) -- guard-band sweep (Q_frac=%.2f fixed), "
                 "exemplar unit %d, %s" % (GUARD_Q_FRAC, CLUSTER_ID, MAG_CONTINGENCY),
                 fontsize=FP.FS_TITLE + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def plot_cross_recording_page(pdf, this_rec_mean_dev):
    """Page 3: is this exemplar's dip unusual among every other NPIX null
    (magnetic) recording, or just ordinary per-recording sampling noise?"""
    df = pd.read_parquet(FP.PARQUET_PATH)
    all_mag_exp, _, _ = get_poscontrols_negresults(df)
    npix_species = {"Pigeon", "Quail", "zebra finch"}
    npix_mag = all_mag_exp.loc[all_mag_exp.species.isin(npix_species)]

    rows = []
    for key, g in npix_mag.groupby(["species", "ID", "date", "rec", "freq"]):
        p = g["p_value"].dropna().values
        if len(p) < 50:
            continue
        sorted_p = np.sort(p)
        ecdf = (np.arange(1, len(p) + 1)) / len(p)
        rows.append(np.mean(ecdf - sorted_p))
    mean_devs = np.array(rows)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(mean_devs, bins=20, color=FP.COLOR_MAG, alpha=0.75)
    ax.axvline(this_rec_mean_dev, color="black", linewidth=1.5,
               label=f"Fig1E exemplar rec\n(mean dev={this_rec_mean_dev:.3f})")
    ax.axvline(0, color=FP.COLOR_NULL, linestyle="--", linewidth=FP.LW_REFERENCE)
    n_as_negative = int(np.sum(mean_devs <= this_rec_mean_dev))
    frac_as_negative = n_as_negative / len(mean_devs)
    ax.set_xlabel("mean(ECDF(p) - p) per recording")
    ax.set_ylabel("# NPIX null recordings")
    ax.set_title(
        f"Fig1E diagnostic (3/3) -- cross-recording check\n"
        f"{len(mean_devs)} NPIX null (magnetic) recordings (Pigeon/Quail/zebra finch)\n"
        f"{n_as_negative}/{len(mean_devs)} ({100*frac_as_negative:.0f}%) dip as hard or "
        f"harder -- range spans {mean_devs.min():.3f} to {mean_devs.max():.3f}",
        fontsize=FP.FS_TITLE)
    ax.legend(fontsize=FP.FS_LEGEND, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=FP.OUT_DIR, help="Output directory for the PDF")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mag_df = load_mag_population()
    group_T = (1 / MAG_FREQ) * np.max(mag_df.period)
    spk_540 = np.sort(mag_df.loc[mag_df.id == CLUSTER_ID, "spk"].values)

    # This exemplar's own mean-ECDF-deviation, at the production Q_frac
    # (0.15, per experiments/20230413_secondsite.yml's mag_Q_frac), for
    # page 3's marker.
    fdf_prod, _ = fit_fourier_sig(mag_df, Q_frac=0.15, diagnostics=False)
    p_prod = fdf_prod["p_value"].values
    sorted_p = np.sort(p_prod)
    ecdf = (np.arange(1, len(p_prod) + 1)) / len(p_prod)
    this_rec_mean_dev = np.mean(ecdf - sorted_p)

    out_path = out_dir / "Fig1E_qfrac_diagnostic.pdf"
    with PdfPages(out_path) as pdf:
        plot_qfrac_sweep_page(pdf, mag_df, spk_540, group_T)
        plot_guardband_sweep_page(pdf, mag_df, spk_540, group_T)
        plot_cross_recording_page(pdf, this_rec_mean_dev)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
