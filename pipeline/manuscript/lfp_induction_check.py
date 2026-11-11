# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (magneto2)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Does the magnet induce a voltage in the Neuropixels probe?
#
# If the coil's changing field induced a voltage in the probe or its leads, every
# channel would carry a sharp line at the stimulus frequency, present only while
# the magnet runs. So we compare the LFP power spectrum around 3 Hz in two
# equally long stretches of one raw recording: just before the magnet turns on,
# and while it runs.
#
# **Recording:** pigeon W25R, hippocampus, `20230413` second site, `mag3`
# (3 Hz). This is the same session as Fig 1's exemplar unit.
#
# **Data:** the Neuropixels LFP band (`Neuropix-PXI-103.1`, 2500 Hz, all 384
# channels), read straight from the raw Open Ephys folder on the NAS. The
# coil's drive signal is recorded on channel 0 of the NI-DAQ (1000 Hz); we use it
# only to find when the magnet turns on.
#
# **Output:** `figs/paper/diagnostic/LFP_induction_spectrum.pdf` and
# `figs/paper/diagnostic/LFP_induction_noise.pdf`

# %%
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from ephysio import openEphysIO

import format_parameters as FP

RECORDING = "//datanas/family/data_raw/20230413/second_site/2023-04-13_17-17-14_W25R_second_site_mag3"
STIM_FREQ = 3.0     # Hz
SEGMENT_S = 28      # length of each stretch, s (a whole number of 3 Hz cycles)
GAP_S = 2           # keep this far from the magnet's onset, s

# %% [markdown]
# ## Load the recording

# %%
ldr = openEphysIO.Loader(RECORDING, cntlbarcodes=True)

lfp_sr = ldr.samplingrate(ldr.lfpstream())                      # 2500 Hz
lfp_uv = ldr.data(ldr.lfpstream()) * ldr.bitvolts(ldr.lfpstream())[0]  # samples x 384, in uV

daq_sr = ldr.samplingrate(ldr.nidaqstream())                    # 1000 Hz
coil = ldr.data(ldr.nidaqstream())[:, 0].astype(float)          # coil drive signal

print(f"LFP: {lfp_uv.shape[1]} channels, {lfp_uv.shape[0] / lfp_sr:.1f} s")
print(f"NI-DAQ: {coil.size / daq_sr:.1f} s")

# %% [markdown]
# ## Find when the magnet turns on
#
# The drive signal is flat (peak-to-peak ~20 ADC units) until the magnet starts,
# then swings by ~4000. The first second whose peak-to-peak range exceeds 1000 is
# the onset. The LFP and NI-DAQ streams start together to within a few
# milliseconds, which doesn't matter for picking 28 s stretches with a 2 s
# margin.

# %%
n_seconds = coil.size // daq_sr
ptp_per_second = np.array([np.ptp(coil[i * daq_sr:(i + 1) * daq_sr]) for i in range(n_seconds)])
onset_s = int(np.argmax(ptp_per_second > 1000))
print(f"Magnet turns on at ~{onset_s} s")

off_window = (onset_s - GAP_S - SEGMENT_S, onset_s - GAP_S)
on_window = (onset_s + GAP_S, onset_s + GAP_S + SEGMENT_S)
print(f"Magnet off: {off_window} s, magnet on: {on_window} s")

# Check: during the on stretch, the drive signal's strongest frequency is 3 Hz.
on_coil = coil[on_window[0] * daq_sr:on_window[1] * daq_sr]
f_coil, p_coil = signal.periodogram(on_coil, fs=daq_sr, detrend="constant")
print(f"Drive signal peaks at {f_coil[np.argmax(p_coil)]:.3f} Hz")


# %% [markdown]
# ## Expected induction from the coil (Faraday's law)
#
# Suppose the field oscillates as $B(t) = B_0 \sin(2\pi f t)$ everywhere inside
# the recording circuit, and that circuit encloses an area $A$. It picks up an
# EMF
#
# $$
# \mathcal{E}(t) = -\frac{d\Phi}{dt} = -A\,\frac{dB}{dt} = -2\pi f\, B_0 A \cos(2\pi f t),
# $$
#
# a sine of amplitude
#
# $$
# \mathcal{E}_0 = 2\pi f\, B_0\, A .
# $$
#
# We take generous values for both unknowns:
#
# - $A = \pi (12.5\ \text{mm})^2 \approx 4.9\ \text{cm}^2$, the cross-section of a
#   pigeon's head, as large as the loop between recording site and reference
#   could be;
# - $B_0 = 1000\ \mu\text{T}$, about 10 times the field we actually applied, and
#   over the whole of $A$, although the coil's field falls off roughly as
#   $1/r^3$.
#
# $$
# \mathcal{E}_0 = 2\pi \times 3\ \text{Hz} \times 10^{-3}\ \text{T} \times 4.9 \times 10^{-4}\ \text{m}^2
# \approx 9\ \mu\text{V}.
# $$
#
# So $9\ \mu\text{V}$ is an upper bound; the field we actually applied would
# induce about a tenth of it.

# %%
LOOP_RADIUS_M = 12.5e-3   # pigeon head cross-section
B0_T = 1000e-6            # ~10x the applied field

loop_area_m2 = np.pi * LOOP_RADIUS_M**2
induced_uv = 2 * np.pi * STIM_FREQ * B0_T * loop_area_m2 * 1e6
print(f"Loop area {loop_area_m2 * 1e4:.1f} cm² -> induced sine amplitude {induced_uv:.1f} uV")

# %% [markdown]
# ## Power spectra
#
# Per channel, we take a plain periodogram of each 28 s stretch (no taper;
# 28 s holds exactly 84 cycles of 3 Hz, so a 3 Hz line falls on one bin of
# width $1/T = 1/28$ Hz). This is the power spectral density $P(f)$, in
# $\mu\text{V}^2/\text{Hz}$.
#
# A sine of amplitude $a$ puts all its power, $a^2/2$, into its one bin, so
# its power spectral density there is $a^2 T / 2$. The induced sine's upper
# bound is therefore
#
# $$
# P_\mathcal{E} = \frac{\mathcal{E}_0^2\, T}{2} .
# $$
#
# The LFP's power, in contrast, is spread over all frequencies, so the sine
# competes only with the LFP's power in that one bin, not with its broadband
# variance.

# %%
def power_spectra(window):
    """Power spectral density (uV^2/Hz) per frequency bin and channel."""
    start, stop = (int(t * lfp_sr) for t in window)
    return signal.periodogram(lfp_uv[start:stop], fs=lfp_sr, detrend="constant", axis=0)

freqs, psd_off = power_spectra(off_window)   # frequencies x channels
_, psd_on = power_spectra(on_window)

induced_psd = induced_uv**2 * SEGMENT_S / 2   # uV^2/Hz in the 3 Hz bin

# Sanity check: a 10 uV, 3 Hz sine gives 10^2 * 28 / 2 = 1400 uV^2/Hz in its bin.
t = np.arange(SEGMENT_S * lfp_sr) / lfp_sr
f_test, p_test = signal.periodogram(10 * np.sin(2 * np.pi * STIM_FREQ * t), fs=lfp_sr)
print(f"Test sine: {p_test.max():.0f} uV^2/Hz at {f_test[p_test.argmax()]:.3f} Hz (expected 1400)")
print(f"Max. induction: {induced_psd:.0f} uV^2/Hz in the 3 Hz bin")

# %% [markdown]
# Two summaries go into the figure:
#
# - **Spectrum (left):** each bin's power, averaged over channels. If every
#   channel carried the induced sine, the 3 Hz bin would sit at about
#   $P_\text{noise} + P_\mathcal{E}$.
# - **Noise distribution (right):** the power of every channel's bins from 2
#   to 4 Hz, except 3 Hz itself, with the magnet off. This is the local noise
#   band an induced 3 Hz sine would have to stand out from. Alongside it, the
#   3 Hz bin of every channel, with the magnet on and with it off. The LFP is
#   largely shared across channels, so a single bin's powers cluster together
#   across channels; the magnet-off 3 Hz bin shows how far such a cluster sits
#   from the noise median by chance alone.

# %%
stim_bin = np.argmin(np.abs(freqs - STIM_FREQ))
local = (freqs >= 2) & (freqs <= 4) & (np.arange(freqs.size) != stim_bin)

noise_psd = psd_off[local].ravel()                  # every channel, every local bin
on_3hz_psd = psd_on[stim_bin]                       # every channel, 3 Hz, magnet on
off_3hz_psd = psd_off[stim_bin]                     # every channel, 3 Hz, magnet off

print(f"Local noise (2-4 Hz, magnet off): median {np.median(noise_psd):.0f} uV^2/Hz, "
      f"95th percentile {np.percentile(noise_psd, 95):.0f} uV^2/Hz")
print(f"3 Hz bin, median over channels: magnet on {np.median(on_3hz_psd):.0f} uV^2/Hz, "
      f"magnet off {np.median(off_3hz_psd):.0f} uV^2/Hz")
print(f"Share of noise bins above the {induced_psd:.0f} uV^2/Hz prediction: "
      f"{np.mean(noise_psd > induced_psd):.0%}")

# %% [markdown]
# ## Figure 1: spectrum

# %%
matplotlib.rc("font", family=FP.FONT_FAMILY, size=FP.FS_BODY)

fig, ax = plt.subplots(figsize=(4, 3))
shown = (freqs >= 1) & (freqs <= 5)
mean_off = psd_off.mean(axis=1)   # mean over channels
mean_on = psd_on.mean(axis=1)
ax.plot(freqs[shown], mean_off[shown], color=FP.COLOR_NULL, linewidth=FP.LW_TRACE, label="Magnet off")
ax.plot(freqs[shown], mean_on[shown], color=FP.COLOR_MAG, linewidth=FP.LW_TRACE, label="Magnet on (3 Hz)")
ax.axvline(STIM_FREQ, color="black", linestyle=":", linewidth=FP.LW_REFERENCE, zorder=0)
ax.axhline(induced_psd, color="black", linestyle="--", linewidth=FP.LW_REFERENCE,
           label=f"Max. induction ({induced_uv:.0f} µV sine)")
ax.set_yscale("log")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (µV²/Hz)")
ax.legend(fontsize=FP.FS_LEGEND, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

out_path = Path(FP.DIAGNOSTIC_DIR) / "LFP_induction_spectrum.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=FP.DPI)
print(f"Saved {out_path}")

# %% [markdown]
# ## Figure 2: local noise band vs the 3 Hz bin

# %%
fig, ax = plt.subplots(figsize=(4, 3))
# Same range as the earlier sine-amplitude version (0.3-100 uV, i.e. 1.3 to
# 1.4e5 uV^2/Hz). Bins are equal steps of log10(power), and density is per unit
# log10(power), so bin height is comparable across the axis.
bins = np.logspace(np.log10(0.3**2 * SEGMENT_S / 2), np.log10(100**2 * SEGMENT_S / 2), 50)
print(f"Channel-bins below the plotted range: {np.mean(noise_psd < bins[0]):.1%} of noise bins")
log_bins = np.log10(bins)
ax.hist(np.log10(noise_psd), bins=log_bins, density=True, color=FP.COLOR_NULL, alpha=FP.ALPHA_TRACE,
        label="Magnet off, 2-4 Hz bins")
ax.hist(np.log10(off_3hz_psd), bins=log_bins, density=True, histtype="step", color=FP.COLOR_NULL,
        linewidth=FP.LW_TRACE, label="Magnet off, 3 Hz bin")
ax.hist(np.log10(on_3hz_psd), bins=log_bins, density=True, histtype="step", color=FP.COLOR_MAG,
        linewidth=FP.LW_TRACE, label="Magnet on, 3 Hz bin")
ax.axvline(np.log10(induced_psd), color="black", linestyle="--", linewidth=FP.LW_REFERENCE,
           label=f"Max. induction ({induced_uv:.0f} µV sine)")
ticks = np.arange(np.ceil(log_bins[0]), log_bins[-1] + 1e-9)
ax.set_xticks(ticks, [f"$10^{{{int(k)}}}$" for k in ticks])
ax.set_xlabel("Power (µV²/Hz)")
ax.set_ylabel("Density per decade")
ax.legend(fontsize=FP.FS_LEGEND, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

out_path = Path(FP.DIAGNOSTIC_DIR) / "LFP_induction_noise.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=FP.DPI)
print(f"Saved {out_path}")
