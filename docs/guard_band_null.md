# A guard band around the analysis frequency (Q_ignore): does it fix the imaging null?

**Date:** 2026-10-03, split from the dispersion-matched null on 2026-10-05. **Branch:**
`guard-band-null`. **Script:** `docs/guard_band_null/guard_band.py` (~10 min; `--figures-only`
replots). Zebrafish only: the 0.3 Hz and 0.1 Hz sets of recordings, where the null is
miscalibrated, and the 2022 Q1 0.4 Hz set, where it is not, as a control. Background:
[`nfc_finite_sample_bias.md`](nfc_finite_sample_bias.md), sections "Slow variation and
floor-clipped traces" and "Excluding ROIs with long dead time". A separate approach, a null
matched to each ROI's spread, is in [`dispersion_matched_null.md`](dispersion_matched_null.md).

**The idea.** NFC compares the power at the analysis bin with the mean power of the M bins on
each side. In the floor-clipped recordings neighbouring bins are correlated, so the noise bins
right next to the analysis bin are not independent of it. A guard band skips G bins on each
side before the noise bins start, to cut that link. If correlation were what miscalibrates the
p-values, a guard band as wide as the correlation should restore calibration.

## Summary

- **Neighbouring bins are correlated out to about 8–10 bins** in the floor-clipped recordings
  (Figure 1A), so G ≈ 10 should be enough to cut the analysis bin loose.
- **The guard band reduces the excess but does not remove it.** At the test frequencies the
  0.3 Hz recordings go from +0.048 to +0.035 at G = 10 and +0.024 at G = 20; the 0.1 Hz
  recordings from +0.035 to +0.030 and +0.026 (Figure 2). Most of the drop comes past G = 10,
  where the correlation has already died out.
- **It biases calibrated recordings downward.** 2022 Q1 drifts from −0.013 to −0.025 at
  G = 20, as do the stationary simulations: once the noise bins move away from the analysis
  bin, the spectrum's curvature biases the noise estimate (Figure 2).
- **Correcting for correlated noise bins (effective M) changes almost nothing** (Figure 2,
  gray dotted).
- **Power** rises at G = 10–20 (Figure 4), partly because the downward drift pushes p-values
  down everywhere.
- **Recommendation: do not adopt Q_ignore.**

## Terms

| term | meaning |
|---|---|
| ordinate | one value of a trace's periodogram: the power \|Y_k\|² at frequency bin k |
| analysis bin | the bin NFC is computed at (the stimulus bin, or a test frequency) |
| noise bins, M | the bins whose mean power is NFC's noise estimate: M on each side of the analysis bin. Production uses the M bins right next to it |
| guard band G (Q_ignore) | the number of bins skipped on each side before the noise bins start. G = 0 is production. Noise bins: f0 − G − M … f0 − G − 1 and f0 + G + 1 … f0 + G + M |
| test frequencies | analysis bins where nothing was presented (every 3rd bin from 0.05 Hz up, skipping any whose widest window, G = 20, would reach the stimulus). Calibration is judged there |
| dev@0.5 | ECDF(0.5) − 0.5: the share of p-values below 0.5, minus the 0.5 expected. 0 = calibrated. Here computed per test frequency and averaged over test frequencies |

The population is today's production population before the coverage threshold (P(iscell) >
0.5, npix ≥ 10, inside the fish outline, flatline removal), because the point is to keep as
many neurons as possible: 4,295 ROI traces (0.1 Hz, 10 recordings), 1,138 (0.3 Hz, 6) and
5,887 (2022 Q1, 6). Every condition uses the same test frequencies. Each ROI contributes one
p-value per test frequency (98–266 test frequencies per recording).

## Two nulls

Both are exact F distributions; at integer M the first agrees with production's eps-corrected
Rayleigh null to within 0.005 in p.

| null | assumes | NFC²/2 ~ |
|---|---|---|
| nominal (production) | 2M independent exponential ordinates | F(2, 4M) |
| effective M | 2M exponential ordinates, correlated as measured in this ROI's own spectrum, worth 2M_eff independent ones: M_eff = M / VIF, VIF = 1 + 2 Σ_L (1 − L/M) ρ_L | F(2, 4M_eff) |

Simulated traces, where the null is true by construction, are the floor-clipped GCaMP model of
Figure 18 in the background report (Poisson events, 2 s decay, noise, a floor), with the event
rate modulated slowly (τ = 100 s) at log-rate SD 0 (stationary), 2 or 3; 4,000 traces per
condition, in the 0.3 Hz and 0.1 Hz recordings' frame counts and Q_frac.

## 1. How wide is the correlation a guard band would have to skip?

**The question.** A guard band only helps if the analysis bin and its noise bins are
correlated, and only out to a certain distance. How far apart must two bins be before they are
independent?

![Correlation](guard_band_null/fig_gb_diagnose.png)

<sub>**Figure 1. Correlation between bins.** *A:* correlation between a ROI's ordinates L bins apart (each divided by the running median of its 51 neighbours, so a sloping spectrum does not count), mean over ROIs, measured over its whole spectrum away from the stimulus. *B:* distribution over ROI traces of M_eff / M, the share of the noise bins that count as independent once that correlation is accounted for.</sub>

- **Neighbouring bins are correlated out to about 8–10 bins** in the floor-clipped recordings
  (Figure 1A): lag-1 correlation 0.37 (0.3 Hz) and 0.29 (0.1 Hz), against 0.06 in 2022 Q1.
  That sets the guard band: G ≈ 10 should cut the analysis bin loose from its noise bins.
- **The correlation reduces the effective number of noise bins only modestly** (Figure 1B):
  median M_eff is 36 of M = 57 (0.3 Hz) and 23 of 32 (0.1 Hz). In 2022 Q1 the median ROI is
  unaffected.

## 2. Does the guard band restore calibration?

**The question.** If correlation between the analysis bin and its noise bins were the cause,
p-values at the test frequencies should become uniform once G is past the correlation width.

![Calibration vs guard band](guard_band_null/fig_gb_calibration.png)

<sub>**Figure 2. Calibration where nothing was presented, vs guard band.** dev@0.5 averaged over test frequencies, for G = 0–20 bins, under the nominal null (orange) and the effective-M null (gray dotted). *Top row:* real recordings. *Middle and bottom rows:* simulated floor-clipped traces in the 0.3 Hz and 0.1 Hz configurations, stationary (left) and slowly modulated (log-rate SD 2, 3). G = 0 under the nominal null is production. For reference, white noise gives about ±0.003.</sub>

| dev@0.5 at test frequencies, nominal null | G = 0 (production) | G = 10 | G = 20 |
|---|---|---|---|
| 0.3 Hz zebrafish | +0.048 | +0.035 | +0.024 |
| 0.1 Hz zebrafish | +0.035 | +0.030 | +0.026 |
| 2022 Q1 zebrafish | −0.013 | −0.018 | −0.025 |
| simulated 0.3 Hz config, log-rate SD 2 | +0.009 | −0.005 | −0.019 |
| simulated 0.1 Hz config, log-rate SD 2 | +0.026 | +0.016 | +0.010 |

- **The excess shrinks with G but does not go away** (Figure 2, top row). Most of the drop
  happens beyond G = 10, past the correlation width of Figure 1A, so it is not the correlation
  being cut.
- **The guard band costs calibration where there was none to fix.** 2022 Q1 drifts further
  below zero as G grows, and the stationary simulations drift down too. In the simulated
  0.3 Hz configuration the excess is not reduced but overshot, to −0.019 at G = 20. Moving
  the noise bins away from the analysis bin lets the spectrum's curvature bias the noise
  estimate. Part of the drop in the floor-clipped recordings is probably this same downward
  drift, not a fix.
- **The effective-M correction does nothing useful** (gray dotted, on top of orange
  everywhere): with 32–57 noise bins per side, even M_eff ≈ 0.65 M leaves F(2, 4M) almost
  unchanged.

![p-value distributions](guard_band_null/fig_gb_curves.png)

<sub>**Figure 3. The whole p-value distribution where nothing was presented.** ECDF(p) − p, pooled over all test frequencies and ROIs (flat at 0 = calibrated), nominal null. Gray: production (G = 0). Blue: G = 10. Navy dashed: G = 20.</sub>

- **The shape stays the same** (Figure 3): the hump in the middle of the p range, with too
  few small p-values and too few near 1, only gets smaller with G. In 2022 Q1 the curve moves
  further below zero.

## 3. What does it cost in power?

**The question.** How often does each guard band detect a stimulus-locked response that is
known to be there?

![Power](guard_band_null/fig_gb_power.png)

<sub>**Figure 4. Detection of a simulated stimulus-locked response.** Simulated floor-clipped traces (log-rate SD 2) whose event rate is multiplied by 1 + m sin(2πft) at the configuration's stimulus frequency. y: share of traces with p < 0.05, nominal null. At m = 0 there is no response and the share should be 0.05 (dotted). Lines as in Figure 3.</sub>

| share with p < 0.05 | m = 0 (no response) | m = 0.3 |
|---|---|---|
| 0.1 Hz config: production (G = 0) | 0.028 | 0.156 |
| 0.1 Hz config: G = 10 | 0.044 | 0.202 |
| 0.1 Hz config: G = 20 | 0.043 | 0.201 |
| 0.3 Hz config: production (G = 0) | 0.038 | 0.209 |
| 0.3 Hz config: G = 10 | 0.046 | 0.235 |
| 0.3 Hz config: G = 20 | 0.050 | 0.248 |

- **Production is conservative in the tail.** With no response it calls only 2.8–3.8% of
  traces significant at 0.05, because few-event traces rarely produce small p-values.
- **A guard band detects more** (+12–30% at m = 0.3) and brings the false-positive rate up
  toward 0.05. Given the downward drift in Figure 2, this gain cannot be separated from a
  general shift of p-values downward, which would also raise false positives in recordings
  that were calibrated to begin with.

## 4. At the stimulus frequency

**The question (read-out only).** What happens to the real recordings' p-values at the
stimulus frequency with a guard band?

![Stimulus frequency](guard_band_null/fig_gb_stimulus.png)

<sub>**Figure 5. p-value distributions at the stimulus frequency.** ECDF(p) − p for every ROI trace, nominal null, G = 0 (production), 10 and 20. Legends give dev@0.5. Gray band: 95% binomial band.</sub>

| dev@0.5 at the stimulus frequency | G = 0 | G = 10 | G = 20 |
|---|---|---|---|
| 0.3 Hz zebrafish | +0.063 | +0.062 | +0.049 |
| 0.1 Hz zebrafish | +0.044 | +0.032 | +0.025 |
| 2022 Q1 zebrafish | +0.024 | +0.025 | +0.024 |

- **The floor-clipped recordings keep most of their excess** at the stimulus frequency, as at
  the test frequencies. Nothing stimulus-specific appears.
- **2022 Q1's +0.024** is the known local stimulus-locked component in `20220301` (background
  report, Figure 12) and does not change with G.

## What this establishes

| | finding |
|---|---|
| **Established** | Neighbouring bins in the floor-clipped recordings are correlated out to about 8–10 bins. |
| **Established** | A guard band of up to 20 bins reduces the excess by a quarter to a half but does not calibrate the floor-clipped recordings. Most of the reduction comes past the correlation width, and it biases calibrated recordings downward (spectral curvature). |
| **Established** | Correlation between noise bins (effective M) is not the problem: correcting for it changes p-values negligibly. |

**Recommendation.** Do not adopt Q_ignore. If a guard band is ever used, the visual-frequency
fits should keep Q_ignore = 0, as requested. Correlation between bins is a symptom, not the
cause; what does predict a ROI's miscalibration is the spread of its periodogram (CV²), which
[`dispersion_matched_null.md`](dispersion_matched_null.md) and
[`few_event_spectra.md`](few_event_spectra.md) follow up.

## Reproducing

```bash
python docs/guard_band_null/guard_band.py                  # ~10 min, writes results_gb_*.csv and fig_gb_*.png
python docs/guard_band_null/guard_band.py --figures-only   # replot from the CSVs
```
