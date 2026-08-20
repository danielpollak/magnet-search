# Fig1 magnetic-population fitting problem: investigation summary

**Date:** 2026-08-19
**Scope:** Diagnostic investigation only. No production pipeline code or config was
left changed as a result of this work (see "Reverted changes" below).

## The observation

In `pipeline/manuscript/fig1.py`'s panel F (ECDF deviation from the null,
p-value calibration check), the magnetic-stimulus ("null") population for the
exemplar rec (`2023-04-13_17-17-14_W25R_second_site_mag3`, experiment
`20230413_secondsite`, N=204 units) shows a systematic **negative** deviation:
its 95% CI band sits below zero across most of the p-value range, rather than
straddling it. Concretely:

- Empirical NFC mean/median: **1.146 / 1.044**
- Theoretical Rayleigh(1) null mean/median: **1.2533 / 1.1774**
- p-value mean/median: **0.551 / 0.578** (should be ~0.5/0.5 under a
  well-calibrated null)
- KS test of p-values vs. Uniform(0,1): stat=0.101, **p=0.030** (significant)

This reads as "overnormalization" — NFCs shifted toward zero relative to what
a correctly-calibrated null predicts, i.e. the per-unit noise-scale estimate
(`sigma_hat`) is systematically too large for this population.

## Hypothesis 1: Q_frac too small — ruled out

Current config: `mag_Q_frac: 0.15` → M=26 off-frequency bins per side at this
rec's T=58s, freq=3Hz (`eps=0.069`).

**Test:** bumped `mag_Q_frac` to 0.3 (M=26→52, `eps` 0.069→0.049), reran
`pipeline/analysis.py --experiment 20230413_secondsite`, regenerated Fig1.

**Result:** no meaningful change.

| | M=26 (Q_frac=0.15) | M=52 (Q_frac=0.3) |
|---|---|---|
| NFC mean/median | 1.146 / 1.044 | 1.143 / 1.047 |

Panel F's mag trace was visually indistinguishable before/after. **Conclusion:**
the `eps`-correction machinery only reshapes the null's tail spread (a
finite-sample-noise correction); it does essentially nothing to a
population-level *central-tendency* shift like this one. Q_frac tuning is not
the lever for this problem. Change was reverted (YAML restored to 0.15, NWB
restored from backup).

## Is this rec special, or a general problem?

Checked all 74 magnetic recs in the pooled `all_fourier_df` population
(`statistics.get_poscontrols_negresults`'s "neg_res"/magnetic set):

- **Pooled across all 74 recs / 18,144 units:** mean/median p = **0.498 / 0.498**,
  pooled KS-test p = **0.235** (not significant). The overall population *is*
  well-calibrated, consistent with Fig2 looking "quite nice" on average.
- **Per-rec spread is real:** std of per-rec mean-p across 74 recs ≈ 0.027.
  This exemplar rec sits at the **97th percentile** — unusually shifted, but
  not literally unique (one other rec is similarly or more extreme).
- 9/74 recs hit `ks_p<0.05` vs. ~3.7 expected by chance — a mild excess of
  miscalibrated recs, in both directions, not a systemic one-directional bias.

**Conclusion:** this experiment isn't broken; it's an unlucky exemplar pick
sitting in the tail of natural rec-to-rec variability. The underlying
mechanism, though, turned out to be common across recordings (next section).

## Hypothesis 2: non-flat (tilted/curved) noise floor

`sigma_hat² = mean(power over the M off-frequency bins) / 2` (`get_sgm`). This
assumes the noise floor is flat (same variance) across the analyzed window. If
it's not, the window-average is a biased estimate of the true noise level *at*
the stimulus frequency.

### Linear slope: real, but a red herring for the mean shift

Regressed per-bin population-averaged power against **signed** distance from
the stimulus frequency: slope=−0.064, **p=0.0011**, r=−0.44 (more power on the
low-frequency side of the window). Screened this across all 72 scoreable recs
(2 mouse recs excluded — no raw per-unit Fourier coefficients persisted, only
precomputed summary pickles):

- **64/72 (89%)** show a negative slope (same direction).
- **35/72 (48.6%)** are significant at p<0.05 — vs. ~5% (3.6 recs) expected by
  chance. So the tilt itself is a pervasive, real, shared property of these
  recordings' spike-train spectra, not idiosyncratic to this one rec.

**Smoke test — did "subtracting the slope" fix it?** No: applying a
mean-preserving linear detrend left NFC mean/median (1.146/1.044 → 1.143/1.038)
and the KS test (p=0.030 → 0.028) essentially unchanged.

**Why:** `get_sgm` computes the *raw mean* of power over the window, not a
residual variance around a fitted line. For a **linear** trend averaged over a
window that is *exactly symmetric* around the stimulus frequency (confirmed:
±0.448 Hz here), the average of a straight line over a symmetric interval
equals the line's value at the center, by construction — verified
numerically (linear fit's value at dist=0 = 0.15839, identical to the raw
window average). A mean-preserving detrend of a linear trend over a symmetric
window is a mathematical no-op. This was the flaw in the first smoke test.

### Curvature: the actual mechanism, and a partial fix

A quadratic fit to the same population-averaged power vs. signed distance
reveals real, positive curvature (coefficient **+0.0997**, i.e. the spectrum is
convex — a "bowl" shape, not a straight tilt). By Jensen's inequality, a
symmetric-window average of a convex function *overestimates* its value at the
center:

- Raw window average (what `get_sgm` uses): **0.1584**
- Quadratic-fit value at dist=0 (true center estimate): **0.1513**
- Ratio (center/raw): **0.9554** → raw sigma is inflated by ~4.7%

**Smoke test — applying this correction** (scaling every unit's sigma by
√0.9554):

| | NFC mean/median | p mean/median | KS-vs-Uniform |
|---|---|---|---|
| theoretical null | 1.2533 / 1.1774 | 0.500 / 0.500 | — |
| raw (production) | 1.146 / 1.044 | 0.551 / 0.578 | stat=0.101, **p=0.030** |
| **population-pooled local quadratic** | **1.173 / 1.068** | **0.540 / 0.564** | stat=0.084, **p=0.105** |

This is a real, substantive improvement — about halfway back to the
theoretical null, and the KS test is no longer significant.

### Is pooling across units the right call, or should each unit be detrended individually?

Validated three ways that pooling (not per-unit fitting) is correct here:

1. **Not an aggregation artifact.** Normalizing each unit's own bins by its own
   mean power first (removing per-unit scale), then averaging shapes, shows
   curvature that's *stronger*, not weaker (coefficient 0.466 vs. 0.0997 raw
   pooled) — ruling out a Simpson's-paradox-style artifact from averaging
   across units with different firing rates.
2. **Per-unit fits are hopelessly noisy on their own.** Individually fitting a
   quadratic to each unit's own 52 off-frequency bins: mean coefficient
   0.0997 (matches the pooled fit, as it must), but std=0.580 — **SNR≈0.17**,
   only 60% of units even get the right sign. A single unit's 52 points can't
   support a reliable 3-parameter fit.
3. **The pooled estimate is stable.** Bootstrapping over units (resample 204
   units w/ replacement, 2000 draws): coefficient 95% CI **[0.019, 0.182]**,
   entirely positive, 99.4% of draws positive. The shared curvature is a
   robust, reproducible feature of this recording (plausibly shared
   broadband/1-over-f-like noise from the recording environment or
   population-wide firing non-stationarity), not a fluke of these 204 units.

**Conclusion:** estimating the shared shape from the pooled population (high
SNR) and applying it to each unit's own scale is the statistically sound
approach; per-unit-only fitting from the Q_frac window alone is not viable.

## Would using the full spectrum (not just the Q_frac window) help?

The Q_frac window is narrow by design (it's built for the *signal* test, not
noise-shape estimation) — only 50-52 points here. Checked whether pulling in
the much wider available spectrum would give enough SNR for genuinely
*per-unit* (not just pooled) curvature correction.

- Computed Fourier power at 839 frequencies (0.5–15 Hz, excluding the on-freq
  bin) for all 204 units — cheap (`allfouriers` is direct summation over spike
  times; sub-second for this).
- Confirmed a real broadband decay: population-average power ~0.148 at 0.5 Hz
  down to ~0.059 at 14 Hz (~2.5× over the range) — not just a narrow-window
  artifact.
- Per-unit log-log slope SNR: **0.25 (narrow, 50 pts) → 1.13 (wide, 839 pts)**,
  fraction-correct-sign 59% → 94%. A ~4.5× SNR improvement — the core
  intuition (more bandwidth → more signal per unit) is correct.

**But the naive version of this overshoots badly.** Fitting a single
power-law (straight line in log-log space) per unit across the full 0.5–15 Hz
band and extrapolating to 3 Hz:

| | NFC mean/median | p mean/median | KS-vs-Uniform |
|---|---|---|---|
| raw (production) | 1.146 / 1.044 | 0.551 / 0.578 | stat=0.101, p=0.030 |
| population-pooled local quadratic | 1.173 / 1.068 | 0.540 / 0.564 | stat=0.084, p=0.105 |
| **wideband per-unit power-law** | **1.539 / 1.431** | **0.408 / 0.359** | stat=0.155, **p=0.0001 (worse)** |

(Verified this isn't a unit-ordering bug: freshly recomputed `fou0` matches
the persisted per-unit table to floating-point precision, same 204 units,
same order.)

**Why it overshoots:** a single global power-law law across such a wide band
doesn't track the *local* curve near 3 Hz — real spike-train spectra
typically decay faster near very low frequencies and flatten toward a noise
plateau at higher frequencies, so a straight line spanning both regimes
extrapolates to a noise level well below the true value at 3 Hz, inflating
NFC past the true null. More bandwidth helps SNR but only if the fitting
approach stays *local* (e.g. a moderately-widened window, or weighted/LOESS-
style fitting) rather than a single global functional form across the whole
spectrum.

## Bottom line

- **Root cause:** a non-flat, *convex* local noise floor near the stimulus
  frequency — not a Q_frac/finite-sample problem, and pervasive across most
  (not just this) magnetic recordings, though usually too small in magnitude
  to matter at the population level.
- **Best fix found so far:** population-pooled local quadratic curvature
  correction (Q_frac-window-local, not global) — gets NFC/p-value calibration
  about halfway to the theoretical null and renders the KS test non-significant.
- **Not a fix:** increasing Q_frac (no effect), linear detrending (mathematically
  a no-op over a symmetric window), or a naive wideband global power-law
  extrapolation (actively overshoots).
- **Not attempted / open for a future project:** a properly *local* wideband
  model (wider than Q_frac but not the full spectrum, with a fitting method
  that doesn't force a single global functional form across regimes that may
  behave differently) could plausibly combine the higher per-unit SNR of a
  wider band with the locality that made the pooled quadratic work. This would
  require touching the protected `fourier_analysis`/`fit_fourier_sig`
  functions (or an alternative estimator) and re-validating against every
  downstream figure (Fig2/Fig3/Fig4/aggregate.py/verify_outputs.py) — out of
  scope for a same-session fix.

## Reverted changes

`experiments/20230413_secondsite.yml`'s `mag_Q_frac` and
`data/20230413_secondsite.nwb` were both restored to their original
(`mag_Q_frac: 0.15`) state after the Q_frac test. No other production files
were modified during this investigation; all analysis above was done via
standalone diagnostic scripts against already-persisted NWB data.
