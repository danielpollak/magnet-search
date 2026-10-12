# Nonstationarity of the imaging traces

**Date:** 2026-10-12. **Branch:** `nonstationarity-block-only` (earlier: `nonstationarity-doc`, `nonstationarity-nested`). **Script:** `docs/nonstationarity/nonstationarity.py --workers 24` (~20 min on 24 cores; `--figures-only` replots). **Related:** [`fourier_grid_shift.md`](fourier_grid_shift.md) (the shifted Fourier grid used here for the 2022 Q1 chunks).

**Question.** Do a ROI's statistics change within a recording more than they would in a stationary trace with the same values and the same short-range autocorrelation? fit_Fourier's null assumes they don't.

**Summary.**
- **The shape of the spectrum is stationary.** CV² is flagged in 4.6–4.7% of ROIs, which is chance.
- **Level and amplitude are not.** The RMS about the trace mean is nonstationary in 52–56% of ROIs, and in 16–98% per recording.
- **The power in the NFC reference window is nonstationary in 20–27% of ROIs.** This is the power NFC divides by.
- **The 1F suspects are not more nonstationary than the other ROIs.** Among ROIs tested at the magnetic frequency, the suspects are flagged as often as the others, or slightly less often.

## Method

**Population.**
- All 50 imaging recordings: every `experiments/engert_*.yml` and `medaka_*.yml`, coil on and off, with and without a visual stimulus. 33,339 ROIs in total.
- The ROIs are production's: P(iscell) > 0.5, npix ≥ 10, inside the fish outline, flatline removal; no coverage threshold.
- The traces are the full-resolution ones production analyses, over the N frames fit_Fourier uses.

**Chunks of whole stimulus cycles, nested across resolutions.** Each trace is cut into K = 5, 10 and 20 equal chunks.
- **Each chunk holds a whole number of cycles of the recording's analysis frequency f.**
- **The resolutions nest.** A K = 5 chunk is exactly two K = 10 chunks and four K = 20 chunks, so every coarse boundary is also a fine one, and all three resolutions use the same frames. The K = 20 chunk length L is the longest ≤ N/20 for which L, 2L and 4L all hold whole cycles. Frames after the last chunk are not used.
- **With 1 s frames this is exact:** chunks are multiples of 10 frames, which is one cycle of 0.1 Hz and three cycles of 0.3 Hz.
- **In the 2022 Q1 recordings it can't be exact,** because the frame period is 1.0212 s. A cycle there is 2.448 frames at 0.4 Hz and 58.75 frames at 1/60 Hz, never a whole number.
  - The leftover, at most 0.07 cycles, is removed by shifting the chunk's Fourier grid by that fraction of a bin. This is the same shift production now applies to whole traces (`fit_Fourier(align="shift")`, [`fourier_grid_shift.md`](fourier_grid_shift.md)).
  - After the shift, one coefficient sits exactly on f, and its neighbours are spaced one chunk-cycle apart.

**Table 1.** Chunk length in frames (stimulus cycles) and reference-window coefficients per chunk, and the frames used (the same at every K).

| recordings | N | f | frames used | K = 5 | K = 10 | K = 20 |
|---|---|---|---|---|---|---|
| medaka | 1080 | 0.1 Hz | 1000 | 200 (20), 6 | 100 (10), 4 | 50 (5), 2 |
| 0.1 Hz zebrafish | 1080 | 0.1 Hz | 1000 | 200 (20), 12 | 100 (10), 6 | 50 (5), 4 |
| 0.1 Hz zebrafish (20221002_fish1) | 1260 | 0.1 Hz | 1200 | 240 (24), 14 | 120 (12), 8 | 60 (6), 4 |
| 0.3 Hz zebrafish | 1200 / 1260 | 0.3 Hz | 1200 | 240 (72), 22 | 120 (36), 10 | 60 (18), 6 |
| 2022 Q1, 02_21 and 02_23 | 1194 | 0.4 Hz | 980 | 196 (80), 24 | 98 (40), 12 | 49 (20), 6 |
| 2022 Q1, 03_01 | 2936 | 0.4 Hz | 2840 | 568 (232), 70 | 284 (116), 34 | 142 (58), 18 |
| 2022 Q1 visual, 02_21 and 02_23 | 1194 | 1/60 Hz | 1180 | 236 (4), 4 | 118 (2), 2 | 59 (1), 2 |
| 2022 Q1 visual, 03_01 | 2936 | 1/60 Hz | 2360 | 472 (8), 10 | 236 (4), 4 | 118 (2), 2 |

Requiring nested whole cycles costs frames in two places:
- **the 1194-frame 0.4 Hz recordings:** 980 of 1194 frames are used (82%);
- **the 03_01 grating test:** 2360 of 2936 frames (80%).

Everywhere else at least 92% of the frames are used.

**Three metrics per chunk:**

| metric | per chunk | what it detects |
|---|---|---|
| **CV²** | var/mean² of the chunk's periodogram ordinates from 0.05 Hz to just below Nyquist, stimulus coefficient excluded (the CV² of [`cv2.md`](cv2.md)). Each ordinate is first divided by that ROI's mean ordinate at the same frequency over the K chunks, so the spectrum's slope does not count. | a change in how spiky or event-like the trace is; 1 for Gaussian noise |
| **RMS** | √mean((x − whole-trace mean)²) | a change in baseline level or amplitude |
| **reference-window power** | mean power of the M = round(Q_frac × cycles) coefficients adjacent to the stimulus coefficient on each side, stimulus coefficient excluded. This is production's NFC window at the chunk's resolution. | a change in the power NFC divides by |

- **With whole cycles, a response at f stays in the excluded stimulus coefficient.** On a pure 0.4 Hz sinusoid in 1194 frames at K = 10, the window holds a fraction 10⁻⁵ of the stimulus coefficient's power.
- **This matters wherever there is a real response,** for example the grating responders and medaka's grating-harmonic cells. In the first version, with arbitrary chunk lengths, up to ~40% of a response leaked into the window.
- **At fine resolution the window holds only a few coefficients** (Table 1), so the metric is noisy there. The null is built the same way, so p stays valid; only the power to detect nonstationarity drops.

**Dispersion across chunks.** The dispersion is the coefficient of variation of the K chunk values, std/mean.
- A raw std carries the trace's units, so it can't be compared across ROIs.
- The per-ROI p-value is essentially the same either way, because shuffling doesn't change the mean.

**Null: block shuffle.** The dispersion's null comes from 1,000 block shuffles of the whole trace.
- **How:** contiguous 10 s blocks are permuted. This keeps the values and the autocorrelation within 10 s, and destroys only slower structure.
- **Shared permutations:** within a recording, every ROI gets the same permutation in a given replicate.
- **p-value:** p = (1 + number of replicates with dispersion ≥ observed) / 1001, so the smallest possible p is 0.001.

**Control.** In each recording, 20 ROIs were replaced by synthetic stationary traces, 1,000 traces in all, and run through the same procedure (rows with `synthetic=True` in the results table).
- **What the synthetic traces are:** first-order autoregressive, AR(1), traces. Each frame is the previous frame times ρ plus fresh Gaussian noise: x<sub>t</sub> = ρ·x<sub>t−1</sub> + ε<sub>t</sub>.
  - **They are correlated frame to frame,** not uncorrelated. ρ is the correlation between consecutive frames, and the correlation at a lag of k frames falls off as ρ<sup>k</sup>.
  - **ρ is each ROI's own lag-1 autocorrelation,** and the trace is scaled to the ROI's mean and variance. The median ρ is 0.29; the 10th–90th percentile range is 0.00–0.41 in medaka and 0.16–0.83 in the 0.3 Hz zebrafish.
  - **They are stationary by construction,** because ρ and the noise level never change. They keep the ROI's short-range memory (the calcium decay) without any slow change, which is exactly the case the null must not flag.
- **The block shuffle is calibrated on them:** p < 0.05 in 4.0–8.0% of these traces across the nine metric × K conditions. RMS is slightly high, at 7–8%.
- **Why not permute every frame:** shuffling single frames destroys short-range autocorrelation along with slow structure, so it tests "independent frames" rather than "stationary". On the same stationary AR(1) traces it flags CV² in 11% and RMS in 20–33%, so it over-rejects through autocorrelation alone.
- **What's still computed:** the script still computes the frame shuffle for this check; its columns are `p_full_*` in the results table. It is not used anywhere else in this report.

## In one example ROI, power rises over the recording while the spectral shape stays put

![Procedure](nonstationarity/fig_ns_procedure.png)

<sub>**Figure 1. One ROI from each recording group, with the chunk boundaries.** Tall dark lines: K = 5. Medium: K = 10. Short light: K = 20. The resolutions nest, so a K = 5 line also marks a K = 10 and a K = 20 boundary (where the lines overlap, the darker one shows). The last dark line ends the last chunk; frames after it are not used. Chunks hold whole cycles of the recording's analysis frequency (Table 1). Each ROI is the one at the 95th percentile of its recording's trace std/mean (an active cell, not an outlier). Traces: full resolution.</sub>

![Metric time series](nonstationarity/fig_ns_metrics.png)

<sub>**Figure 2. The three metrics, chunk by chunk, for Figure 1's 0.1 Hz zebrafish ROI** (engert_20221001_fish2_magneto_0, ROI 2198). Each step is one chunk; colour = K. The ROI's activity grows over the recording (Figure 1, third row), and RMS and reference-window power rise with it. CV² wanders with no trend.</sub>

![Null distributions](nonstationarity/fig_ns_null.png)

<sub>**Figure 3. Where the same ROI's dispersions sit on the block-shuffle null.** Rows: metric. Columns: K. Histograms: dispersion (std/mean across chunks) in 1,000 block shuffles, 40 bins; shade = K; legends give p. Black: observed. CV² is typical (block p = 0.35–0.85). RMS is beyond all or nearly all replicates (block p ≤ 0.002). Reference-window power is in the top 1–4% (block p = 0.014–0.037).</sub>

## Power drifts in a quarter to half of all ROIs, most in the 0.3 Hz zebrafish and least in medaka; the spectral shape drifts in none

![Population](nonstationarity/fig_ns_population.png)

<sub>**Figure 4. Share of ROIs whose dispersion has block-shuffle p < 0.05, per recording.** Columns: metric. Within each group, colour = K (dark = 5, medium = 10, light = 20). Filled circles: recordings without a visual stimulus. Open triangles: recordings with the grating. AR(1) control: the synthetic traces pooled per recording group, one point per group and K. Dashed: 0.05, the share expected when every ROI is stationary.</sub>

**Table 2.** Share of ROIs with block-null p < 0.05 (all 33,339 ROIs).

| | K = 5 | K = 10 | K = 20 |
|---|---|---|---|
| CV² | 4.7% | 4.7% | 4.6% |
| RMS | 52% | 54% | 56% |
| reference-window power | 27% | 24% | 20% |

**Recording groups differ** (RMS and reference-window power: share of ROIs flagged per recording, over the three K):

| | RMS | reference-window power |
|---|---|---|
| 0.3 Hz zebrafish | 44–98% | 15–88% |
| 0.1 Hz zebrafish | 36–85% | 8–53% |
| 2022 Q1 | 33–73% | 9–45% |
| medaka | 16–43% | 6–16% |

- **The grating makes no consistent difference.**
- **Coil on vs coil off.** Each magneto recording was compared with its no_magneto twin (same fish, same trial number; 19 pairs, K = 10).
  - Nonstationarity tends to be higher with the coil on: RMS in 14 of 19 pairs (median +9 percentage points, Wilcoxon p = 0.07), reference-window power in 12 of 19 (+7 points, p = 0.04).
  - CV² does not differ (p = 0.78).
  - Coil condition and acquisition order are confounded, though. The two conditions are separate acquisitions. Medaka's no-magnet folder is named `14844_after`, so it was recorded later; the order is unknown for the zebrafish.

## The suspects are not more nonstationary than the other ROIs

**The question.** Are ROIs with a significant production test (p < 0.01 at the recording's analysis frequency, current parquet) flagged more often than the other ROIs of the same kind of test?

The tests split into three kinds that should not be pooled:
- **magnetic frequency, no visual stimulus:** 119 suspects of 11,659 ROIs (the trial-0 recordings and the 2022 Q1 magnet and nostim recordings);
- **magnetic frequency, grating trials:** 223 of 15,994 (trials 1–2 and 2022 Q1 visual+magnet);
- **grating frequency, 1/60 Hz:** 2,781 of 5,686 (the four 2022 Q1 visual-only recordings). These are real grating responders.

**Table 3.** Share flagged at K = 10 (block p < 0.05), suspects vs the other ROIs of the same kind; Fisher's exact test.

| tests | CV² | RMS | reference-window power |
|---|---|---|---|
| magnetic, no visual | 5.0% vs 4.5% (p = 0.66) | 60% vs 54% (p = 0.27) | 21% vs 25% (p = 0.39) |
| magnetic, grating trials | 6.3% vs 4.8% (p = 0.34) | 47% vs 55% (p = 0.03) | 15% vs 22% (p = 0.02) |
| grating 1/60 Hz | 4.7% vs 4.8% (p = 0.95) | 49% vs 52% (p = 0.01) | 24% vs 28% (p = 8e-5) |

- **No kind of test shows suspects that are more nonstationary.** K = 5 and 20 agree (results table).
- **The first version of this report found an enrichment (32% vs 26%).** It came from two things:
  - pooling the grating responders with the magnetic suspects;
  - chunks that weren't whole cycles, so part of each response leaked into the reference window.

## Drifting power does not produce the suspects

- **The spectral shape fit_Fourier's null assumes holds chunk by chunk.** CV² is at chance in every group.
- **What changes over time is how much power the trace has.**
  - Level and amplitude drift: activity epochs, the onset transient visible at the start of several traces in Figure 1, slow drift.
  - With them, the power in the NFC reference window drifts.
- **The drift doesn't pick out the suspects.** The suspects are flagged as nonstationary no more often than the other ROIs, so nonstationarity is not a candidate explanation for the excess suspects.

Outputs (gitignored, next to the script):
- `results_nonstationarity.csv`: one row per ROI, with:
  - each metric × K × null's observed dispersion, p and ratio to the null median;
  - the chunk length, number of cycles and leftover per K;
  - the production p-value at the analysis frequency.
- `results_nonstationarity_examples.npz`: the four example recordings' traces, chunk series and null replicates.
