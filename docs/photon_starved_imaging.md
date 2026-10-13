# Photon-starved imaging: what went wrong, what was fixed, and what remains

**Date:** 2026-10-12

**Scope:** an index of the work on the photon-starved two-photon recordings. These are the 0.1 / 0.3 Hz zebrafish sessions (all but 20221002_fish1) and the medaka fish. This page sums up the story and links to the reports that hold the evidence.
- The 2022 Q1 zebrafish (0.4 Hz) and 20221002_fish1 are ordinary fluorescence and are not part of it.
- Every number here comes from [`photon_starved/summary.py`](photon_starved/summary.py) or from the linked reports.

## Summary

**What was wrong.** These movies count photons on a camera offset of 100: in most pixels of most frames nothing arrives. Three steps of the standard suite2p pipeline throw most of that signal away:
1. suite2p halves uint16 input, which merges 0 and 1 photon into one level;
2. it truncates its registered frames to integers;
3. its registration reference becomes flat, so every frame gets the same shift.

The result was traces that sat on a floor, ROIs that were fragments of cells, and movies that were not motion corrected.

**What was done, in order:**
1. **Full-resolution extraction** (2026-10-05) removes the halving and truncation from the analysed traces.
2. **Tiffs ×64 for medaka** (2026-10-07) let suite2p register them and segment the six trials together.
3. **For the zebrafish** (2026-10-12), suite2p still can't register even scaled tiffs. Their drift is therefore measured ourselves, corrected in whole pixels, and suite2p segments the ×64 corrected movie with its own registration off.

**What it bought, and what it didn't:**

| | zebrafish (5 sessions) | medaka (1 fish, 6 trials) |
|---|---|---|
| ROI size, median npix of selected ROIs | 15 → 32 px (**+113%**) | 13 → 23 px (**+77%**) |
| selected ROIs (P(iscell) > 0.5, npix ≥ 10) | 2,309 → 3,488 (**+51%**) | 274 per trial (separate runs) → 342 shared by all six |
| suite2p's own trace on its floor (median ROI) | 97% → 28% of frames | 90% → 0% |
| analysed trace on its floor | none truncated since 2026-10-05; 28% of frames are frames with no photon in the ROI | none truncated since 2026-10-05 |
| still nonstationary (block shuffle, K = 10; chance 5%) | RMS 68% → 60%, reference-window power 30% → 30% | RMS 35%, reference-window power 10% (no "before" exists) |
| magnetic result (Fig 2A) | unchanged: 1 of 88 recordings flagged, before and after | unchanged |

"Before" for the zebrafish is the previous production run (suite2p v1, tiffs as acquired). For medaka it is the original suite2p 0.10.1 runs, one per trial; the v1 runs on the as-acquired medaka tiffs no longer exist. "Selected" here leaves out the fish outline and flatline removal, so that every run is treated alike.

**Remaining nonstationarity.** Most of what remains is a slow change in the image itself, which no in-plane correction removes. The leading hypothesis, untested, is that the focal plane (z) drifts (section 4).

## 1. What was wrong

| problem | where it shows | report |
|---|---|---|
| Photon counting on an offset of 100: 96–98% of pixels at 100, one level per photon | raw tiffs | [`cv2.md`](cv2.md) section 4, [`medaka_concat_suite2p.md`](medaka_concat_suite2p.md) (data inventory), [`custom_registration.md`](custom_registration.md) section 4 |
| suite2p halves uint16 (`// 2`, so 100 and 101 both become 50) and truncates registered frames to int16 | traces at one value in 91–98% of frames; CV² well below 1 | [`cv2.md`](cv2.md), [`nfc_finite_sample_bias.md`](nfc_finite_sample_bias.md) ("Slow variation and floor-clipped traces") |
| Floor-clipped traces break `fit_Fourier`'s null | p-value excess at every analysis frequency | [`nfc_finite_sample_bias.md`](nfc_finite_sample_bias.md), [`few_event_spectra.md`](few_event_spectra.md), [`guard_band_null.md`](guard_band_null.md), [`dispersion_matched_null.md`](dispersion_matched_null.md) |
| suite2p's registration reference is flat after halving, so every frame gets the same shift (no motion correction) | `ops` yoff/xoff constant | `.claude/CLAUDE.md` ("Registration (2026-10-07)"), [`custom_registration.md`](custom_registration.md) section 1 |
| Segmentation on the halved movie: most ROIs under 10 px, fragments of cells | 94% of ROIs < 10 px in 20220915_fish1; one cell split three ways | [`custom_registration.md`](custom_registration.md) section 4, [`resegmentation.md`](resegmentation.md) section 1 |
| Separate segmentations per medaka trial: no ROI identity across trials | the neuron key merged unrelated cells | [`medaka_concat_suite2p.md`](medaka_concat_suite2p.md) |
| ROIs outside the fish | dark background | [`body_outline.md`](body_outline.md) |

## 2. What was fixed, step by step

| date | step | for | what it fixes | report |
|---|---|---|---|---|
| 2026-10-05 | Full-resolution extraction: suite2p's own warp in float, no halving or truncation (`pipeline/ophys_extraction.py`, `F_full.npy`) | all imaging | truncation in the analysed traces | [`cv2.md`](cv2.md) section 4, [`full_resolution_rerun.md`](full_resolution_rerun.md), [`coverage_full_resolution.md`](coverage_full_resolution.md) |
| 2026-10-05 | suite2p v1 (1.1.0, GPU) replaces 0.10.1 / 0.14.4 | all imaging | | [`suite2p_v1_comparison.md`](suite2p_v1_comparison.md) |
| 2026-10-07 | Medaka: tiffs ×64, the six trials segmented as one movie | medaka | registration (suite2p's shifts now vary, up to 3 px); cross-trial identity; suite2p's own traces no longer truncated | [`medaka_concat_suite2p.md`](medaka_concat_suite2p.md); PRs [#62](https://github.com/danielpollak/magnet-search/pull/62), [#63](https://github.com/danielpollak/magnet-search/pull/63) |
| 2026-10-12 | Zebrafish drift measured from 10-frame averages | zebrafish | finds the motion suite2p missed: up to 20 px in 20220915_fish1, 2–3 px elsewhere | [`custom_registration.md`](custom_registration.md) sections 1–3 |
| 2026-10-12 | Zebrafish re-segmentation, now in production: tiffs ×64, drift corrected in whole pixels, suite2p's registration off | zebrafish | ROI size and count; suite2p's own traces no longer truncated | [`resegmentation.md`](resegmentation.md) |

**Keeping the steps apart:**
- **Truncation of the analysed traces** was removed by the full-resolution extraction, before any scaling.
- **The ×64 scaling** fixes what suite2p itself does with the movie: its segmentation, its medaka registration, and its own traces. With ×64 tiffs suite2p's halving loses nothing, so its own trace equals the full-resolution one (zebrafish: on its floor in 28% of frames, against 29% for the full-resolution trace).
- **The drift correction** aligns the frames. On its own it adds few ROIs: 3,488 selected with whole-pixel correction against 3,672 without any correction ([`resegmentation.md`](resegmentation.md) section 2). It does make the larger ROIs follow their cells. Without it a 33 px ROI's grating p-value is slightly worse than its fragments' (−3.8 vs −4.0, log₁₀); with it, it is as good or better (Table 2 there).

## 3. What changed, in numbers

**ROIs** ([`resegmentation.md`](resegmentation.md) Tables 1–2; [`photon_starved/results_summary.csv`](photon_starved/results_summary.csv)):
- **Size:** zebrafish selected ROIs grow from a median of 15 to 32 px; medaka from 13 to 23 px.
- **Classifier:** median P(iscell) per session is 0.86–0.90, against 0.81–0.91 before (zebrafish).
- **Same cells, better traces:** a re-segmented zebrafish ROI collects 7.5 photons per frame against 4.2 for the production fragments inside it. It responds to the grating as often (67% vs 66%).
- **Count:** the zebrafish gain varies by session. It is 0.9–1.4× in four sessions and 2.9× in 20220915_fish1, the one with the 20 px drift.
- **Cap:** detection stops at suite2p's limit of 5,000 ROIs in every run, production's included. The limit is kept as a conservative cut ([`resegmentation.md`](resegmentation.md) section 4).

**Floor share** ([`photon_starved/results_summary.csv`](photon_starved/results_summary.csv); median over selected ROIs of the share of frames on which the trace equals its own minimum):

| | suite2p's own trace | full-resolution trace |
|---|---|---|
| zebrafish, before (previous production) | 97% (halved + truncated) | 11%* |
| zebrafish, after (current production) | 28% | 29% |
| medaka, before (0.10.1, per trial) | 90% | n/a (never extracted) |
| medaka, after (current production) | 0% | 0.1%* |

<sub>\* A warped trace rarely lands exactly on its minimum, because the bilinear interpolation mixes in neighbouring pixels. So the warped full-resolution traces (zebrafish before, medaka after) can't be compared with the unwarped zebrafish "after" trace on this measure. The zebrafish "after" 28% is the share of frames in which none of the ROI's pixels caught a photon.</sub>

**Nonstationarity** ([`photon_starved/results_nonstationarity.csv`](photon_starved/results_nonstationarity.csv); share of ROIs with block-shuffle p < 0.05 at K = 10, median over recordings, chance 5%; method in [`nonstationarity.md`](nonstationarity.md)):

| | RMS (overall level) | reference-window power | CV² (spectral shape) |
|---|---|---|---|
| zebrafish, before | 68% (43–98%) | 30% (11–83%) | 4.5% |
| zebrafish, after | 60% (41–95%) | 30% (10–75%) | 4.9% |
| medaka, after | 35% (24–38%) | 10% (8–12%) | 4.1% |

- **Most zebrafish ROIs remain nonstationary.** The level and the power around the analysis frequency drift over the recording, while the spectral shape stays at chance.
- **Neither the larger ROIs nor the drift correction removes it.**
- **It isn't what makes the suspects.** [`nonstationarity.md`](nonstationarity.md) shows the suspects are no more nonstationary than other ROIs.

**Magnetic and positive-control results** ([`photon_starved/results_fig2.csv`](photon_starved/results_fig2.csv), [`resegmentation.md`](resegmentation.md) section 5):
- **Fig 2A:** 1 of 88 magnetic recordings flagged before and after (20221002_fish2 magneto_0).
- **Fig 2B:** 63 of 108 flagged before and after.
- **Neurons:** zebrafish neurons in the magnetic population go from 5,533 to 6,709.
- **0.3 Hz suspects:** at chance (0.7–1.1%) with every segmentation.
- **0.1 Hz suspects:** stay at about 1.4–1.7%, with the coil on and off alike, as before. That is the grating's 6th harmonic, and 20221002_fish2's unexplained excess (`.claude/CLAUDE.md`).

## 4. What remains looks like a change of focal plane, but that is untested

The nonstationarity left after the fixes has to come from somewhere. The evidence, all from 20220915_fish1, the session with the largest drift:

- **After the lateral correction, early frames still don't match later ones.**
  - A 10-frame average's correlation with the session image climbs for about 7 minutes. That is also true in the coil-off trials, where the fish moves under 2.3 px ([`custom_registration.md`](custom_registration.md) section 2, Figures 2–3).
  - In the first minute, structures appear that have no counterpart later.
- **It isn't a rotation, stretch or warp.**
  - Local block shifts left after the translation are 1–7 px and reproducible across interleaved trials (r up to 0.97).
  - No affine map improves the match, and in the first minutes the blocks match poorly anywhere ([`custom_registration.md`](custom_registration.md) section 5, Figure 5).
- **Videos:**
  - [`custom_registration/roi58_magneto_0.mp4`](custom_registration/roi58_magneto_0.mp4) and [`custom_registration/roi58_magneto_0_crop96_all_rois.mp4`](custom_registration/roi58_magneto_0_crop96_all_rois.mp4) show the old registration and the correction side by side over a whole trial: the 20 px drift in the first 3 minutes, and image content that keeps changing after it.
  - [`resegmentation/vid_20220915_fish1_magneto_0_roi33_corrected_int.mp4`](resegmentation/vid_20220915_fish1_magneto_0_roi33_corrected_int.mp4) shows the same field in the current production movie next to the old one.
- **Interpretation.**
  - A shift of the focal plane would replace structures rather than move them. Cross-correlation would then lock onto the nearest similar structure, giving local shifts that are reproducible but uncorrectable in-plane.
  - The coil-on and coil-off trials are separate acquisitions and may also sit at slightly different depths.
- **Not established:** there is no z-stack to test this against, and it has only been examined in one session. Slow brightness changes (bleaching, the fish settling) would also produce nonstationarity, and are not ruled out.

## 5. Anything missing? Open items

1. **The z hypothesis is untested.** A z-stack of these fish would test it. Short of that, the block analysis could be run on the other four sessions to see whether the same non-affine, reproducible pattern appears.
2. **Brightness drift (bleaching) has not been separated from z drift.** The trace mean over time per ROI, against the image-correlation curve, would separate them.
3. **[`nonstationarity.md`](nonstationarity.md) predates the zebrafish re-segmentation.** Its zebrafish numbers describe the previous ROIs. Rerunning `docs/nonstationarity/nonstationarity.py` would update it; the changes are the few points in section 3.
4. **Other docs predate it too.** Several (`cv2.md`, `full_resolution_rerun.md`, `coverage_full_resolution.md`, `suite2p_v1_comparison.md`) describe earlier segmentations. They remain correct about what they tested.
5. **The 0.1 Hz excess** is attributed to the grating's 6th harmonic only for medaka per ROI; for the zebrafish it is not yet checked per ROI (`.claude/CLAUDE.md`).
6. **Motion can't fake a magnetic response.** Motion locked to the stimulus is under 0.04 px in all 50 imaging recordings (`.claude/CLAUDE.md`).

## Reproducing

```bash
NS_DATA_DIR=<checkout>/data python docs/photon_starved/summary.py   # magneto2, ~2 min; reads the NAS (medaka 0.10.1 runs), read-only
```

The pre-switch parquet is the source of the "before" Fig 2 row. It is read from `C:\Users\dan\nwb_backup_pre_reseg_2026-10-12\all_fourier_df.parquet` if that still exists.
