# GCaMP imaging: from raw tiffs to traces

This covers the zebrafish (`engert`) and medaka (`medaka`) two-photon recordings. The history behind each step is in [`../photon_starved_imaging.md`](../photon_starved_imaging.md), the index of the imaging reports.

## Where the data live

**On the NAS.** Each imaging session is a directory holding the raw tiffs (uint16) and the original suite2p output (0.10.1 or 0.14.4) in `suite2p/plane0/`. The pipeline no longer reads that output; the NAS is only the source of the raw tiffs. Never write to the NAS.

**Segmentation: suite2p v1 (1.1.0, GPU), on local disk.** Every session is segmented with suite2p 1.1.0 (conda env `suite2p1.0`) by `docs/suite2p_v1/run_suite2p_v1.py`, with settings mapped from the original run's `ops.npy`. It copies the tiffs to `C:/Users/dan/suite2p_v1_production/<run>/`, writes suite2p's output to `<run>/output/suite2p/plane0/`, and refuses any path on the NAS. suite2p concatenates every tiff of a run. Each imaging YAML's `suite2p_path` names its run's `plane0/`.

There are three kinds of run, because suite2p cannot register the photon-starved movies as acquired:

| Sessions | Run directory | How |
|---|---|---|
| 2022 Q1 zebrafish (`2022_02_21`, `2022_02_23`, `2022_03_01`) and `2022_10_02-fish1` | `<session>/` | tiffs as saved, suite2p's own registration |
| the other 0.1 / 0.3 Hz zebrafish (5 sessions) | `<session>_x64_drift/` | drift measured from 10-frame averages, each frame moved back by it in whole pixels, tiffs ×64, suite2p registration off (`--scale 64 --shifts`); see [`../resegmentation.md`](../resegmentation.md) |
| medaka (`fish3_8dpf_*`, 6 trials) | `fish3_8dpf_concat_x64/` | all six trials in one run, tiffs ×64 (`--scale 64 --add-session`); see [`../medaka_concat_suite2p.md`](../medaka_concat_suite2p.md) |

`2022_10_02-fish1`'s tiffs were saved ×256 and are not photon-starved. In the ×64 runs, traces are in ×64 units.

| File in `plane0/` | What the pipeline uses it for |
|---|---|
| `ops.npy` | `file_list` and `frames_per_file` (where each trial sits in the concatenated movie), registration shifts (`yoff`/`xoff` rigid, `yoff1`/`xoff1` nonrigid), mean image |
| `stat.npy` | each ROI's pixels and weights (`ypix`, `xpix`, `lam`, `overlap`) |
| `iscell.npy` | classifier probability, written as `p_iscell` |
| `F.npy` | suite2p's traces: stored as `RoiResponseSeries_suite2p` and used for the exact-match check |
| `F_full.npy`, `fullres_check.json` | full-resolution traces and their exact-match check, written by `pipeline/ophys_extraction.py` (below) |
| `meanImg_full.npy` | mean of the full-resolution registered movie, written to the NWB file for the fish outline |
| `Fneu_full.npy` | full-resolution neuropil traces (`--neuropil`), checked against `Fneu.npy` the same way; not used by production |

## Sessions, trials and protocols

One experiment YAML = one trial = one tiff. All trials of a session share one segmentation, so ROI *k* is the same cell in every trial of that session, and the ROI index is the neuron's `id`. The YAML's `tiff_name` picks the trial's frames out of the run.

| Set | Sessions | Design |
|---|---|---|
| 2022 Q1 zebrafish | `2022_02_21`, `2022_02_23`, `2022_03_01` | one ~20 min tiff per condition: magnet (0.4 Hz), visual (grating, 1/60 Hz), both (`visualmagnet`), none. Bright, not photon-starved. Frame period 1.0212 s |
| 0.3 Hz zebrafish | `2022_09_14-fish2`, `2022_09_15-fish1` | `magneto_0/1/2` and `no_magneto_0/1/2` |
| 0.1 Hz zebrafish | `2022_10_01-fish1/2`, `2022_10_02-fish1/2` | as above; magnet 2 Hz gated 5 s on / 5 s off |
| medaka 0.1 Hz | `fish3_8dpf_*` | as above; magnet 0.1 Hz |

In the 0.1 / 0.3 Hz zebrafish and medaka, `_0`, `_1` and `_2` are three **stimulus conditions** interleaved in 60 s chunks during acquisition, not repeat trials: 0 = no visual stimulus, 1 = grating moving left, 2 = grating moving right. Each trial tiff is the first 18 chunks of its condition stitched together, so it has a seam every 60 frames. `magneto` and `no_magneto` are coil on and coil off. `2022_10_02-fish1`'s `_2`/`_3` tiffs on the NAS are byte-identical copies of `_1` and have no YAML.

**How these recordings are used in the figures** (`magpyneto2.statistics.get_poscontrols_negresults`):
- A recording with a visual stimulus is a visual positive control, analysed at 1/60 Hz (`analysis.visual_f`), and is not part of the magnetic population.
- Since 2026-10-18 a visual recording with the coil on (`visualmagnet`, `magneto_1`, `magneto_2`) is left out of the positive control too.

## Why traces are re-extracted

suite2p (0.10.1, 0.14.4, and still 1.1.0) stores its registered movie as int16:
- it halves uint16 tiffs (`// 2`);
- it truncates the registered frames back to integers (in 1.1.0, the nonrigid warp ends in `.short()`).

In the photon-starved movies (offset 100, ~95% of pixels at the offset in any frame, one level ≈ one photon), those two steps erase most of the signal between events, which broke the frequency-domain null ([`../cv2.md`](../cv2.md)).

**How (`pipeline/ophys_extraction.py`, run once per run directory in the `suite2p1.0` env, on the GPU).** Every ROI's trace is computed from the local tiff copies twice, with suite2p's stored registration and ROI weights:
1. **With suite2p's two lossy steps,** using suite2p's own `transform_data`: this must reproduce `F.npy` to float precision, or extraction fails (the *exact-match check*).
2. **Without them, in float32:** the same warp minus the final truncation. This is the full-resolution trace, `F_full.npy`, stored as `RoiResponseSeries`.

The check only passes on the GPU: on the CPU, torch's bilinear sampling lands one level lower in ~0.4% of pixels. Runs with registration off are handled too. Processing (magneto2 env) then reads `F_full.npy`/`F.npy` per trial and never touches the tiffs. Neuropil is not subtracted.

```bash
# suite2p1.0 env, after the suite2p runs; one or more plane0 directories
C:/Users/dan/anaconda3/envs/suite2p1.0/python.exe pipeline/ophys_extraction.py --plane0 C:/Users/dan/suite2p_v1_production/*/output/suite2p/plane0
```

## At analysis time

`pipeline/analysis_stages/engert.py` and `medaka.py` keep an ROI if all of these hold:
- `p_iscell > iscell_threshold` (0.5);
- `npix > npix_threshold` (9);
- inside the fish outline (`body_outline`, see [`../body_outline.md`](../body_outline.md));
- coverage ≥ `coverage_threshold`, currently 0 (off);
- not a flatline (`magpyneto2.engert_helpers.remove_flatlines`).

The 2022 Q1 recordings are not whole stimulus cycles, so `fit_Fourier` evaluates the transform on a grid shifted onto the stimulus frequency (`align="shift"`, the default; [`../fourier_grid_shift.md`](../fourier_grid_shift.md)).
