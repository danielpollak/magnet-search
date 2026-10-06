# Re-running the GCaMP sessions with suite2p v1 (1.1.0, GPU)

The question: do the original suite2p outputs (0.10.1 for the 2022-09/10 zebrafish and the
medaka trials, 0.14.4 for the 2022 Q1 zebrafish) differ from what the current GPU suite2p
finds on the same tiffs, in ROI count, P(iscell), ROI size (npix) and activity coverage?

- `run_suite2p_v1.py`: copies one session's tiffs to local disk and runs suite2p v1 with
  settings mapped from the original `ops.npy` (env `suite2p1.0`).
- `compare_v1.py`: compares original and v1 outputs, per session and per recording
  (env `magneto2`).

## Safety

Nothing is written to the NAS. The run script reads the original `suite2p/plane0/ops.npy`
and copies the tiffs, read-only. `data_path`, `save_path0`, `fast_disk` and the log
folder all point to `C:/Users/dan/suite2p_v1_runs/<run-name>/`. The script refuses to run
if any of them starts with `//` or `\\`, contains `datanas`, sits on a network drive, or
lies outside the local root. It also records the size and mtime of every file in the
original `plane0/` before and after the run, and raises an error if any of them changed.

## Environment

`suite2p1.0` (`C:/Users/dan/anaconda3/envs/suite2p1.0`): Python 3.11, suite2p 1.1.0, torch
2.14.1+cu126. torch uses the RTX 3060 (`torch_device="cuda"`). pynwb is not installed,
which only disables suite2p's NWB export (not used).

## Running the remaining sessions

From the repo root:

```bash
S2P=C:/Users/dan/anaconda3/envs/suite2p1.0/python.exe
PY=C:/Users/dan/anaconda3/envs/magneto2/python.exe

$S2P docs/suite2p_v1/run_suite2p_v1.py --list        # the 15 session keys
$S2P docs/suite2p_v1/run_suite2p_v1.py --session 2022_02_21 --dry-run   # print settings + mapping only

$S2P -u docs/suite2p_v1/run_suite2p_v1.py --session 2022_02_21
$PY docs/suite2p_v1/compare_v1.py --session 2022_02_21

# medaka: one directory per trial, quote the space
$S2P -u docs/suite2p_v1/run_suite2p_v1.py --session "Medaka Experiments/fish3_8dpf_gcamp_good/fish3_8dpf_magneto_0"
$PY docs/suite2p_v1/compare_v1.py --session "Medaka Experiments/fish3_8dpf_gcamp_good/fish3_8dpf_magneto_0"
```

The run script stops if `output/suite2p` already exists. Pass `--force` to delete it and
rerun. A tiff that has already been copied with the right size is not copied again.
`--delete-tiffs` removes the local copies after a successful run. Run one session at a
time: one GPU, and the scripts do no locking.

Options of `run_suite2p_v1.py`:

- `--run-name`: local folder name (default: last path component).
- `--local-root`: default `C:/Users/dan/suite2p_v1_runs`.
- `--torch-device cpu`: run without the GPU.
- `--reg-batch-size N`: use this only if the GPU runs out of memory during registration.
  The original runs used 500, and 500 fits in 12 GB at 477x477.

`compare_v1.py` reads the sample period from the session's `experiments/*.yml`. This is
1.0 s for all sessions except 2022_03_01, which is 1.02 s. Rerunning `compare_v1.py` for a
session replaces that session's rows in `results/*.csv` and regenerates
`figs/<run-name>.png` and `figs/overview_counts.png`.

## Outputs

| where | what |
|---|---|
| `C:/Users/dan/suite2p_v1_runs/<run>/tiffs/` | local tiff copies (old `filelist` order) |
| `.../<run>/output/suite2p/plane0/` | v1 output: `F.npy`, `stat.npy`, `iscell.npy`, `ops.npy`, `data.bin`, ... |
| `.../<run>/output/settings_used.json` | db + settings passed to suite2p, the old->v1 mapping table, versions, GPU, per-file frame counts, runtimes, and the NAS-unchanged check |
| `.../<run>/output/suite2p_log/run.log` | suite2p's own log |
| `.../<run>/run.log` | stdout of the run script (when redirected as above) |
| `docs/suite2p_v1/results/sessions.csv` | one row per session x {old, v1}: ROI counts, P(iscell) and npix quantiles, runtime |
| `docs/suite2p_v1/results/recordings.csv` | one row per recording x {old, v1} x {all ROIs, passing ROIs}: coverage quantiles, share >= 0.1, share of clipped traces |
| `docs/suite2p_v1/results/halving.csv` | raw tiff vs v1 `data.bin` vs old `data.bin` (first 100 frames), plus F |
| `docs/suite2p_v1/results/rois_<run>.csv` | per ROI (gitignored) |
| `docs/suite2p_v1/figs/` | per-session figure and the across-session ROI-count overview |

"Passing" means the production thresholds: P(iscell) > 0.5 and npix >= 10. Coverage
comes from `pipeline/roi_coverage.py`'s `activity`: the share of 60 s windows with at least
3 frames above the trace's floor. It is computed on each recording's slice of F, using the
cumulative `frames_per_file` of the respective run.

## Disk and time

The pilot run, 2022_10_01-fish2_cytoGCaMP (6 x 1080 frames, 477 x 477, 2.95 GB of tiffs):

- Copying from the NAS took 26 s.
- suite2p v1 took 52 s end to end on the GPU: registration 16 s, registration metrics 10 s,
  detection 11 s, extraction 9 s.
- The original 0.10.1 run took 276 s, and that figure excludes its registration, which was
  done in an earlier pass.
- The output takes 3.4 GB, mostly `data.bin`.

Budget about 2.2x the tiff size per session. Estimates for the other sessions, scaled by
frames x pixels:

| session | tiffs | estimate |
|---|---|---|
| 2022_02_21, 2022_02_23 | 4 x 1.17 GB (4776 frames, 700x700) | ~1.5 min each + copy |
| 2022_03_01 | 4 x 2.88 GB = 11.5 GB (11744 frames, 700x700) | ~4 min + ~2 min copy |
| other 0.3/0.1 Hz zebrafish | ~3.0-3.4 GB (6480-7560 frames, 477x477) | ~1 min each |
| medaka trials | 0.49 GB each (1080 frames) | ~15 s each (registration metrics are skipped below 1500 frames) |

All 15 sessions together need about 45 GB of tiffs and about 100 GB including outputs. On
2026-10-05 the disk had 516 GB free. To reclaim space afterwards, delete `tiffs/` and
`output/suite2p/plane0/data.bin`. `compare_v1.py` needs neither, except for the halving
check: without the v1 `data.bin` that check reports only the raw tiff and the old `data.bin`.

## Settings mapping: decisions to know about

The full table is under `"mapping"` in each `settings_used.json`. The choices that matter:

- **`fs` stays 10 Hz**, the value in the original ops (the suite2p default), even though
  these movies run at about 1 frame/s. `fs` sets the detection bin size
  (`round(tau*fs)` = 10 frames), the baseline window and deconvolution. Changing it would
  confound the version comparison.
- **Classifier**: the original runs used the processing machine's *user* classifier, which
  is not recorded in ops. On this machine, `~/.suite2p/classifiers/classifier_user.npy` is
  byte-identical to the built-in `classifier.npy` of both 1.1.0 and 0.14.4. The v1 runs
  therefore set `use_builtin_classifier=True` explicitly.
- **`max_iterations` maps to `sparsery_settings.max_ROIs = 250 * max_iterations` (= 5000)**.
  Old sparsery ran `250*max_iterations` iterations. In the pilot, both runs found close to
  this cap (4840 old, 4896 v1), so the "all ROIs" count is set mostly by the cap. Compare
  the counts after the P(iscell)/npix thresholds instead.
- `spatial_taper` is carried over (40; v1's default is 3.45). `nimg_init`,
  `registration.batch_size`, `block_size`, `high_pass` (now `highpass_time`),
  `spatial_hp_detect` (now `highpass_neuropil`), `nbinned` (now `nbins`), `neucoeff` (now
  `neuropil_coefficient`) and the baseline settings are all carried over.
- Old settings with no v1 equivalent: `1Preg`, `spatial_hp_reg`, `pre_smooth`,
  `spatial_hp`, `frames_include`, `aspect`, `pretrained_model`, `spatial_hp_cp`. `1Preg`
  was False in every original run, so the 1P options had no effect there either.
- v1-only settings are left at their defaults, which reproduce the old behaviour:
  `npix_norm_min/max`, `active_percentile`, `extraction.snr_threshold`,
  `circular_neuropil`, `upsample_meanImg`.
- In both old and v1 runs, spatial scale estimation fails on these movies and falls back
  to 6 px (`spatscale_pix` 6 in old ops; "FORCED spatial scale ~6 pixels" in the v1 log).

## uint16 halving still happens in v1

suite2p 1.1.0 still does `im // 2` on uint16 tiffs (`io/tiff.py`) and stores int16. This
was checked empirically on the pilot's first 100 frames (`results/halving.csv`):

| | mean | mean / raw | most common value | share at that value | distinct values |
|---|---|---|---|---|---|
| raw tiff (uint16) | 100.07 | 1 | 100 | 95.4% | 28 |
| v1 data.bin (int16) | 50.00 | 0.4996 | 50 | 99.4% | 8 |
| old data.bin | 50.00 | 0.4997 | 50 | 99.9% | 8 |

The raw tiffs already sit at an offset floor of 100. Halving merges 100 and 101 into 50,
and 101 holds 3% of all pixels. That removes most of the above-floor pixels: 4.6% of raw
pixels are above the floor, against 0.12% after halving and registration. v1 does exactly what 0.10.1 did, so
the F floor at 50.0 remains.
