# NWB file structure

Every experiment has exactly one `data/{name}.nwb` file, and it is genuinely shared by both pipeline stages: `processing.py` creates it with the raw/processed recording containers, and `analysis.py` reopens that *same* file, adds a `processing["analysis"]` module with the Fourier results, and writes the whole thing back out — there's no separate "analysis file." (Internally this is a read → export-to-temp → `os.replace` cycle, not an in-place append, since HDMF can't resize existing datasets on a second open.)

The read/write logic lives in `pipeline/nwb_io.py`; `build_modulation_frame()` and `read_fourier_results_as_full_fourier_df()` reconstruct the old pickle-era `modulation_df`/`full_fourier_df` shapes on demand from these tables — nothing below is stored as those flat DataFrames anymore.

## Electrophysiology: `Units` + `stimulus_epochs`

**`nwbfile.units`** — one row per Kilosort cluster, for *every* cluster (not just `good` ones; filtering happens at analysis/read time via `kilosort_label`):

| Column | Meaning |
|---|---|
| `spike_times` | Spike times, seconds (standard NWB Units column) |
| `kilosort_label` | `"good"` / `"mua"` / `"noise"` / `""` |
| `channel` | Probe channel (NaN if unknown) |
| `cluster_id` | Original Kilosort cluster id (unique only within `rec` when `rec` is non-empty) |
| `rec` | `""` = shared/concatenated across every recording in the file; a real rec name = this unit's sorting is local to just that one recording (e.g. gutfreund) |

**`nwbfile.intervals["stimulus_epochs"]`** — one row per stimulus epoch/recording block:

| Column | Meaning |
|---|---|
| `rec` | Recording/trial-block name — joins against `Units.rec` |
| `stim_type` | `magnetic` \| `visual_gratings` \| `white_noise` \| `oddball[_long_on/off/both]` \| `visual_bars` |
| `frequency` | Stimulus frequency, Hz |
| `orientation_deg`, `skips` | Grating orientation / leading crossings skipped (NaN/0 if n/a) |
| `local_offset_seconds` | Offset subtracted from `Units.spike_times` to recover the legacy recording-local spike time convention |
| `period_crossings` (ragged) | Schmitt-trigger crossing times, seconds |
| `phase_method` | How period/phase are derived — see below |
| `period_frequency`, `min_spikes_on_full_session`, `truncate_spike_samples`, `period_crossing_inclusive`, `gratings_group` | Paradigm-specific bug-compat flags (see `pipeline/nwb_io.py` docstrings for exact semantics) |
| `window_starts`/`window_stops`/`window_offsets`, `phase_anchors`, `synthetic_period_markers` (ragged) | Only present for `"stitched_*"` epochs — describe how disjoint real windows are re-indexed onto one synthetic timeline |

`phase_method` controls which formula reconstructs `period`/`phase` for that epoch: `"crossings"` (real Schmitt-trigger sample crossings — openephys, spikeglx_direct), `"arithmetic"` (continuous floor/modulo on float seconds — gutfreund, single-window white noise), `"gratings"` (openephys_multistim visual gratings, with a separate wider phase-crossings marker array), `"stitched_floor"`/`"stitched_crossings_unnorm"` (multi-window white noise and oddball, spikes re-indexed onto a synthetic timeline first).

## Imaging (engert/medaka): `PlaneSegmentation` + `RoiResponseSeries`

**`nwbfile.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]`** — every suite2p ROI, unfiltered:

| Column | Meaning |
|---|---|
| `p_iscell` | suite2p classifier probability (`iscell.npy` column 1) |
| `npix` | ROI pixel count |
| `x`, `y` | ROI centroid |
| `pixel_mask` | `(x, y, weight)` triples from suite2p's ROI mask |

**`nwbfile.processing["ophys"]["Fluorescence"]`** — two trace series for all ROIs, each `(n_frames, n_rois)`, `rate` = imaging sampling rate, over this experiment's own tiff:

| Series | Content |
|---|---|
| `RoiResponseSeries` | Full-resolution traces re-extracted from the raw tiff (raw tiff units, detector offset included). **What analysis uses.** Its description records the exact-match check's error. |
| `RoiResponseSeries_suite2p` | suite2p's own `F.npy` over the same frames (halved, truncated movie). Kept for provenance and for the reports that describe it. |

`nwb_io.read_roi_data(nwbfile, series="full" | "suite2p")` reads either one. **`nwbfile.processing["ophys"]["SummaryImages"]["meanImg"]`** holds suite2p's mean image, which the fish outline (`pipeline/body_outline.py`) is drawn on. There is no per-unit epochs table analog here — imaging analysis calls `fit_Fourier()` directly on the frame-indexed traces rather than reconstructing period/phase.

## Analysis results: `nwbfile.processing["analysis"]`

Written by both the ephys and imaging analysis paths into the same three tables (safe to re-run; existing rows are rebuilt, not duplicated):

- **`null_distribution_models`** — one row per distinct `Q` (off-frequency bin count) used anywhere in the file: `Q`, `eps` (the finite-`Q` correction factor), and the corrected null PDF/CDF grids (`support_r`, `pdf_corrected`, `cdf_corrected`).
- **`fourier_group_results`** — one row per `(rec, freq, harmonic)` group analyzed: `rec`, `frequency`, `harmonic` (`"1F"`/`"2F"`), `Q`, `T` (duration, s), `C` (unit/cell count), `ff_alt` (off-frequencies analyzed).
- **`per_unit_fourier_results`** — one row per unit per group: `group_1f_index`/`group_2f_index` (row into `fourier_group_results`, `-1` if no paired 2F group), `unit_id`, `p_value`, `NFC`, `2f_NFC`, `2f_p_value`, `sens`, `sens_2f`, `fou0_real`/`fou0_imag`, `fou_alt_real`/`fou_alt_imag`, `sigma`. Ephys rows also carry `spk_count`; imaging rows carry `n_frames` instead (never both).

`read_fourier_results_as_full_fourier_df()` joins these three tables back into the flat per-unit shape `aggregate.py` consumes (columns `id, p_value, NFC, freq, rec, 2f_NFC, 2f_p_value, sens, sens_2f, Q, Q_2f, spk_count/n_frames`).

## Other bookkeeping

`nwbfile.scratch["sampling_rate"]` (ephys) and `nwbfile.scratch["imaging_dims"]` (imaging) hold values needed to round-trip seconds back to integer sample/pixel indices; `nwbfile.subject` carries `subject_id`/`species` when set. `Units.spike_times` and `RoiResponseSeries.data` are gzip-compressed on write.

