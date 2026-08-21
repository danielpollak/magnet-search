# MagnetSearch Pipeline

A YAML-parameterized analysis pipeline for multispecies neural recording data, designed to identify neural responses to magnetic stimuli across brain regions and species.

## Exploring Manuscript Figures as Notebooks

The manuscript figure scripts (`pipeline/manuscript/fig*.py`) can be converted to Jupyter notebooks using [jupytext](https://jupytext.readthedocs.io/), which lets you run them cell-by-cell and inspect intermediate outputs.

**Install jupytext:**
```bash
pip install jupytext
# or, if using conda:
conda install -c conda-forge jupytext
```

**Convert a figure script to a notebook:**
```bash
# Single file
jupytext --to notebook pipeline/manuscript/fig1.py

# All figure scripts at once
jupytext --to notebook pipeline/manuscript/fig*.py
```

Each converted `.ipynb` file is placed alongside the source `.py` file. Open it in JupyterLab or VS Code and run cells normally. The notebooks are not committed to the repository; re-run the command if you need to regenerate them after the source scripts change.

---

## Overview

This repository contains the complete analysis pipeline for the MagnetSearch collaboration, a distributed neurophysiology project searching for magnetic field responses in neural populations. The pipeline processes raw electrophysiology and 2-photon imaging data, performs spectral analysis, and generates publication-quality figures.

**Key features:**
- Unified pipeline across multiple recording modalities (Neuropixel, OpenEphys, suite2p)
- Support for 7+ species: mouse, zebra finch, pigeon, quail, owl, zebrafish, medaka
- YAML-based experiment configuration for reproducibility
- A single [NWB](https://www.nwb.org/) file per experiment holding both the processed recording and the appended analysis results — no per-stage pickle files
- Automated quality control and false discovery rate correction
- Publication figures with sensitivity analysis

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/magnet-search-pipeline.git
cd magnet-search-pipeline
```

2. Install dependencies (requires `magneto2` conda environment):
```bash
conda activate magneto2
pip install -e .
```

3. Verify installation:
```bash
python pipeline/processing.py --help
```

### Regenerating Figures (for Manuscript Readers)

If you just want to regenerate the manuscript figures, you don't need the raw data or the processing/analysis stages — just the per-experiment NWB files.

#### Prerequisites

Download the experiment NWB files (see [Data Access](#data-access) below) and place them in `data/`:
```bash
# data/{name}.nwb for every experiment (one file per experiment; each already
# contains both the processed recording and its analysis results)
# data/precomputed/mouse_analysis.pickle and owl_analysis.pickle (these two
# species can't be re-processed from raw data -- see Data Access)
```

#### Generate Figures

```bash
# From the repo root directory (magneto2 conda env):

# Aggregate every data/{name}.nwb + data/precomputed/*.pickle into a single parquet
python pipeline/aggregate.py

# Generate manuscript figures
python pipeline/manuscript/fig1.py    # Exemplar NPIX unit + GCaMP cell
python pipeline/manuscript/fig2.py    # Excess-count barplots + NFC distributions
python pipeline/manuscript/fig3.py    # P-value and q-value uniformity
python pipeline/manuscript/fig4.py    # Modulation-detectability simulation (pigeon-HP pseudopopulation)
```

Output figures: `figs/paper/Fig1.pdf`, `Fig2.pdf`, `Fig3.pdf`, `Fig4.pdf`

## Full Pipeline (For Developers)

### Pipeline Order

```
raw data → processing → analysis → aggregate → manuscript figures
```

Every experiment's outputs live in a single `data/{name}.nwb` file — there are no per-experiment pickles. Processing and analysis are two separate CLI runs that both write into that same file (see [NWB File Structure](#nwb-file-structure) below for exactly what each stage adds).

### Stage 1: Processing

Read raw recording data and write the `Units`/`stimulus_epochs` tables (electrophysiology) or `PlaneSegmentation`/`RoiResponseSeries` (2-photon imaging) into `data/{name}.nwb`:

```bash
# Single experiment
python pipeline/processing.py --experiment 20230413_firstsite

# All experiments (parallelized)
python pipeline/processing.py --all --workers 4

# Subset by species
python pipeline/processing.py --all --filter engert    # zebrafish only
python pipeline/processing.py --all --filter medaka    # medaka only
python pipeline/processing.py --all --filter 2023      # pigeon only (2023 sessions)
python pipeline/processing.py --all --filter Q         # quail only
```

**Output:** `data/{name}.nwb` (created)

`engert`/`medaka` (suite2p imaging) read the raw `F.npy`/`stat.npy`/`iscell.npy`/`ops.npy` files exactly once here and write *all* ROIs, unfiltered — `iscell_threshold`/`npix_threshold` filtering happens at analysis time instead, so tuning those thresholds only requires re-running analysis, not reprocessing. `mouse` and `owl` have no processing stage at all (see [Data Access](#data-access)); `manual` paradigms are skipped with a warning.

### Stage 2: Analysis

Read `data/{name}.nwb` back, run the Fourier significance test (`fit_fourier_sig()` for ephys, `fit_Fourier()` for imaging), and append the results into the *same* NWB file:

```bash
# Single experiment
python pipeline/analysis.py --experiment 20230413_firstsite

# All experiments (parallelized)
python pipeline/analysis.py --all --workers 4

# Subset by species (same filters as processing)
python pipeline/analysis.py --all --filter engert --workers 8
```

**Output:** `data/{name}.nwb` (appended — adds a `processing["analysis"]` module) + diagnostic PDFs in `figs/analysis/`

### Stage 3: Aggregate

Combine every `data/{name}.nwb` file's analysis tables with the precomputed mouse/owl DataFrames into a single parquet:

```bash
python pipeline/aggregate.py
```

**Output:** `data/manuscript/all_fourier_df.parquet`

### Stage 4: Manuscript Figures

Generate publication figures (run `aggregate.py` first):

```bash
python pipeline/manuscript/fig1.py
python pipeline/manuscript/fig2.py
python pipeline/manuscript/fig3.py
python pipeline/manuscript/fig4.py --recompute  # --recompute to force re-simulation; --workers N to parallelize the sweep
```

**Output:** `figs/paper/Fig*.pdf`

## NWB File Structure

Every experiment has exactly one `data/{name}.nwb` file, and it is genuinely shared by both pipeline stages: `processing.py` creates it with the raw/processed recording containers, and `analysis.py` reopens that *same* file, adds a `processing["analysis"]` module with the Fourier results, and writes the whole thing back out — there's no separate "analysis file." (Internally this is a read → export-to-temp → `os.replace` cycle, not an in-place append, since HDMF can't resize existing datasets on a second open.)

The read/write logic lives in `pipeline/nwb_io.py`; `build_modulation_frame()` and `read_fourier_results_as_full_fourier_df()` reconstruct the old pickle-era `modulation_df`/`full_fourier_df` shapes on demand from these tables — nothing below is stored as those flat DataFrames anymore.

### Electrophysiology: `Units` + `stimulus_epochs`

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

### Imaging (engert/medaka): `PlaneSegmentation` + `RoiResponseSeries`

**`nwbfile.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]`** — every suite2p ROI, unfiltered:

| Column | Meaning |
|---|---|
| `p_iscell` | suite2p classifier probability (`iscell.npy` column 1) |
| `npix` | ROI pixel count |
| `x`, `y` | ROI centroid |
| `pixel_mask` | `(x, y, weight)` triples from suite2p's ROI mask |

**`nwbfile.processing["ophys"]["Fluorescence"]["RoiResponseSeries"]`** — suite2p's `F.npy` fluorescence traces for all ROIs, `(n_frames, n_rois)`, `rate` = imaging sampling rate. There is no per-unit epochs table analog here — imaging analysis calls `fit_Fourier()` directly on the frame-indexed traces rather than reconstructing period/phase.

### Analysis results: `nwbfile.processing["analysis"]`

Written by both the ephys and imaging analysis paths into the same three tables (safe to re-run; existing rows are rebuilt, not duplicated):

- **`null_distribution_models`** — one row per distinct `Q` (off-frequency bin count) used anywhere in the file: `Q`, `eps` (the finite-`Q` correction factor), and the corrected null PDF/CDF grids (`support_r`, `pdf_corrected`, `cdf_corrected`).
- **`fourier_group_results`** — one row per `(rec, freq, harmonic)` group analyzed: `rec`, `frequency`, `harmonic` (`"1F"`/`"2F"`), `Q`, `T` (duration, s), `C` (unit/cell count), `ff_alt` (off-frequencies analyzed).
- **`per_unit_fourier_results`** — one row per unit per group: `group_1f_index`/`group_2f_index` (row into `fourier_group_results`, `-1` if no paired 2F group), `unit_id`, `p_value`, `NFC`, `2f_NFC`, `2f_p_value`, `sens`, `sens_2f`, `fou0_real`/`fou0_imag`, `fou_alt_real`/`fou_alt_imag`, `sigma`. Ephys rows also carry `spk_count`; imaging rows carry `n_frames` instead (never both).

`read_fourier_results_as_full_fourier_df()` joins these three tables back into the flat per-unit shape `aggregate.py` consumes (columns `id, p_value, NFC, freq, rec, 2f_NFC, 2f_p_value, sens, sens_2f, Q, Q_2f, spk_count/n_frames`).

### Other bookkeeping

`nwbfile.scratch["sampling_rate"]` (ephys) and `nwbfile.scratch["imaging_dims"]` (imaging) hold values needed to round-trip seconds back to integer sample/pixel indices; `nwbfile.subject` carries `subject_id`/`species` when set. `Units.spike_times` and `RoiResponseSeries.data` are gzip-compressed on write.

## Data Access

### Experiment NWB files

`data/*.nwb` is not committed to git — these files are fully regenerable from raw data via `pipeline/processing.py` + `pipeline/analysis.py`, and HDF5/NWB binaries don't delta-compress well between versions, so tracking every re-run would permanently bloat the repository. They (and `data/manuscript/all_fourier_df.parquet`, which is derived entirely from them) are instead distributed via a GitHub Release — see the repository's Releases page, or [INSERT CaltechDATA DOI] once published.

Each `{name}.nwb` file is self-contained: it holds both the processed recording (`Units`/`stimulus_epochs` or `PlaneSegmentation`/`RoiResponseSeries`) and, once analysis has run, the appended Fourier results (`processing["analysis"]`) — see [NWB File Structure](#nwb-file-structure) above.

### Precomputed pickles (mouse and owl only)

Two pickle files remain in active use, for the two species that can't be re-processed from raw data through this pipeline:
- `data/precomputed/mouse_analysis.pickle` — KyuHyunLee mouse recordings (Intan RHD `.mat` format)
- `data/precomputed/owl_analysis.pickle` — Gutfreund barn owl recordings (pre-computed Bayesian coefficients)

`pipeline/aggregate.py` loads these directly alongside the `.nwb`-derived DataFrames. (A legacy fallback also lets `aggregate.py` pick up a stale `{name}_analysis.pickle` left over from before the NWB migration if an experiment has no `.nwb` at all yet, but no current paradigm writes that file format.)

### Raw Data

Raw neural recordings are hosted on [institutional archive/server]:
- OpenEphys binary recordings (Neuropixel, OpenEphys)
- suite2p outputs (zebrafish, medaka)
- Gutfreund recordings (barn owl, quail)
- SpikeGLX recordings (Q-series birds)

Contact the MagnetSearch collaboration for access.

## Configuration

Experiments are defined in YAML files under `experiments/`:

```yaml
# experiments/20230413_firstsite.yml
name: "20230413_firstsite"
paradigm: "openephys_multistim"
good: true
stream_id: "2"
recording_ldr_cntlbarcodes: false
analysis:
  Q_frac: 0.15
  mag_rec_substring: "Mag"
  visual_rec_substring: "visual"
  # ... stimulus-specific parameters
```

**Key YAML fields:**
- `paradigm` — recording type (openephys, engert, medaka, etc.)
- `analysis.Q_frac` — Fourier window half-width, as a fraction of `analysis.f` (half-width_Hz = Q_frac * f)
- `analysis.f` — stimulus frequency in Hz
- `iscell_threshold` / `npix_threshold` — suite2p cell filtering (engert/medaka)

See [.claude/CLAUDE.md](.claude/CLAUDE.md) for detailed field semantics.

## Verification

Compare current NWB outputs against the frozen original pickle-era fixtures in `MagnetSearch/data/` (permanently archived):

```bash
python verify_outputs.py                        # all experiments
python verify_outputs.py 20230413_firstsite     # one (or more) experiments, by name
```

Expected results:
- **PASS** — DataFrames match exactly (column order and dtypes ignored)
- **DRIFT_TOLERATED** — matches within a small float tolerance (HDF5 round-trip, not a logic bug)
- **STALE_OLD** — Analysis differs from the old fixture, but re-running `fit_fourier_sig()` on the *old* modulation_df reproduces the current NWB result (acceptable — the algorithm evolved since the fixture was generated)
- **STALE_SEMANTICS** — Same reconciliation as STALE_OLD, but because `Q` was redefined (raw off-frequency bin count → fraction of the analyzed frequency); expect this on essentially every NPIX analysis row, since every old fixture predates that redefinition
- **FAIL** — Genuine mismatch (investigate)

`engert`/`medaka` have no old fixture at all (there was no processing stage before the NWB migration); for those, verification instead re-runs the analysis computation independently and diffs it against what's actually persisted in the NWB file, as a serialization-fidelity check.

## Project Structure

```
.
├── README.md                          # This file
├── pipeline/
│   ├── processing.py                  # Processing stage CLI
│   ├── analysis.py                    # Analysis stage CLI
│   ├── aggregate.py                   # Combine {name}.nwb + precomputed/ → parquet
│   ├── nwb_io.py                      # NWB read/write library (see NWB File Structure)
│   ├── schema.py                      # YAML config dataclasses
│   ├── dispatch.py                    # Paradigm routing
│   ├── paradigms/                     # Recording modality handlers
│   ├── analysis_stages/               # Per-paradigm analysis
│   ├── diagnostics/                   # Diagnostic figure generation
│   └── manuscript/
│       ├── fig1.py                    # Exemplar NPIX + GCaMP
│       ├── fig2.py                    # Excess-count barplots + NFC distributions
│       ├── fig3.py                    # P/q-value uniformity
│       └── fig4.py                    # Modulation-detectability simulation
├── experiments/                       # YAML experiment configs
├── data/
│   ├── *.nwb                          # (generated/downloaded) Per-experiment processed data + analysis results
│   ├── precomputed/                   # (download) Mouse/owl pickles — not re-processable
│   └── manuscript/
│       ├── all_fourier_df.parquet     # (generated) Aggregated data
│       └── *.pkl                      # (generated) Cached figure-simulation intermediates (e.g. Fig4's sweeps)
├── figs/
│   ├── analysis/                      # (generated) Diagnostic PDFs
│   ├── processing/                    # (generated) Processing diagnostics
│   └── paper/                         # (generated) Manuscript figures
└── .claude/
    └── CLAUDE.md                      # Detailed technical docs
```

## Key Concepts

### Normalized Fourier Coefficient (NFC)

The primary test statistic for detecting neural modulation:

$$\hat{c} = \frac{|c_s|}{\hat{\sigma}}$$

where $c_s$ is the Fourier coefficient at the stimulus frequency and $\hat{\sigma}$ is the RMS of coefficients at nearby off-frequencies.

Under the null hypothesis (no modulation), $\hat{c}$ follows the Rayleigh distribution:
$$P_0(\hat{c}) = \hat{c} \exp(-\frac{1}{2}\hat{c}^2)$$

In practice this null is evaluated with a finite-sample correction factor `eps` (derived from `Q`, the number of off-frequency bins used to estimate $\hat{\sigma}$) rather than the plain Rayleigh formula above — every p-value/q-value/threshold computation in the pipeline uses this eps-corrected null distribution. See `magpyneto2.statistics.get_epsilon`/`normalized_Fourier_PDF_corrected`.

### P-Values and Q-Values

- **p-value**: Probability of observing $\hat{c}$ ≥ the measured value under the (eps-corrected) null hypothesis
- **q-value**: False discovery rate (Storey's method) — minimum FDR threshold at which this measurement is significant

### Suspects and Excess Suspects

- **Suspect**: neuron with p-value < 0.01 (NFC above the eps-corrected null's 99th percentile)
- **Excess suspects**: session where the observed suspect count exceeds the 95% CI of the binomial null on that count

## Requirements

- Python 3.8+
- `magpyneto2` library (magnetic processing and statistics)
- `ephysio` library (OpenEphys I/O)
- `pynwb` / `hdmf` (NWB file read/write)
- Standard scientific stack: numpy, pandas, scipy, matplotlib
- `ecdfbounds` for confidence bands in p-value plots

See `environment.yml` for full dependency list.

## Contributing

To add a new experiment:

1. Create a YAML file in `experiments/` with the paradigm, recording parameters, and analysis config
2. Run processing: `python pipeline/processing.py --experiment name`
3. Run analysis: `python pipeline/analysis.py --experiment name`
4. Re-aggregate: `python pipeline/aggregate.py`
5. Regenerate figures: `python pipeline/manuscript/fig*.py`

See [.claude/CLAUDE.md](.claude/CLAUDE.md) for detailed YAML field reference.

## Citation

If you use this pipeline or data, please cite:

```
[INSERT FULL CITATION WITH DOI]
```

## License

[INSERT LICENSE]

## Contact

For questions or data access requests:
- Markus Meister (meister@caltech.edu)
- Daniel Pollak (dpollak@caltech.edu)

## Acknowledgments

This work was a collaboration between the Meister and Wagenaar labs at Caltech and partner laboratories: Engert (Harvard), Gutfreund (Technion), and others.
