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

If you just want to regenerate the manuscript figures:

#### Prerequisites
Download the analysis pickle files from CaltechDATA (DOI: [INSERT DOI]):
```bash
# Download all *_analysis.pickle files and place in data/ directory
# Also download precomputed/ folder (mouse and owl pickles)
```

#### Generate Figures

```bash
# From the repo root directory (magneto2 conda env):

# Aggregate all analysis pickles into a single parquet
python pipeline/aggregate.py

# Generate manuscript figures
python pipeline/manuscript/fig1.py    # Exemplar NPIX + GCaMP + p-value diagnostics
python pipeline/manuscript/fig2.py    # P-value and q-value uniformity
python pipeline/manuscript/fig3.py    # Excess-count barplots and c-hat distributions
python pipeline/manuscript/fig4.py    # Modulation sensitivity simulation
```

Output figures: `figs/paper/Fig1.pdf`, `Fig2.pdf`, `Fig3.pdf`, `Fig4.pdf`

## Full Pipeline (For Developers)

### Pipeline Order

```
raw data → processing → analysis → aggregate → manuscript figures
```

### Stage 1: Processing

Convert raw neural data to spike/fluorescence modulation DataFrame:

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

**Output:** `data/{name}_processing.pickle`

#### Processing pickle contents

`data/{name}_processing.pickle` is a pandas DataFrame (`modulation_df`) with one row per spike (electrophysiology) or one row per imaging frame (GCaMP), containing:

| Column | Description |
|---|---|
| `period` | Integer index of which stimulus cycle the spike/frame fell in |
| `spk` | Spike time in seconds |
| `phase` | Phase within the stimulus cycle (0–2π radians) |
| `freq` | Stimulus frequency in Hz |
| `id` | Unit or cluster ID |
| `rec` | Recording name; in multi-stimulus sessions encodes stimulus type and orientation (e.g. `recname_Mag`, `recname_45` for 45° gratings) |

**Per-paradigm details:**

- **`openephys`** — Standard columns above. One row per spike from Kilosort-sorted units during magnetic stimulation. Only spikes within the concatenated recording window are included; spike times are relative to the start of each recording segment.

- **`openephys_multistim`** — Same as `openephys`, but `modulation_df` is the vertical concatenation of magnetic trials and all auxiliary stimulus blocks (visual gratings, white noise, oddball, visual bars). The `rec` column encodes both the recording name and stimulus identity (e.g. `20230413_Mag_Rec` for magnetic trials, `20230413_visual_90` for 90° gratings, `20230413_WN` for white noise). Each auxiliary block uses its own stimulus frequency in `freq`.

- **`gutfreund`** — Standard columns plus `spk_samples` (spike time in raw AP samples), `label` (Kilosort quality label), and `recname` (basename of the data folder). Only spikes that fall within the last detected magnet-on TTL window are retained.

- **`spikeglx_direct`** — Standard columns. Phase and period are derived by linearly interpolating within each threshold-crossing interval of the smoothed NIDAQ magnetic channel.

- **`engert` / `medaka`** — No processing pickle; these paradigms skip the processing stage entirely because suite2p outputs are pre-computed. Run `pipeline/analysis.py` directly.


### Stage 2: Analysis

Perform Fourier spectral analysis and build analysis DataFrames:

```bash
# Single experiment
python pipeline/analysis.py --experiment 20230413_firstsite

# All experiments (parallelized)
python pipeline/analysis.py --all --workers 4

# Subset by species (same filters as processing)
python pipeline/analysis.py --all --filter engert --workers 8
```

**Output:** `data/{name}_analysis.pickle` + diagnostic PDFs in `figs/analysis/`

### Stage 3: Aggregate

Combine all analysis pickles + precomputed species into a single parquet:

```bash
python pipeline/aggregate.py
```

**Output:** `data/manuscript/all_fourier_df.parquet`

### Stage 4: Manuscript Figures

Generate publication figures (run aggregate.py first):

```bash
python pipeline/manuscript/fig1.py
python pipeline/manuscript/fig2.py
python pipeline/manuscript/fig3.py
python pipeline/manuscript/fig4.py --recompute  # --recompute to force re-simulation
```

**Output:** `figs/paper/Fig*.pdf`

## Data Access

### Analysis Pickles (For Regenerating Figures)

Download from CaltechDATA [INSERT DOI]:
- `data/*_analysis.pickle` — processed analysis DataFrames for all experiments
- `data/precomputed/mouse_analysis.pickle` — KyuHyunLee mouse recordings
- `data/precomputed/owl_analysis.pickle` — Gutfreund barn owl recordings

### Processing Pickles (For Re-Analysis)

Download from CaltechDATA [INSERT DOI]:
- `data/*_processing.pickle` — intermediate spike/fluorescence modulation data
- `data/MM_*.pickle` — diagnostic modulation matrices (OpenEphys only)

**Note:** Processing pickles are large (2.3 GB total). Only download if you plan to re-run analysis with modified parameters.

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
  Q: 100
  mag_rec_substring: "Mag"
  visual_rec_substring: "visual"
  # ... stimulus-specific parameters
```

**Key YAML fields:**
- `paradigm` — recording type (openephys, engert, medaka, etc.)
- `analysis.Q` — Fourier window width (number of off-frequency bins on each side)
- `analysis.f` — stimulus frequency in Hz
- `iscell_threshold` / `npix_threshold` — suite2p cell filtering (engert/medaka)

See [.claude/CLAUDE.md](.claude/CLAUDE.md) for detailed field semantics.

## Verification

Verify new outputs against original MagnetSearch/data/ pickles:

```bash
python verify_outputs.py                        # all experiments
python verify_outputs.py 20230413_firstsite     # one experiment
python verify_outputs.py --stage processing --experiments 20220916 20230415
```

Expected results:
- **PASS** — DataFrames match exactly (column order and dtypes ignored)
- **STALE_OLD** — Analysis differs from old pickle, but `find_outliers()` on old modulation_df yields new result (acceptable; algorithm evolved)
- **FAIL** — Genuine mismatch (investigate)

## Project Structure

```
.
├── README.md                          # This file
├── pipeline/
│   ├── processing.py                  # Processing stage CLI
│   ├── analysis.py                    # Analysis stage CLI
│   ├── aggregate.py                   # Combine pickles → parquet
│   ├── schema.py                      # YAML config dataclasses
│   ├── dispatch.py                    # Paradigm routing
│   ├── paradigms/                     # Recording modality handlers
│   ├── analysis_stages/               # Per-paradigm analysis
│   ├── diagnostics/                   # Diagnostic figure generation
│   └── manuscript/
│       ├── fig1.py                    # Exemplar NPIX + GCaMP
│       ├── fig2.py                    # P/q-value uniformity
│       ├── fig3.py                    # Excess-count barplots
│       └── fig4.py                    # Sensitivity simulation
├── experiments/                       # YAML experiment configs
├── data/
│   ├── *_processing.pickle            # (generated) Processing outputs
│   ├── *_analysis.pickle              # (generated) Analysis outputs
│   ├── precomputed/                   # (download) Mouse/owl pickles
│   └── manuscript/
│       └── all_fourier_df.parquet     # (generated) Aggregated data
├── figs/
│   ├── analysis/                      # (generated) Diagnostic PDFs
│   ├── processing/                    # (generated) Processing diagnostics
│   └── paper/                         # (generated) Manuscript figures
└── .claude/
    └── CLAUDE.md                      # Detailed technical docs
```

## Key Concepts

### Normalized Fourier Response (NFR)

The primary test statistic for detecting neural modulation:

$$\hat{c} = \frac{|c_s|}{\hat{\sigma}}$$

where $c_s$ is the Fourier coefficient at stimulus frequency and $\hat{\sigma}$ is the RMS of coefficients at nearby frequencies.

Under the null hypothesis (no modulation), $\hat{c}$ follows the Rayleigh distribution:
$$P_0(\hat{c}) = \hat{c} \exp(-\frac{1}{2}\hat{c}^2)$$

### P-Values and Q-Values

- **p-value**: Probability of observing $\hat{c}$ ≥ measured value under null hypothesis
- **q-value**: False discovery rate (Storey's method) — minimum FDR threshold at which this measurement is significant

### Suspects and Excess Suspects

- **Suspect**: neuron with p-value < 0.01 (NFR above 99th percentile of null)
- **Excess suspects**: session where observed suspect count exceeds 95% CI of binomial null

## Requirements

- Python 3.8+
- `magpyneto2` library (magnetic processing and statistics)
- `ephysio` library (OpenEphys I/O)
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
