# Running the pipeline

```
raw data → processing → analysis → aggregate → manuscript figures
```

Every experiment's outputs live in a single `data/{name}.nwb` file; there are no per-experiment pickles. Processing and analysis are two separate CLI runs that both write into that same file (see [NWB file structure](nwb_format.md) for what each stage adds). Run everything from the repo root in the `magneto2` conda env.

## Stage 1: Processing

Read raw recording data and write the `Units`/`stimulus_epochs` tables (electrophysiology) or `PlaneSegmentation`/`RoiResponseSeries` (2-photon imaging) into `data/{name}.nwb`:

```bash
# Single experiment
python pipeline/processing.py --experiment 20230413_firstsite

# All experiments (parallelized)
python pipeline/processing.py --all --workers 24

# Subset: --filter matches a substring of the YAML filename
python pipeline/processing.py --all --filter engert    # zebrafish only
python pipeline/processing.py --all --filter medaka    # medaka only
python pipeline/processing.py --all --filter 2023      # pigeon only (2023 sessions)
python pipeline/processing.py --all --filter Q         # quail only
```

**Output:** `data/{name}.nwb` (created)

`engert`/`medaka` (suite2p imaging) read the session's suite2p output and its full-resolution traces (see [GCaMP imaging](imaging.md)) and write *all* ROIs, unfiltered. ROI selection happens at analysis time instead, so tuning those thresholds only requires re-running analysis, not reprocessing. `mouse` and `owl` have no processing stage at all (see the README's Data access section); `manual` paradigms are skipped with a warning.

## Stage 2: Analysis

Read `data/{name}.nwb` back, run the Fourier significance test (`fit_fourier_sig()` for ephys, `fit_Fourier()` for imaging), and append the results into the *same* NWB file:

```bash
python pipeline/analysis.py --experiment 20230413_firstsite
python pipeline/analysis.py --all --workers 24
python pipeline/analysis.py --all --filter engert --workers 24
```

**Output:** `data/{name}.nwb` (appended: adds a `processing["analysis"]` module) + diagnostic PDFs in `figs/analysis/`

## Stage 3: Aggregate

Combine every `data/{name}.nwb` file's analysis tables with the precomputed mouse/owl DataFrames into a single parquet. Re-run it after any change to analysis outputs, before regenerating figures. An `.nwb` file with no matching YAML in `experiments/` is skipped.

```bash
python pipeline/aggregate.py
```

**Output:** `data/manuscript/all_fourier_df.parquet`

## Stage 4: Manuscript figures

See [Manuscript figures](manuscript.md).

## Adding an experiment

1. Create a YAML file in `experiments/` (see [Configuration](configuration.md)).
2. `python pipeline/processing.py --experiment name`
3. `python pipeline/analysis.py --experiment name`
4. `python pipeline/aggregate.py`
5. Regenerate the figures.

## Verification

Compare current NWB outputs against the frozen original pickle-era fixtures in `MagnetSearch/data/` (permanently archived):

```bash
python verify_outputs.py                        # all experiments
python verify_outputs.py 20230413_firstsite     # one (or more) experiments, by name
```

Expected results:
- **PASS**: DataFrames match exactly (column order and dtypes ignored).
- **DRIFT_TOLERATED**: matches within a small float tolerance (HDF5 round-trip, not a logic bug).
- **STALE_OLD**: analysis differs from the old fixture, but re-running `fit_fourier_sig()` on the *old* modulation_df reproduces the current NWB result (acceptable: the algorithm evolved since the fixture was generated).
- **STALE_SEMANTICS**: same reconciliation as STALE_OLD, but because `Q` was redefined (raw off-frequency bin count → fraction of the analyzed frequency); expect this on essentially every NPIX analysis row.
- **FAIL**: genuine mismatch (investigate).

`engert`/`medaka` have no old fixture (there was no processing stage before the NWB migration); for those, verification re-runs the analysis computation independently and diffs it against what's persisted in the NWB file, as a serialization-fidelity check.

## Project structure

```
.
├── README.md
├── pipeline/
│   ├── processing.py                  # Processing stage CLI
│   ├── analysis.py                    # Analysis stage CLI
│   ├── aggregate.py                   # Combine {name}.nwb + precomputed/ → parquet
│   ├── nwb_io.py                      # NWB read/write library
│   ├── ophys_extraction.py            # Full-resolution imaging traces (suite2p1.0 env, GPU)
│   ├── schema.py                      # YAML config dataclasses
│   ├── dispatch.py                    # Paradigm routing
│   ├── paradigms/                     # Recording modality handlers
│   ├── analysis_stages/               # Per-paradigm analysis
│   ├── diagnostics/                   # Diagnostic figure generation
│   └── manuscript/                    # Figure scripts, numbers.tex export (see manuscript.md)
├── experiments/                       # YAML experiment configs
├── magpyneto2/                        # Statistics and imaging helpers
├── data/
│   ├── *.nwb                          # (generated/downloaded) Per-experiment data + analysis results
│   ├── precomputed/                   # (download) Mouse/owl pickles, not re-processable
│   └── manuscript/                    # (generated) all_fourier_df.parquet, cached simulations
├── figs/
│   ├── analysis/                      # (generated) Diagnostic PDFs
│   ├── processing/                    # (generated) Processing diagnostics
│   └── paper/                         # (generated) Manuscript figures
└── docs/
    ├── guide/                         # These reference pages
    └── *.md                           # Dated investigation reports
```
