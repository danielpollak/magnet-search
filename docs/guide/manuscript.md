# Manuscript figures and numbers

Run `python pipeline/aggregate.py` first; every script below reads `data/manuscript/all_fourier_df.parquet` (Figs 1 and 4 also read individual NWB files). Outputs go to `figs/paper/`; supplementary figures (`Fig3S1_second_harmonic.pdf`, `Fig3S2_all_recordings.pdf`) to `figs/paper/supplement/`; diagnostic figures (the scripts not in the manuscript, plus `fig4.py --rec-boundaries`/`--panel-a-unit`) go to `figs/paper/diagnostic/`.

## Figures

| Script | Output | Content |
|---|---|---|
| `fig1.py` | `Fig1.pdf` | Analysis pipeline on one pigeon unit under magnetic and visual stimulation; NFC distributions with p-value ECDF insets |
| `fig2.py` | `Fig2.pdf` | Suspect counts per experiment against the null (A magnetic, B visual & auditory); p-value ECDF minus uniform, one recording per neuron, the one closest to 3 Hz as in Fig 3 (C, D) |
| `fig3.py` | `Fig3.pdf`, `Fig3_by_modality.pdf`, `Fig3S2_all_recordings.pdf` | Sorted p- and q-values for four populations. `Fig3.pdf` uses one recording per neuron, the one closest to 3 Hz (positive-control ties, e.g. grating orientations, to a pseudo-random one); `Fig3_by_modality.pdf` splits the positive control into white noise and every visual stimulus, spikes and imaging alike (gratings, bars, oddball flashes, fish gratings; `Fig3_by_modality_CB.pdf`: cerebellum in C/D; `Fig3_WN_oddball.pdf`: white noise and oddball, HP and CB pages); the supplement shows every recording, split by occurrence (page 1) and pooled, by modality (page 2) |
| `fig3s1_second_harmonic.py` | `Fig3S1_second_harmonic.pdf` | Fig 3 at twice the stimulus frequency |
| `fig3s4_bucketings.py` | `Fig3S4_bucketings.pdf` | Other ways of bucketing Fig 3's populations (species, frequency, brain area, equal-sized buckets); page numbers printed on each page, list in the script's docstring |
| `fig4.py --recompute` | `Fig4.pdf` | Modulation-detectability simulation (`--recompute` forces re-simulation, `--workers N` parallelizes the sweep) |

Further supplementary and diagnostic scripts (`fig3s3_reproducibility.py`, `fig3_modulation_depth.py`, `fig2_variants.py`, `fig2A_imaging_annotated.py`, `fig4B_diagnostics.py`, `fig1E_qfrac_diagnostic.py`, `fig1_auditory.py`) describe themselves in their docstrings. `fig1_extract_raw_snippets.py` is a one-time NAS extraction that caches Fig 1's raw-voltage snippets locally. `oddball_stimulus.py` draws the oddball flash train (`supplement/oddball_stimulus_strip.pdf`) and a real-time video of it per session (`supplement/oddball_stimulus_{session}.mp4`) from the photodiode times that `oddball_extract_ttl.py` caches once from the NAS.

Plot styling (fonts, colours, figure sizes) is shared through `format_parameters.py`.

## Populations

`magpyneto2.statistics.get_poscontrols_negresults` splits the parquet into the two populations every figure uses:
- **Magnetic** (negative results): recordings at a magnetic stimulus frequency. Imaging recordings with a visual stimulus are not included.
- **Visual & auditory** (positive controls): recordings at a visual or auditory stimulus frequency. Imaging recordings with the coil on are not included (since 2026-10-18).

A neuron is identified by `(species, ID, date, id)`.

## Numbers in the manuscript text

The manuscript is a separate git repository, a clone of the Overleaf project, at `pipeline/manuscript/<overleaf_project_id>/` (gitignored here). Numbers computed by the pipeline go into its `numbers.tex`, and `main.tex` cites them as `\importnumber{key}`:
- `tex_numbers.write_numbers(source, {key: value})` rewrites only that source's `% BEGIN source` / `% END source` block, so several scripts can each own their numbers.
- Current sources: `fig1.py` (exemplar NFCs, neuron counts), `fig2.py` (share of experiments above the null bound), and `neuron_counts.py` (species, animal and neuron counts; a jupytext script, open it as a notebook with `jupytext --to notebook`).
- An undefined key compiles as a bold `??key??` with a LaTeX warning.

`sync_to_overleaf.py` copies `figs/paper/*.pdf` into the clone (`--dry-run` to preview). Neither it nor `tex_numbers` commits or pushes; review and push the clone yourself, after merging the latest Overleaf edits.

## Figure scripts as notebooks

The figure scripts can be converted to Jupyter notebooks with [jupytext](https://jupytext.readthedocs.io/) to run them cell by cell:

```bash
jupytext --to notebook pipeline/manuscript/fig1.py   # or fig*.py
```

Each `.ipynb` lands next to its `.py`. Notebooks are not committed; re-run the command after the scripts change.
