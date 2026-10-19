# MagnetSearch Pipeline

A YAML-parameterized analysis pipeline for multispecies neural recording data, designed to identify neural responses to magnetic stimuli across brain regions and species.

This repository contains the complete analysis pipeline for the MagnetSearch collaboration, a distributed neurophysiology project searching for magnetic field responses in neural populations. It processes raw electrophysiology and two-photon imaging data, tests every neuron for modulation at the stimulus frequency, and generates the manuscript figures.

- Recordings from 7 species: mouse, zebra finch, pigeon, quail, owl, zebrafish, medaka (Neuropixels, OpenEphys, SpikeGLX, suite2p)
- One YAML file per experiment in `experiments/`
- One [NWB](https://www.nwb.org/) file per experiment, holding both the processed recording and the analysis results
- False discovery rate control and a sensitivity analysis of what modulation could have been detected

## Quick start

```bash
git clone https://github.com/danielpollak/magnet-search.git
cd magnet-search
conda env create -f environment.yml   # creates the magneto2 env
conda activate magneto2
pip install -e .
```

**Regenerating the figures** needs only the per-experiment NWB files, not the raw data. Download them (see [Data access](#data-access)) into `data/`, and the mouse and owl pickles into `data/precomputed/`, then from the repo root:

```bash
python pipeline/aggregate.py              # every data/{name}.nwb + precomputed pickles → one parquet
python pipeline/manuscript/fig1.py
python pipeline/manuscript/fig2.py
python pipeline/manuscript/fig3.py
python pipeline/manuscript/fig4.py
```

Figures are written to `figs/paper/`.

## Documentation

| Page | Contents |
|---|---|
| [Running the pipeline](docs/guide/pipeline.md) | Processing, analysis and aggregate stages; adding an experiment; verification; project layout |
| [Experiment configuration](docs/guide/configuration.md) | The YAML files and their fields |
| [GCaMP imaging](docs/guide/imaging.md) | Segmentation, full-resolution traces, protocols and ROI selection for the zebrafish and medaka |
| [NWB file structure](docs/guide/nwb_format.md) | What each stage writes into `data/{name}.nwb` |
| [Manuscript figures and numbers](docs/guide/manuscript.md) | Figure scripts, the two populations, `numbers.tex` and Overleaf |

Dated investigation reports (why a step is done the way it is) are the other files in [`docs/`](docs/).

## Key concepts

**Normalized Fourier coefficient (NFC).** The test statistic for each neuron:

$$\hat{c} = \frac{|c_s|}{\hat{\sigma}}$$

where $c_s$ is the Fourier coefficient at the stimulus frequency and $\hat{\sigma}$ the RMS of the coefficients at nearby off-frequencies. Without modulation, $\hat{c}$ follows a Rayleigh distribution, $P_0(\hat{c}) = \hat{c} \exp(-\hat{c}^2/2)$. The pipeline uses a version corrected for the finite number of off-frequencies (`eps`, from `Q`; see `magpyneto2.statistics.normalized_Fourier_PDF_corrected`).

- **p-value:** probability of an NFC at least as large under the corrected null.
- **q-value:** Storey's false discovery rate, the smallest FDR at which the neuron counts as significant.
- **Suspect:** a neuron with p < 0.01.
- **Excess suspects:** an experiment whose suspect count is above the 95th percentile of the null binomial distribution of that count.

## Data access

- **Experiment NWB files** (`data/*.nwb`) and the aggregated parquet are not in git. They are distributed as a GitHub Release (see the Releases page), or [INSERT CaltechDATA DOI] once published. Each file is self-contained.
- **Mouse and owl** can't be re-processed through this pipeline. Their precomputed results are `data/precomputed/mouse_analysis.pickle` (KyuHyunLee, Intan RHD) and `owl_analysis.pickle` (Gutfreund, precomputed coefficients), loaded by `aggregate.py`.
- **Raw recordings** (OpenEphys, SpikeGLX, Gutfreund recordings, GCaMP tiffs) are hosted on [institutional archive/server]. Contact the MagnetSearch collaboration for access.

## Requirements

- Python 3.8+
- `magpyneto2` (statistics and imaging helpers, in this repo) and `magpyneto`
- `ephysio` (OpenEphys I/O), `pynwb` / `hdmf`
- numpy, pandas, scipy, matplotlib; `ecdfbounds` for some confidence bands
- suite2p 1.1.0 in a separate `suite2p1.0` env, only to re-segment imaging data

See `environment.yml` for the full list.

## Citation

If you use this pipeline or data, please cite:

```
[INSERT FULL CITATION WITH DOI]
```

## License

[INSERT LICENSE]

## Contact

- Markus Meister (meister@caltech.edu)
- Daniel Pollak (dpollak@caltech.edu)

## Acknowledgments

This work was a collaboration between the Meister and Wagenaar labs at Caltech and partner laboratories: Engert (Harvard), Gutfreund (Technion), and others.
