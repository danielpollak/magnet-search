# Experiment configuration

Each experiment is one YAML file in `experiments/`, parsed into the dataclasses in `pipeline/schema.py`:

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

## Paradigms

| `paradigm:` | When to use |
|---|---|
| `openephys` | Pure-magnetic OpenEphys sessions |
| `openephys_multistim` | OpenEphys sessions with auxiliary stimuli (visual gratings, white noise, oddball, bars) |
| `gutfreund` | Q117, Q134 recordings |
| `spikeglx_direct` | SpikeGLX binary files (Q146, Q148, Q_magner) |
| `engert` | Zebrafish GCaMP two-photon sessions |
| `medaka` | Medaka GCaMP two-photon sessions |
| `manual` | Skipped with a warning |

## Common fields

- `paradigm`: recording type, as above.
- `analysis.f`: stimulus frequency in Hz.
- `analysis.Q_frac`: half-width of the off-frequency window used to estimate $\hat\sigma$, as a fraction of `analysis.f` (half-width in Hz = `Q_frac * f`). Converted to a bin count by `magpyneto2.statistics.bins_for_fraction`.

## Electrophysiology fields

- `good`: keep only Kilosort `good` clusters (filtered at read time).
- `stream_id`: OpenEphys stream (`"0"` for older sessions, `"2"` for newer).
- `analysis.mag_rec_substring` / `visual_rec_substring` / ...: how `openephys_multistim` tells recording types apart; `analysis.mag_Q_frac` / `visual_Q_frac` / `WN_Q_frac` override `Q_frac` per type.
- `auxiliary_stimuli:`: a list dispatched by `kind:` (`visual_gratings`, `white_noise`, `oddball`, `visual_bars`).

## Imaging fields (`engert` / `medaka`)

- `session_path`: the session's directory on the NAS (source of the tiffs).
- `suite2p_path`: the run's local `plane0/` (see [GCaMP imaging](imaging.md)).
- `tiff_name`: this trial's tiff, which picks its frames out of the run.
- `sample_period`: seconds per frame (1.0; 1.0212 for the 2022 Q1 sessions, [`../frame_timing.md`](../frame_timing.md)).
- `iscell_threshold` / `npix_threshold`: ROI selection, strict `>` (0.5 and 9 in every YAML).
- `coverage_threshold`: share of 60 s windows an ROI must be active in; 0 (off) in every YAML.
- `body_outline`: the fish outline in the mean image, set with `docs/nfc_finite_sample_bias/body_outline_gui.ipynb`.
- `analysis.visual_f`: an optional second, independent analysis frequency for recordings with a visual stimulus (1/60 Hz); unset by default.
- `date` / `subject_id`: medaka only, written straight into the results.

Every field, including the paradigm-specific compatibility flags, is defined in `pipeline/schema.py`; the paradigm modules in `pipeline/paradigms/` document how they are used.
