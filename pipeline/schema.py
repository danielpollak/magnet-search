"""Config dataclasses loaded from experiment YAML files."""
from __future__ import annotations
from dataclasses import dataclass, field
import os
import yaml


@dataclass
class TrialConfig:
    frequency: float
    folder: str = ""
    recname: str = ""   # derived from folder basename if not given
    skips: int = 0

    def __post_init__(self):
        if self.recname == "" and self.folder:
            self.recname = os.path.basename(self.folder.rstrip("/\\"))


@dataclass
class AuxStimulusConfig:
    recname: str
    kind: str           # "visual_gratings" | "white_noise" | "oddball" | "visual_bars"
    channel: int
    thr_on: float
    thr_off: float
    frequency: float
    deststream: str
    sourcestream: str
    sourcebarcode: str
    n_orientations: int = 8
    trial_gap_samples: int = 120000
    iup_min_filter: int = 0
    duration_s: float = 300.0
    # oddball: each candidate trial is classified on TWO independent axes --
    # its own stimulus-on duration (onset->offset) and its own preceding
    # off/silence duration (previous offset->this onset) -- NOT the combined
    # down-to-down gap _handle_oddball used to use, which silently conflated
    # two distinct manipulations (a long-ON stimulus vs. a long-OFF/silence
    # period before an otherwise-normal stimulus) into one "deviant" bucket.
    # <= *_normal_max_s -> "normal" on that axis; >= *_long_min_s -> "long";
    # in between is a dead zone, dropped (ambiguous on that axis) from every
    # category. See _classify_oddball_trials in openephys_multistim.py.
    on_normal_max_s: float = 0.7
    on_long_min_s: float = 0.9
    off_normal_max_s: float = 0.7
    off_long_min_s: float = 0.9
    # WN multi-window: use floor-division period and normalized phase (20220916 style)
    wn_legacy_formula: bool = False
    # WN single-window: override the freq used in period/phase computation (bug-compat mode).
    # Original scripts sometimes used a leaked `freq` from the mag loop. Set to the leaked
    # value to reproduce old pickles exactly; leave at -1.0 to use aux_cfg.frequency.
    wn_period_freq: float = -1.0
    wav_path: str = ""
    orientation_csv: str = ""   # path to video_presentation.txt for visual_gratings


@dataclass
class AnalysisConfig:
    # Off-frequency half-window width, as a fraction of the frequency being
    # analyzed (half-width_Hz = Q_frac * freq) -- see bins_for_fraction in
    # magpyneto2/statistics.py. Sentinel -1.0 means "unset"; deliberately NOT
    # named `Q` (that used to be a raw integer bin count with a totally
    # different meaning) so an unmigrated YAML fails loudly at load instead
    # of silently reinterpreting an old integer as a fraction.
    Q_frac: float = -1.0
    # stimulus frequency for Engert/GCaMP paradigm (NPIX ignores this)
    f: float = -1.0
    # multi-stim fields (used when paradigm == openephys_multistim)
    mag_Q_frac: float = -1.0
    visual_Q_frac: float = -1.0
    WN_Q_frac: float = -1.0
    # oddball's long_on/long_off/long_both recs (see multistim.py) are split
    # out of visual_mask entirely and never run through fit_fourier_sig --
    # each is far too sparse (a handful of trials vs. hundreds of standard
    # ones) for any Q_frac to clear bins_for_fraction's minimum-bins floor.
    # They're still written as full NWB epochs + covered by the trial
    # diagnostics, just NWB-epochs/diagnostics-only, no Fourier group.
    mag_rec_substring: str = "mag"
    visual_rec_substring: str = "visual"
    wn_rec_substring: str = "WN"


@dataclass
class ExperimentConfig:
    name: str
    paradigm: str

    # common
    good: bool = True
    notes: str = ""

    # openephys / openephys_multistim
    cntlbarcodes: bool = False
    stream_id: str = "2"
    # per-row stream_ids for recordings that vary (e.g. 20220408); None entry = skip that row
    streams: list = field(default_factory=list)
    # cntlbarcodes value for per-recording Loaders (usually False even when global is True)
    recording_ldr_cntlbarcodes: bool = False
    # recname -> recording path override for unusual recordings (e.g. 20220621 Taeniopygia)
    recording_overrides: dict = field(default_factory=dict)
    threshold: int = 300
    window_size: int = 5
    aggregated_path: str = ""
    metadata_csv: str = ""

    # spikeglx_direct
    nidaq_channels: int = 9
    ap_channels: int = 385
    mag_channel: int = 0
    smooth_window: int = 50
    mag_threshold: int = -20000

    # engert / GCaMP / medaka
    session_path: str = ""   # session root dir (parent of suite2p/); tiffs live here too
    tiff_name: str = ""      # filename of the specific tiff to analyze (frames sliced via len_df)
    sample_period: float = 1.0  # seconds per frame (T in fit_Fourier); 1.02 for 2022_03_01
    iscell_threshold: float = 0.7
    npix_threshold: int = 20
    # subject/session metadata — historically medaka-only, promoted to a
    # universal field for the NWB replatform (NWBFile.session_start_time /
    # Subject.subject_id / Subject.species). Optional: create_nwbfile()
    # falls back to inferring session_start_time from settings.xml and
    # omits Subject entirely when subject_id is blank, so leaving these
    # unset does not block processing/analysis.
    date: str = ""           # experiment date (YYYY-MM-DD)
    subject_id: str = ""     # animal/fish/subject identifier
    species: str = ""        # e.g. "zebra finch", "Pigeon", "medaka"
    area: str = ""           # canonical brain-area label, e.g. "HP", "CB", "NCM", "wulst"
    # "mag" | "positive control" -- experiment-level default classification for
    # aggregate.py's `contingency` annotation column. For openephys/openephys_multistim,
    # this is the default applied to `trials`-derived recs only: `auxiliary_stimuli`
    # (visual gratings/white noise/oddball/bars) recs are always "positive control"
    # regardless of this field, since they are definitionally non-magnetic controls.
    contingency: str = "mag"

    trials: list = field(default_factory=list)
    auxiliary_stimuli: list = field(default_factory=list)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    # output paths (resolved after init)
    data_dir: str = ""

    def __post_init__(self):
        self.trials = [
            TrialConfig(**t) if isinstance(t, dict) else t
            for t in self.trials
        ]
        self.auxiliary_stimuli = [
            AuxStimulusConfig(**a) if isinstance(a, dict) else a
            for a in self.auxiliary_stimuli
        ]
        if isinstance(self.analysis, dict):
            self.analysis = AnalysisConfig(**self.analysis)
        # resolve data_dir relative to this file's repo root
        if not self.data_dir:
            self.data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data"
            )

    def processing_path(self) -> str:
        """Legacy naming convention, retired as of the NWB replatform's
        Phase 7 cutover -- no paradigm writes this file anymore (see
        nwb_path() below). Kept only so old, frozen fixtures under
        MagnetSearch/data/ and one-off debugging scripts can still name the
        pre-migration shape by convention."""
        return os.path.join(self.data_dir, f"{self.name}_processing.pickle")

    def analysis_path(self) -> str:
        """See processing_path()'s docstring -- same retirement, same
        reason."""
        return os.path.join(self.data_dir, f"{self.name}_analysis.pickle")

    def nwb_path(self) -> str:
        """Single-file NWB replacement for the processing/analysis pickle
        pair (see pipeline/nwb_io.py). Every paradigm's processing
        stage writes this file; analysis reads it back and appends results
        to it -- this is now the only per-experiment output artifact."""
        return os.path.join(self.data_dir, f"{self.name}.nwb")

    def validate(self):
        valid_paradigms = {"openephys", "openephys_multistim", "gutfreund", "spikeglx_direct", "manual", "engert", "medaka"}
        if self.paradigm not in valid_paradigms:
            raise ValueError(f"{self.name}: unknown paradigm '{self.paradigm}'")
        if self.paradigm in {"openephys", "openephys_multistim"} and not self.metadata_csv:
            raise ValueError(f"{self.name}: openephys paradigm requires metadata_csv")
        if self.paradigm in {"openephys", "openephys_multistim"} and not self.aggregated_path:
            raise ValueError(f"{self.name}: openephys paradigm requires aggregated_path")
        if self.paradigm in {"engert", "medaka"}:
            if not self.session_path:
                raise ValueError(f"{self.name}: {self.paradigm} paradigm requires session_path")
            if self.analysis.f <= 0:
                raise ValueError(f"{self.name}: {self.paradigm} paradigm requires analysis.f > 0")
        if self.paradigm == "openephys_multistim":
            valid_kinds = {"visual_gratings", "white_noise", "oddball", "visual_bars"}
            for aux in self.auxiliary_stimuli:
                if aux.kind not in valid_kinds:
                    raise ValueError(f"{self.name}: unknown aux stimulus kind '{aux.kind}'")
            if self.analysis.Q_frac <= 0 and self.analysis.mag_Q_frac <= 0:
                raise ValueError(
                    f"{self.name}: openephys_multistim requires analysis.Q_frac "
                    f"(single-fraction fallback) or analysis.mag_Q_frac/"
                    f"visual_Q_frac/WN_Q_frac > 0")
        elif self.paradigm != "manual" and self.analysis.Q_frac <= 0:
            raise ValueError(f"{self.name}: paradigm '{self.paradigm}' requires analysis.Q_frac > 0")


def load_experiment(yaml_path: str) -> ExperimentConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = ExperimentConfig(**raw)
    cfg.validate()
    return cfg


def load_all_experiments(experiments_dir: str) -> list[ExperimentConfig]:
    import glob
    paths = sorted(glob.glob(os.path.join(experiments_dir, "*.yml")))
    return [load_experiment(p) for p in paths]
