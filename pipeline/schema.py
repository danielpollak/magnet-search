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
    max_interval_s: float = 1.2  # oddball: keep events with inter-event gap < this
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
    Q: int = 100
    # stimulus frequency for Engert/GCaMP paradigm (NPIX ignores this)
    f: float = -1.0
    # multi-stim fields (used when paradigm == openephys_multistim)
    mag_Q: int = -1
    visual_Q: int = -1
    WN_Q: int = -1
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
    date: str = ""           # experiment date (YYYY-MM-DD), used by medaka analysis stage
    subject_id: str = ""     # animal/fish identifier, used by medaka analysis stage

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
        return os.path.join(self.data_dir, f"{self.name}_processing.pickle")

    def analysis_path(self) -> str:
        return os.path.join(self.data_dir, f"{self.name}_analysis.pickle")

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


def load_experiment(yaml_path: str) -> ExperimentConfig:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = ExperimentConfig(**raw)
    cfg.validate()
    return cfg


def load_all_experiments(experiments_dir: str) -> list[ExperimentConfig]:
    import glob
    paths = sorted(glob.glob(os.path.join(experiments_dir, "*.yml")))
    return [load_experiment(p) for p in paths]
