"""Routes experiment configs to the correct paradigm and analysis modules."""
import importlib

PARADIGM_MAP = {
    "openephys":           "pipeline.paradigms.openephys",
    "openephys_multistim": "pipeline.paradigms.openephys_multistim",
    "gutfreund":           "pipeline.paradigms.gutfreund",
    "spikeglx_direct":     "pipeline.paradigms.spikeglx_direct",
    "engert":              "pipeline.paradigms.engert",
    "medaka":              "pipeline.paradigms.medaka",
}

ANALYSIS_MAP = {
    "openephys":           "pipeline.analysis_stages.simple",
    "openephys_multistim": "pipeline.analysis_stages.multistim",
    "gutfreund":           "pipeline.analysis_stages.simple",
    "spikeglx_direct":     "pipeline.analysis_stages.simple",
    "engert":              "pipeline.analysis_stages.engert",
    "medaka":              "pipeline.analysis_stages.medaka",
}


def run_processing(cfg):
    if cfg.paradigm == "manual":
        print(f"[SKIP] {cfg.name}: paradigm=manual — run manually from MagnetSearch/code/processing/")
        return
    module = importlib.import_module(PARADIGM_MAP[cfg.paradigm])
    module.run_processing(cfg)


def run_analysis(cfg):
    if cfg.paradigm == "manual":
        print(f"[SKIP] {cfg.name}: paradigm=manual — run manually from MagnetSearch/code/analysis/")
        return
    module = importlib.import_module(ANALYSIS_MAP[cfg.paradigm])
    module.run_analysis(cfg)
