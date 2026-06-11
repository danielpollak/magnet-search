"""Analysis stage for single-Q experiments (openephys, gutfreund, spikeglx_direct)."""
import pickle
from pathlib import Path

from magpyneto2 import find_outliers


def run_analysis(cfg):
    with open(cfg.processing_path(), "rb") as f:
        modulation_df = pickle.load(f)

    full_fourier_df, log_dict = find_outliers(
        modulation_df, Q=cfg.analysis.Q, diagnostics=False)

    with open(cfg.analysis_path(), "wb") as f:
        pickle.dump(full_fourier_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    from pipeline.diagnostics.analysis import plot_analysis_diagnostics
    diag_dir = Path(cfg.data_dir).parent / "figs" / "analysis"
    diag_dir.mkdir(parents=True, exist_ok=True)
    plot_analysis_diagnostics(cfg, modulation_df, full_fourier_df, log_dict, diag_dir)
