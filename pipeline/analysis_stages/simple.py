"""Analysis stage for single-Q experiments (openephys, gutfreund, spikeglx_direct)."""
import os
from pathlib import Path

from magpyneto2 import find_outliers
from pipeline import nwb_io


def run_analysis(cfg):
    if not os.path.exists(cfg.nwb_path()):
        raise FileNotFoundError(
            f"{cfg.nwb_path()} not found -- run the processing stage for "
            f"{cfg.name} first.")

    io_p, nwbfile = nwb_io.read_nwbfile(cfg.nwb_path())
    modulation_df = nwb_io.build_modulation_frame(nwbfile, good_only=cfg.good)
    io_p.close()

    full_fourier_df, log_dict = find_outliers(
        modulation_df, Q=cfg.analysis.Q, diagnostics=False)

    nwb_io.append_results(
        cfg.nwb_path(),
        lambda nwbfile: nwb_io.write_fourier_results(
            nwbfile, full_fourier_df, log_dict, cfg.analysis.Q),
    )

    # Read diagnostics input back from the just-written NWB file rather than
    # the in-memory objects above — proves read-back fidelity
    # ("traceability without recomputation").
    io_r, nwbfile_r = nwb_io.read_nwbfile(cfg.nwb_path())
    mod_df_nwb = nwb_io.build_modulation_frame(nwbfile_r, good_only=cfg.good)
    fourier_df_nwb = nwb_io.read_fourier_results_as_full_fourier_df(nwbfile_r)
    log_dict_nwb = nwb_io.read_log_dict_equivalent(nwbfile_r)

    from pipeline.diagnostics.analysis import plot_analysis_diagnostics
    diag_dir = Path(cfg.data_dir).parent / "figs" / "analysis"
    diag_dir.mkdir(parents=True, exist_ok=True)
    plot_analysis_diagnostics(cfg, mod_df_nwb, fourier_df_nwb, log_dict_nwb, diag_dir)
    io_r.close()
