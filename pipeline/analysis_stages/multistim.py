"""
Analysis stage for multi-stimulus experiments (openephys_multistim).
Splits modulation_df by rec-name substring and applies per-stimulus-type Q values.
"""
import os
from pathlib import Path

from magpyneto2 import find_outliers
from pipeline import nwb_io


def run_analysis(cfg):
    a = cfg.analysis

    if not os.path.exists(cfg.nwb_path()):
        raise FileNotFoundError(
            f"{cfg.nwb_path()} not found -- run the processing stage for "
            f"{cfg.name} first.")

    io_p, nwbfile = nwb_io.read_nwbfile(cfg.nwb_path())
    modulation_df = nwb_io.build_modulation_frame(nwbfile, good_only=cfg.good)
    io_p.close()

    # Detect whether this experiment has per-stimulus Q values
    has_multiq = a.mag_Q > 0

    # (fourier_df, log_dict, Q) per sub-analysis -- written to NWB below in
    # the SAME per-Q groups find_outliers() was actually called with
    # (write_fourier_results is safe to call multiple times on one nwbfile,
    # see its docstring).
    nwb_results = []

    if not has_multiq:
        # Fall back to single Q if YAML only specifies analysis.Q
        full_fourier_df, log_dict = find_outliers(
            modulation_df, Q=a.Q, diagnostics=False)
        nwb_results.append((full_fourier_df, log_dict, a.Q))
    else:
        mag_mask = [a.mag_rec_substring in rec for rec in modulation_df.rec]
        visual_mask = [a.visual_rec_substring in rec for rec in modulation_df.rec]
        wn_mask = [a.wn_rec_substring in rec for rec in modulation_df.rec]

        for mask, Q, label in [
            (mag_mask,    a.mag_Q,    "mag"),
            (visual_mask, a.visual_Q, "visual"),
            (wn_mask,     a.WN_Q,     "WN"),
        ]:
            sub = modulation_df.loc[mask]
            if sub.empty:
                continue
            fourier_df, log_dict = find_outliers(sub, Q=Q, diagnostics=False)
            nwb_results.append((fourier_df, log_dict, Q))

    nwb_io.append_results(
        cfg.nwb_path(),
        lambda nwbfile: [
            nwb_io.write_fourier_results(nwbfile, fdf, ld, Q)
            for fdf, ld, Q in nwb_results
        ],
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
