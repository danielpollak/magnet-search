"""
Analysis stage for multi-stimulus experiments (openephys_multistim).
Splits modulation_df by rec-name substring and applies per-stimulus-type Q values.
"""
import os
from pathlib import Path

from magpyneto2 import fit_fourier_sig
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

    # Detect whether this experiment has per-stimulus Q_frac values
    has_multiq = a.mag_Q_frac > 0

    # (fourier_df, log_dict) per sub-analysis -- written to NWB below.
    # write_fourier_results no longer takes a Q argument (it sources each
    # group's actual bin count from log_dict itself -- see its docstring),
    # so there's no need to carry a Q/fraction alongside each pair anymore.
    nwb_results = []

    if not has_multiq:
        # Fall back to single fraction if YAML only specifies analysis.Q_frac
        full_fourier_df, log_dict = fit_fourier_sig(
            modulation_df, Q_frac=a.Q_frac, diagnostics=False)
        nwb_results.append((full_fourier_df, log_dict))
    else:
        mag_mask = [a.mag_rec_substring in rec for rec in modulation_df.rec]
        visual_mask = [a.visual_rec_substring in rec for rec in modulation_df.rec]
        wn_mask = [a.wn_rec_substring in rec for rec in modulation_df.rec]

        for mask, frac, label in [
            (mag_mask,    a.mag_Q_frac,    "mag"),
            (visual_mask, a.visual_Q_frac, "visual"),
            (wn_mask,     a.WN_Q_frac,     "WN"),
        ]:
            sub = modulation_df.loc[mask]
            if sub.empty:
                continue
            fourier_df, log_dict = fit_fourier_sig(sub, Q_frac=frac, diagnostics=False)
            nwb_results.append((fourier_df, log_dict))

    nwb_io.rebuild_and_replace_analysis(
        cfg.nwb_path(),
        lambda nwbfile: [
            nwb_io.write_fourier_results(nwbfile, fdf, ld)
            for fdf, ld in nwb_results
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
