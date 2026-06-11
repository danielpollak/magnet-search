"""
Analysis stage for multi-stimulus experiments (openephys_multistim).
Splits modulation_df by rec-name substring and applies per-stimulus-type Q values.
"""
import pickle
from pathlib import Path

import pandas as pd

from magpyneto2 import find_outliers


def run_analysis(cfg):
    a = cfg.analysis

    with open(cfg.processing_path(), "rb") as f:
        modulation_df = pickle.load(f)

    # Detect whether this experiment has per-stimulus Q values
    has_multiq = a.mag_Q > 0

    all_log_dicts = {}

    if not has_multiq:
        # Fall back to single Q if YAML only specifies analysis.Q
        full_fourier_df, log_dict = find_outliers(
            modulation_df, Q=a.Q, diagnostics=False)
        all_log_dicts.update(log_dict)
    else:
        sub_dfs = []

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
            sub_dfs.append(fourier_df)
            all_log_dicts.update(log_dict)

        full_fourier_df = pd.concat(sub_dfs).reset_index(drop=True)

    with open(cfg.analysis_path(), "wb") as f:
        pickle.dump(full_fourier_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    from pipeline.diagnostics.analysis import plot_analysis_diagnostics
    diag_dir = Path(cfg.data_dir).parent / "figs" / "analysis"
    diag_dir.mkdir(parents=True, exist_ok=True)
    plot_analysis_diagnostics(cfg, modulation_df, full_fourier_df, all_log_dicts, diag_dir)
