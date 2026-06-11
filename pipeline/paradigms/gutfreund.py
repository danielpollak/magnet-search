"""
Paradigm: gutfreund

Uses Gutfreund_generator() for Q117 and Q134 recordings.
Phase is computed arithmetically from spike times and stimulation frequency.
The TTL schmitt trigger defines the magnet-on window; only spikes within
that window are included.
"""
import os
import pickle
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use

import numpy as np
import pandas as pd

from magpyneto2 import schmitt
from magpyneto2.gutfreund_helpers import Gutfreund_generator


def run_processing(cfg):
    label = "good" if cfg.good else None
    locations_freqs = [(cfg.aggregated_path, trial.frequency) for trial in cfg.trials]

    get_phase = lambda times, period: (times % period) / period * np.pi * 2

    modulation_df_l = []

    for (data_path, freq, gutfreund_files, gutfreund_data,
         relevant_measures, conversion_rates) in Gutfreund_generator(locations_freqs, label):

        (TTL_trace, AP_last_trace, AP_sr, all_sts_d, all_sts,
         unit_df, NIDAQ_recording, cap, ttl_df_unfilt, timestamp_df) = gutfreund_data

        (result, bins, fps, NIDAQ_to_AP, AP_to_NIDAQ, AP_sr, NIDAQ_sr) = conversion_rates

        # Detect last magnet-on window from TTL trace
        ons, offs = schmitt(TTL_trace.astype(float), 60, 20)
        on  = ons[-1]  * NIDAQ_to_AP   # AP samples
        off = offs[-1] * NIDAQ_to_AP   # AP samples

        period_s = 1 / freq  # seconds per stimulation cycle

        for unit_id, st in all_sts_d.items():
            if len(st) <= 50:
                continue
            # Only spikes during magnet-on window
            st_samples = st[(st > on) & (st < off)]
            if len(st_samples) == 0:
                continue
            st_s = st_samples / AP_sr   # seconds
            periods = [t // period_s for t in st_s]
            modulation_df_l.append(pd.DataFrame({
                "period":      periods,
                "spk":         st_s,
                "phase":       get_phase(st_s, period_s),
                "spk_samples": st_samples,
                "freq":        freq,
                "id":          unit_id,
                "rec":         os.path.basename(data_path),
                "label":       label,
                "recname":     os.path.basename(data_path),
            }))

    modulation_df = pd.concat(modulation_df_l).dropna()

    with open(cfg.processing_path(), "wb") as f:
        pickle.dump(modulation_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    from pathlib import Path
    from pipeline.diagnostics.processing import plot_recording_timeline
    diag_dir = Path(cfg.data_dir).parent / "figs" / "processing"
    diag_dir.mkdir(parents=True, exist_ok=True)
    plot_recording_timeline(cfg, None, modulation_df, diag_dir)
