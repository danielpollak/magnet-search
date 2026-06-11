"""
Paradigm: spikeglx_direct

SpikeGLX recordings (Gutfreund lab, Q146/Q148/Q_magner).
Loads NIDAQ + AP binary files directly, detects magnet periods via
threshold crossing on a smoothed NIDAQ channel, builds modulation_df.
No MM_d / save_diagnostics_MM — these scripts never had that.
"""
import glob
import os
import pickle

import numpy as np
import pandas as pd
import spikeinterface.extractors as se

from magpyneto2 import smooth, get_sampling_rates
from ephysio.kilosortIO import Reader


def run_processing(cfg):
    data_path = cfg.aggregated_path

    nidaq_path = glob.glob(data_path + r"/*.nidq.bin")[0]
    ap_path = glob.glob(data_path + r"/**/*.ap.bin")[0]
    ks_path = glob.glob(data_path + r"/**/*kilosort4")[0]

    AP_sr, NIDAQ_sr = get_sampling_rates(data_path)

    recording = se.BinaryRecordingExtractor(
        nidaq_path, NIDAQ_sr, cfg.nidaq_channels, dtype="int16")
    ap_recording = se.BinaryRecordingExtractor(
        ap_path, AP_sr, cfg.ap_channels, dtype="int16")

    ks_label = "good" if cfg.good else None
    reader = Reader(ks_path)
    all_sts = {
        cluster_id: spikes / AP_sr
        for cluster_id, spikes in reader.spikesbycluster(ks_label).items()
    }

    # Detect magnet periods via threshold crossing on smoothed NIDAQ channel
    mag_trace = recording.get_traces(channel_ids=[cfg.mag_channel]).flatten()
    dif = np.diff((smooth(mag_trace, cfg.smooth_window) < cfg.mag_threshold).astype(int))
    threshold_crossings = np.where(dif == 1)[0]

    # Build phase (theta) and period index arrays.
    # Arrays are AP-sized, but filled using raw NIDAQ sample indices —
    # this matches the original processing scripts exactly.
    n_ap_frames = ap_recording.get_num_frames()
    theta = np.full(n_ap_frames, np.nan)
    periods = np.full(n_ap_frames, np.nan)
    for i in range(1, len(threshold_crossings)):
        prev = threshold_crossings[i - 1]
        curr = threshold_crossings[i]
        theta[prev:curr] = np.linspace(0, 2 * np.pi, curr - prev)
        periods[prev:curr] = i

    freq = cfg.trials[0].frequency

    modulation_df_l = []
    for unit_id, st in all_sts.items():
        if len(st) < 50:
            continue
        ap_indices = (st * AP_sr).astype(int)
        # clip to valid range
        valid = (ap_indices >= 0) & (ap_indices < n_ap_frames)
        st_v = st[valid]
        idx_v = ap_indices[valid]
        modulation_df_l.append(pd.DataFrame({
            "period": periods[idx_v],
            "spk": st_v,
            "phase": theta[idx_v],
            "freq": freq,
            "id": unit_id,
            "rec": os.path.basename(data_path),
        }))

    modulation_df = pd.concat(modulation_df_l).dropna()

    with open(cfg.processing_path(), "wb") as f:
        pickle.dump(modulation_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    from pathlib import Path
    from pipeline.diagnostics.processing import plot_recording_timeline
    diag_dir = Path(cfg.data_dir).parent / "figs" / "processing"
    diag_dir.mkdir(parents=True, exist_ok=True)
    plot_recording_timeline(cfg, None, modulation_df, diag_dir)
