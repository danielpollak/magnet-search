"""
Paradigm: gutfreund

Uses Gutfreund_generator() for Q117 and Q134 recordings.
Phase is computed arithmetically from spike times and stimulation frequency.
The TTL schmitt trigger defines the magnet-on window; only spikes within
that window are included.
"""
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use

import numpy as np
import pandas as pd

from magpyneto2 import schmitt
from magpyneto2.gutfreund_helpers import Gutfreund_generator
from pipeline import nwb_io


def run_processing(cfg):
    label = "good" if cfg.good else None
    locations_freqs = [(cfg.aggregated_path, trial.frequency) for trial in cfg.trials]

    get_phase = lambda times, period: (times % period) / period * np.pi * 2

    modulation_df_l = []
    nwb_epochs = []  # one entry per (data_path, freq) -- see NWB dual-write below
    nwbfile = nwb_io.create_nwbfile(cfg)

    for (data_path, freq, gutfreund_files, gutfreund_data,
         relevant_measures, conversion_rates) in Gutfreund_generator(locations_freqs, label):

        (TTL_trace, AP_last_trace, AP_sr, all_sts_d, all_sts,
         unit_df, NIDAQ_recording, cap, ttl_df_unfilt, timestamp_df,
         all_sts_d_full, unit_df_full) = gutfreund_data

        (result, bins, fps, NIDAQ_to_AP, AP_to_NIDAQ, AP_sr, NIDAQ_sr) = conversion_rates

        # Detect last magnet-on window from TTL trace
        ons, offs = schmitt(TTL_trace.astype(float), 60, 20)
        on  = ons[-1]  * NIDAQ_to_AP   # AP samples
        off = offs[-1] * NIDAQ_to_AP   # AP samples

        period_s = 1 / freq  # seconds per stimulation cycle
        rec_name = os.path.basename(data_path)

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
                "rec":         rec_name,
                "label":       label,
                "recname":     rec_name,
            }))

        # --- NWB dual-write prep (see .claude/plans — NWB replatform, Phase 3) ---
        # Each (data_path, freq) has its OWN independent Kilosort sorting
        # (unlike openephys's single concatenated catalog), so Units are
        # written per-recording with rec-scoping (see nwb_io.write_units_and
        # _spikes's `rec` param). `period`/`phase` here are computed
        # arithmetically on continuous float seconds (`t // period_s`, `t`
        # measured from absolute zero, not from `on`) rather than from real
        # Schmitt-trigger sample crossings, so `phase_method="arithmetic"`
        # tells build_modulation_frame to reproduce that formula directly
        # instead of routing it through the (sample-quantized)
        # period_crossings machinery every other paradigm uses -- see
        # write_epochs_table's phase_method docstring.
        nwb_io.write_units_and_spikes(
            nwbfile, all_sts_d_full, unit_df_full, sampling_rate=AP_sr, rec=rec_name,
            # gutfreund's own good-filter (get_all_spiketrains) matches on
            # unit_df.group, not KSLabel -- see write_units_and_spikes's
            # label_column docstring. These sessions went through phy
            # curation, so group can genuinely differ from KSLabel per
            # cluster; using the wrong column silently changes which units
            # count as "good" downstream.
            label_column="group")

        nwb_epochs.append({
            "rec": rec_name,
            "stim_type": "magnetic",
            "frequency": float(freq),
            "start_time": float(on) / AP_sr,
            "stop_time": float(off) / AP_sr,
            "period_crossings": np.array([int(round(on)), int(round(off))], dtype=np.int64),
            "phase_method": "arithmetic",
            # min_spikes (>50) is checked against the unit's FULL-SESSION
            # spike count above (`if len(st) <= 50: continue`, on
            # all_sts_d[unit_id] before windowing), not the windowed count
            # -- the windowed count only needs to be non-empty (`if
            # len(st_samples) == 0: continue`). See build_modulation_frame's
            # min_spikes_on_full_session docstring.
            "min_spikes_on_full_session": True,
            "_sampling_rate": AP_sr,
        })

    modulation_df = pd.concat(modulation_df_l).dropna()

    # Every gutfreund recording in practice shares the same AP_sr, but each
    # epoch computed its own above rather than assuming it -- guard against
    # silently mixing sample domains if that ever isn't true.
    sampling_rates = {ep.pop("_sampling_rate") for ep in nwb_epochs}
    if len(sampling_rates) > 1:
        raise ValueError(
            f"gutfreund recordings in {cfg.name} have different AP sampling "
            f"rates ({sampling_rates}) -- write_epochs_table needs one "
            f"consistent sampling_rate for the whole file.")
    if sampling_rates:
        nwb_io.write_epochs_table(nwbfile, nwb_epochs, sampling_rate=sampling_rates.pop())
        nwb_io.write_nwbfile(nwbfile, cfg.nwb_path())

    from pathlib import Path
    from pipeline.diagnostics.processing import plot_recording_timeline
    diag_dir = Path(cfg.data_dir).parent / "figs" / "processing"
    diag_dir.mkdir(parents=True, exist_ok=True)
    plot_recording_timeline(cfg, None, modulation_df, diag_dir)
