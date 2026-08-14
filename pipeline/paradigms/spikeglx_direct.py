"""
Paradigm: spikeglx_direct

SpikeGLX recordings (Gutfreund lab, Q146/Q148/Q_magner).
Loads NIDAQ + AP binary files directly, detects magnet periods via
threshold crossing on a smoothed NIDAQ channel, builds modulation_df.
No MM_d / save_diagnostics_MM — these scripts never had that.
"""
import glob
import os

import numpy as np
import pandas as pd
import spikeinterface.extractors as se

from magpyneto2 import smooth, get_sampling_rates, get_cluster_info
from ephysio.kilosortIO import Reader
from pipeline import nwb_io


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

    # Unfiltered variant for the NWB Units table (dual-write, below): every
    # cluster (not just `good`) plus cluster_info, so `cfg.good` filtering
    # happens at analysis time instead of being baked in permanently. Raw
    # sample values (not seconds) — matches write_units_and_spikes's expected
    # input, same convention as openephys.py.
    all_sts_full_samples = reader.spikesbycluster(label=None)
    udf_full = get_cluster_info(ks_path)
    # `reader.kslabel` is the label dict `spikesbycluster(ks_label)` above
    # ACTUALLY filtered on -- built from cluster_KSLabel.tsv, overridden by
    # cluster_group.tsv when present (see ephysio.kilosortIO.Reader.__init__).
    # cluster_info.tsv's own KSLabel/group columns are a separate export and
    # were confirmed on real data to NOT always agree with this (neither
    # column alone matched the actual `good` population) -- worse, some real
    # cluster ids present in cluster_KSLabel.tsv/spike_clusters.npy (and
    # therefore in `all_sts_full_samples`) are simply ABSENT from
    # cluster_info.tsv's own id numbering entirely (a real numbering
    # mismatch between the two exports, confirmed on real magnerNPX2_g0
    # data). Build the label table directly from reader.kslabel (covers
    # every cluster reader.spikesbycluster can return) and left-join
    # cluster_info.tsv's channel/position columns on top, best-effort.
    id_col = "cluster_id" if "cluster_id" in udf_full.columns else "id"
    label_df = pd.DataFrame({
        id_col: list(reader.kslabel.keys()),
        "resolved_label": list(reader.kslabel.values()),
    })
    udf_full = label_df.merge(udf_full, on=id_col, how="left")

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

    # --- NWB write (see .claude/plans — NWB replatform) ---
    # Single recording, single Kilosort catalog per experiment (unlike
    # gutfreund) -- Units are shared/rec="" exactly like openephys, just with
    # only one epoch. `period_crossings` are stored as the RAW threshold-
    # crossing sample indices unchanged: the legacy code above indexes the
    # AP-sized theta/periods arrays directly with these NIDAQ-domain indices
    # (see the comment above, "this matches the original processing scripts
    # exactly") -- a bug-compatibility quirk this file intentionally
    # preserves, so the NWB reconstruction must reuse the same raw values
    # rather than "fixing" them to a true NIDAQ->AP conversion.
    rec_name = os.path.basename(data_path)
    nwbfile = nwb_io.create_nwbfile(cfg)
    # label_column="resolved_label": see the reader.kslabel comment above --
    # neither of cluster_info.tsv's own KSLabel/group columns reliably
    # matched `ks_label="good"`'s actual selection on real data, so use
    # Reader's own resolved label directly instead of guessing a column.
    nwb_io.write_units_and_spikes(
        nwbfile, all_sts_full_samples, udf_full, sampling_rate=AP_sr, label_column="resolved_label")
    epochs = [{
        "rec": rec_name,
        "stim_type": "magnetic",
        "frequency": float(freq),
        "start_time": float(threshold_crossings[0]) / AP_sr,
        "stop_time": float(threshold_crossings[-1]) / AP_sr,
        "period_crossings": np.asarray(threshold_crossings, dtype=np.int64),
        # min_spikes (`if len(st) < 50: continue` above) is checked against
        # the unit's FULL-SESSION spike count, before any windowing -- any
        # spike outside [threshold_crossings[0], threshold_crossings[-1])
        # gets NaN period/phase (theta/periods are only filled inside that
        # range) and is dropped later by modulation_df's `.dropna()`, but
        # that's unrelated to the 50-count check. See
        # build_modulation_frame's min_spikes_on_full_session docstring.
        "min_spikes_on_full_session": True,
        # AP_sr here is read from metadata and is NOT necessarily an exact
        # integer (e.g. 30000.3) -- legacy's own `(st * AP_sr).astype(int)`
        # truncates, which can differ from rounding by one sample once the
        # spike time has round-tripped through a division by that same
        # non-integer AP_sr. See build_modulation_frame's docstring.
        "truncate_spike_samples": True,
        # legacy's `periods[prev:curr] = i` slice loop puts a sample exactly
        # AT crossing `prev` into bucket i (inclusive of the lower bound),
        # whereas the shared default counts crossings strictly less than
        # the sample -- only differs at exact-integer collisions, which
        # `truncate_spike_samples` makes non-negligible here. See
        # compute_period_at_samples's docstring.
        "period_crossing_inclusive": True,
    }]
    nwb_io.write_epochs_table(nwbfile, epochs, sampling_rate=AP_sr)
    nwb_io.write_nwbfile(nwbfile, cfg.nwb_path())

    from pathlib import Path
    from pipeline.diagnostics.processing import plot_recording_timeline
    diag_dir = Path(cfg.data_dir).parent / "figs" / "processing"
    diag_dir.mkdir(parents=True, exist_ok=True)
    plot_recording_timeline(cfg, None, modulation_df, diag_dir)
