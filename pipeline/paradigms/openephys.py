"""
Paradigm: openephys

Pure-magnetic OpenEphys recordings.
Loads spike times from Kilosort via Reader, builds contingency_d from
metadata CSV, calls process_raw_data_NPIX, writes Units/epochs to
`{name}.nwb` (see pipeline/nwb_io.py), and saves MM_d diagnostics.
"""
import os
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import tqdm.auto as tqdm

from ephysio import openEphysIO
from ephysio.kilosortIO import Reader
from magpyneto2 import (
    get_cluster_info, process_raw_data_NPIX, update_MM_d_mag,
    save_diagnostics_MM,
)
from magpyneto2.utils import get_MM_offset
from pipeline import nwb_io

AP_SR = 30_000


def run_processing(cfg):
    data_path = cfg.aggregated_path
    label = "good" if cfg.good else None

    cat_df = pd.read_csv(cfg.metadata_csv)
    cat_df["cumulate"] = np.cumsum(cat_df["nframes"].values)

    ksr = Reader(data_path)
    all_sts = ksr.spikesbycluster(label=label)

    udf = get_cluster_info(data_path)
    if cfg.good:
        udf = udf.loc[udf.KSLabel == "good", :]

    # NWB Units table gets these SAME cfg.good-filtered spike trains --
    # only `good` units are written by default. To recover MUA later, set
    # `good: False` in the YAML and reprocess (label becomes None above, so
    # all_sts/udf include everything) -- see .claude/plans, NWB replatform
    # "good-only" cutover. (Previously this always wrote every cluster
    # regardless of cfg.good, deferring the good/mua split to analysis time
    # via build_modulation_frame's good_only filter -- confirmed unused:
    # no experiment YAML has ever set good: False, so MUA was read,
    # decompressed, and loaded on every analysis run only to be discarded.)
    all_sts_nwb = all_sts
    udf_nwb = udf

    # per-row stream_ids: cfg.streams overrides cfg.stream_id when provided
    row_streams = cfg.streams if cfg.streams else [cfg.stream_id] * len(cat_df)

    contingency_d = {}
    aux_d = {}

    for catrow_i, catrow in tqdm.tqdm(cat_df.iterrows(), total=len(cat_df)):
        stream = row_streams[catrow_i] if catrow_i < len(row_streams) else cfg.stream_id
        if stream is None:
            continue  # this row is intentionally skipped (e.g. 20220408)

        spath = catrow.path.split(catrow.recname)
        recroot_path = spath[0] + catrow.recname

        recpath = "\\".join(
            [r"\\datanas\family\data_raw"] + recroot_path.split("\\")[-2:]
        ).replace("\\", "/")

        # Allow per-recording path override (e.g. 20220621 Taeniopygia)
        if catrow.recname in cfg.recording_overrides:
            rec_load_path = cfg.recording_overrides[catrow.recname].replace("\\", "/")
        else:
            rec_load_path = recpath

        try:
            recording = se.OpenEphysBinaryRecordingExtractor(
                rec_load_path, stream_id=stream)
        except Exception:
            print(f"  WARNING: could not load {rec_load_path}, skipping")
            continue

        ldr = openEphysIO.Loader(recpath, cntlbarcodes=cfg.recording_ldr_cntlbarcodes)

        beginning_time = 0 if catrow_i == 0 else cat_df.iloc[catrow_i - 1]["cumulate"]

        aux_d[catrow.recname] = (recording, ldr)
        contingency_d[catrow.recname] = {
            cr.cluster_id: all_sts[cr.cluster_id][
                (all_sts[cr.cluster_id] > beginning_time) &
                (all_sts[cr.cluster_id] < catrow.cumulate)
            ] - beginning_time
            for _, cr in udf.iterrows()
        }

    folder_locations_freq_skips = []
    for trial in cfg.trials:
        recname = trial.recname
        if recname not in contingency_d:
            raise KeyError(
                f"Trial recname '{recname}' not in contingency_d. "
                f"Available: {list(contingency_d.keys())}"
            )
        folder_locations_freq_skips.append((
            trial.folder,
            trial.frequency,
            trial.skips,
            contingency_d[recname],
            *aux_d[recname],
        ))

    modulation_df, data, _ = process_raw_data_NPIX(
        folder_locations_freq_skips, THRES=cfg.threshold)

    MM_d = {"spikes": all_sts, "aux": {}}
    MM_d = update_MM_d_mag(MM_d, data, cat_df)
    save_diagnostics_MM(MM_d, cfg.name)

    # --- NWB write (see .claude/plans — NWB replatform; Phase 7 cutover: the
    # legacy modulation_df/MM_d pickles this paradigm used to also write are
    # retired now that verify_outputs.py confirms NWB parity -- see
    # cfg.processing_path()'s docstring for the frozen-fixture history this
    # replaces). MM_d/modulation_df above stay purely in-memory: they're
    # still needed as inputs to build the epochs/Units below and for
    # save_diagnostics_MM's plot.
    nwbfile = nwb_io.create_nwbfile(cfg)
    # label_column="KSLabel": matches this paradigm's own good-filter
    # (`udf.KSLabel == "good"` above) -- see write_units_and_spikes's
    # label_column docstring for why this must be stated explicitly rather
    # than auto-detected.
    nwb_io.write_units_and_spikes(
        nwbfile, all_sts_nwb, udf_nwb, sampling_rate=AP_SR, label_column="KSLabel")

    epochs = []
    for freq in data.keys():
        for folder in data[freq].keys():
            recname = os.path.basename(folder)
            offset = get_MM_offset(cat_df, recname)
            _, period_crossings = data[freq][folder]
            full_crossings = (np.asarray(period_crossings) + offset).astype(np.int64)
            trial_skips = next((t.skips for t in cfg.trials if t.recname == recname), 0)
            epochs.append({
                "rec": recname,
                "stim_type": "magnetic",
                "frequency": float(freq),
                "start_time": float(full_crossings[0]) / AP_SR,
                "stop_time": float(full_crossings[-1]) / AP_SR,
                "period_crossings": full_crossings,
                "skips": trial_skips,
                # Units.spike_times (all_sts_nwb) are in the aggregated
                # multi-recording catalog domain, same as full_crossings
                # above (both offset by `offset`) -- but legacy
                # modulation_df.spk is recording-local (all_sts - offset).
                # Persist `offset` so build_modulation_frame can undo it.
                "local_offset_samples": int(offset),
            })
    nwb_io.write_epochs_table(nwbfile, epochs, sampling_rate=AP_SR)
    nwb_io.write_nwbfile(nwbfile, cfg.nwb_path())

    from pathlib import Path
    from pipeline.diagnostics.processing import plot_recording_timeline
    diag_dir = Path(cfg.data_dir).parent / "figs" / "processing"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Read diagnostics input back from the just-written NWB file rather than
    # the in-memory MM_d/modulation_df above — proves read-back fidelity
    # ("traceability without recomputation"). Falls back to the in-memory
    # objects if anything about the NWB round-trip goes wrong, so a
    # diagnostics-plotting bug can never block the processing stage itself.
    try:
        io_r, nwbfile_r = nwb_io.read_nwbfile(cfg.nwb_path())
        MM_d_nwb = nwb_io.read_mm_d_equivalent(nwbfile_r)
        modulation_df_nwb = nwb_io.build_modulation_frame(nwbfile_r, good_only=cfg.good)
        plot_recording_timeline(cfg, MM_d_nwb, modulation_df_nwb, diag_dir)
        io_r.close()
    except Exception as exc:
        print(f"  WARNING: NWB-sourced diagnostics failed ({exc}); "
              f"falling back to in-memory MM_d/modulation_df")
        plot_recording_timeline(cfg, MM_d, modulation_df, diag_dir)
