"""
Paradigm: openephys

Pure-magnetic OpenEphys recordings.
Loads spike times from Kilosort via Reader, builds contingency_d from
metadata CSV, calls process_raw_data_NPIX, saves modulation_df and MM_d diagnostics.
"""
import os
import pickle
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
    save_MM_d_pickle, save_diagnostics_MM,
)


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

    with open(cfg.processing_path(), "wb") as f:
        pickle.dump(modulation_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    mm_path = os.path.join(cfg.data_dir, f"MM_{cfg.name}.pickle")
    save_MM_d_pickle(MM_d, mm_path)
    save_diagnostics_MM(MM_d, cfg.name)

    from pathlib import Path
    from pipeline.diagnostics.processing import plot_recording_timeline
    diag_dir = Path(cfg.data_dir).parent / "figs" / "processing"
    diag_dir.mkdir(parents=True, exist_ok=True)
    plot_recording_timeline(cfg, MM_d, modulation_df, diag_dir)
