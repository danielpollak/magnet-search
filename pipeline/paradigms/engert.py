"""Engert GCaMP paradigm — processing stage.

Reads Suite2p outputs (ALL ROIs, unfiltered) once and writes them to this
experiment's NWB file: PlaneSegmentation (segmentation masks + p_iscell/npix
columns so iscell_threshold/npix_threshold filtering stays deferred to
analysis time, matching CLAUDE.md's documented tutorial workflow) and a
RoiResponseSeries holding this experiment's own tiff-sliced fluorescence
trace (multiple engert experiments can share one session_path's suite2p
output, distinguished only by tiff_name -- see get_len_df).
"""
import os

import numpy as np

from magpyneto2.engert_helpers import get_len_df
from pipeline import nwb_io


def run_processing(cfg):
    suite2p_dir = os.path.normpath(os.path.join(cfg.session_path, "suite2p", "plane0"))
    F      = np.load(os.path.join(suite2p_dir, "F.npy"),      allow_pickle=True)
    stat   = np.load(os.path.join(suite2p_dir, "stat.npy"),   allow_pickle=True)
    iscell = np.load(os.path.join(suite2p_dir, "iscell.npy"), allow_pickle=True)
    ops    = np.load(os.path.join(suite2p_dir, "ops.npy"),    allow_pickle=True).item()

    if cfg.tiff_name:
        len_df = get_len_df(cfg.session_path)
        tiff_full = os.path.join(cfg.session_path, cfg.tiff_name)
        row = len_df.loc[len_df["path"] == tiff_full]
        if len(row) == 0:
            raise ValueError(
                f"tiff_name '{cfg.tiff_name}' not found in len_df for {cfg.session_path}.\n"
                f"Available: {list(len_df['path'])}"
            )
        start, end = int(row["start"].values[0]), int(row["end"].values[0])
        F_slice = F[:, start:end]
        print(f"[engert] {cfg.name}: slicing frames [{start}:{end}] for tiff {cfg.tiff_name}")
    else:
        F_slice = F

    nwbfile = nwb_io.create_nwbfile(cfg)
    ps = nwb_io.write_imaging_plane_and_rois(
        nwbfile, stat, iscell, ops, sampling_rate=1.0 / cfg.sample_period)
    nwb_io.write_roi_response_series(
        nwbfile, F_slice, ps, sampling_rate=1.0 / cfg.sample_period)
    nwb_io.write_nwbfile(nwbfile, cfg.nwb_path())
    print(f"[engert] {cfg.name}: wrote {len(stat)} ROIs + "
          f"{F_slice.shape[1]}-frame trace -> {cfg.nwb_path()}")
