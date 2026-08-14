"""Medaka GCaMP paradigm — processing stage.

Reads Suite2p outputs (ALL ROIs, unfiltered) once and writes them to this
experiment's NWB file: PlaneSegmentation + a RoiResponseSeries holding the
full fluorescence trace (no tiff slicing — each medaka session_path is its
own independent trial directory, unlike engert's shared-session tiffs).
"""
import os

import numpy as np

from pipeline import nwb_io


def run_processing(cfg):
    suite2p_dir = os.path.normpath(os.path.join(cfg.session_path, "suite2p", "plane0"))
    F      = np.load(os.path.join(suite2p_dir, "F.npy"),      allow_pickle=True)
    stat   = np.load(os.path.join(suite2p_dir, "stat.npy"),   allow_pickle=True)
    iscell = np.load(os.path.join(suite2p_dir, "iscell.npy"), allow_pickle=True)
    ops    = np.load(os.path.join(suite2p_dir, "ops.npy"),    allow_pickle=True).item()

    nwbfile = nwb_io.create_nwbfile(cfg)
    ps = nwb_io.write_imaging_plane_and_rois(
        nwbfile, stat, iscell, ops, sampling_rate=1.0 / cfg.sample_period)
    nwb_io.write_roi_response_series(
        nwbfile, F, ps, sampling_rate=1.0 / cfg.sample_period)
    nwb_io.write_nwbfile(nwbfile, cfg.nwb_path())
    print(f"[medaka] {cfg.name}: wrote {len(stat)} ROIs + "
          f"{F.shape[1]}-frame trace -> {cfg.nwb_path()}")
