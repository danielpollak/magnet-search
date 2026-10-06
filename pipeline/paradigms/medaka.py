"""Medaka GCaMP paradigm -- processing stage.

Writes this experiment's NWB file from its suite2p v1 segmentation (cfg.suite2p_path, local disk):
PlaneSegmentation (ALL suite2p ROIs) and the same two RoiResponseSeries as engert
(full-resolution traces, analysed, and suite2p's F.npy, provenance; see
pipeline/paradigms/engert.py and pipeline/ophys_extraction.py). No tiff slicing: each medaka
trial was segmented on its own and has one tiff.
"""
import os

import numpy as np

from pipeline import nwb_io, ophys_extraction


def run_processing(cfg):
    suite2p_dir = os.path.normpath(cfg.suite2p_path)
    stat   = np.load(os.path.join(suite2p_dir, "stat.npy"),   allow_pickle=True)
    iscell = np.load(os.path.join(suite2p_dir, "iscell.npy"), allow_pickle=True)
    ops    = ophys_extraction.load_ops(suite2p_dir)

    print(f"[medaka] {cfg.name}: full-resolution traces")
    tr = ophys_extraction.trial_traces(suite2p_dir)

    nwbfile = nwb_io.create_nwbfile(cfg)
    ps = nwb_io.write_imaging_plane_and_rois(
        nwbfile, stat, iscell, ops, sampling_rate=1.0 / cfg.sample_period)
    nwb_io.write_roi_response_series(
        nwbfile, tr["F_full"], ps, sampling_rate=1.0 / cfg.sample_period,
        F_suite2p=tr["F_suite2p"], repro_err=tr["max_err"])
    nwb_io.write_mean_image(nwbfile, ops["meanImg"])   # for the fish-outline diagnostics
    nwb_io.write_nwbfile(nwbfile, cfg.nwb_path())
    print(f"[medaka] {cfg.name}: wrote {len(stat)} ROIs + "
          f"{tr['F_full'].shape[1]}-frame traces -> {cfg.nwb_path()}")
