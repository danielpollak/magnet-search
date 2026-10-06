"""Medaka GCaMP paradigm -- processing stage.

Writes this experiment's NWB file from suite2p's segmentation and the raw tiff:
PlaneSegmentation (ALL suite2p ROIs) and the same two RoiResponseSeries as engert
(full-resolution traces, analysed, and suite2p's F.npy, provenance; see
pipeline/paradigms/engert.py and pipeline/ophys_extraction.py). No tiff slicing: each medaka
session_path is its own independent trial directory with one tiff.
"""
import os

import numpy as np

from pipeline import nwb_io, ophys_extraction


def run_processing(cfg):
    suite2p_dir = os.path.normpath(os.path.join(cfg.session_path, "suite2p", "plane0"))
    stat   = np.load(os.path.join(suite2p_dir, "stat.npy"),   allow_pickle=True)
    iscell = np.load(os.path.join(suite2p_dir, "iscell.npy"), allow_pickle=True)
    ops    = np.load(os.path.join(suite2p_dir, "ops.npy"),    allow_pickle=True).item()

    print(f"[medaka] {cfg.name}: extracting at full resolution")
    tr = ophys_extraction.extract(cfg.session_path, None,
                                  log=lambda m: print(f"[medaka] {cfg.name}:{m}"))

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
