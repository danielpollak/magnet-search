"""Medaka GCaMP paradigm -- processing stage.

Writes this experiment's NWB file from its suite2p v1 segmentation (cfg.suite2p_path, local disk):
PlaneSegmentation (ALL suite2p ROIs) and the same two RoiResponseSeries as engert
(full-resolution traces, analysed, and suite2p's F.npy, provenance; see
pipeline/paradigms/engert.py and pipeline/ophys_extraction.py). Since 2026-10-07 the six medaka
trials share one segmentation (suite2p over all six tiffs), so, as for engert, cfg.tiff_name
picks this trial's frames and the ROIs are the same cells in every trial.
"""
import os

import numpy as np

from pipeline import nwb_io, ophys_extraction


def run_processing(cfg):
    suite2p_dir = os.path.normpath(cfg.suite2p_path)
    stat   = np.load(os.path.join(suite2p_dir, "stat.npy"),   allow_pickle=True)
    iscell = np.load(os.path.join(suite2p_dir, "iscell.npy"), allow_pickle=True)
    ops    = ophys_extraction.load_ops(suite2p_dir)

    print(f"[medaka] {cfg.name}: full-resolution traces of {cfg.tiff_name or 'all tiffs'}")
    tr = ophys_extraction.trial_traces(suite2p_dir, cfg.tiff_name or None)

    nwbfile = nwb_io.create_nwbfile(cfg)
    ps = nwb_io.write_imaging_plane_and_rois(
        nwbfile, stat, iscell, ops, sampling_rate=1.0 / cfg.sample_period)
    nwb_io.write_roi_response_series(
        nwbfile, tr["F_full"], ps, sampling_rate=1.0 / cfg.sample_period,
        F_suite2p=tr["F_suite2p"], repro_err=tr["max_err"])
    # for the fish outline: the full-resolution movie's mean (suite2p's meanImg is nearly flat
    # in the photon-starved sessions, and the automatic outline then covers the whole field)
    nwb_io.write_mean_image(nwbfile, ophys_extraction.mean_image(suite2p_dir))
    nwb_io.write_nwbfile(nwbfile, cfg.nwb_path())
    print(f"[medaka] {cfg.name}: wrote {len(stat)} ROIs + "
          f"{tr['F_full'].shape[1]}-frame traces -> {cfg.nwb_path()}")
