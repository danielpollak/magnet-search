"""Engert GCaMP paradigm -- processing stage.

Writes this experiment's NWB file from the session's suite2p v1 segmentation (cfg.suite2p_path,
local disk): PlaneSegmentation (ALL suite2p ROIs, with p_iscell/npix columns so
iscell_threshold/npix_threshold filtering stays deferred to analysis time) and two
RoiResponseSeries over this experiment's own tiff (several engert experiments share one
session's suite2p output, distinguished only by tiff_name):
  RoiResponseSeries           full-resolution traces (F_full.npy, re-extracted from the raw
                              tiff with suite2p's registration and ROI weights, minus suite2p's
                              lossy int16 conversions; pipeline/ophys_extraction.py) -- analysed
  RoiResponseSeries_suite2p   suite2p's own F.npy over the same frames (provenance)
The tiff's frames within suite2p's concatenated movie come from suite2p's own file list
(ops file_list / frames_per_file).
"""
import os

import numpy as np

from pipeline import nwb_io, ophys_extraction


def run_processing(cfg):
    suite2p_dir = os.path.normpath(cfg.suite2p_path)
    stat   = np.load(os.path.join(suite2p_dir, "stat.npy"),   allow_pickle=True)
    iscell = np.load(os.path.join(suite2p_dir, "iscell.npy"), allow_pickle=True)
    ops    = ophys_extraction.load_ops(suite2p_dir)

    print(f"[engert] {cfg.name}: full-resolution traces of {cfg.tiff_name or 'all tiffs'}")
    tr = ophys_extraction.trial_traces(suite2p_dir, cfg.tiff_name or None)

    nwbfile = nwb_io.create_nwbfile(cfg)
    ps = nwb_io.write_imaging_plane_and_rois(
        nwbfile, stat, iscell, ops, sampling_rate=1.0 / cfg.sample_period)
    nwb_io.write_roi_response_series(
        nwbfile, tr["F_full"], ps, sampling_rate=1.0 / cfg.sample_period,
        F_suite2p=tr["F_suite2p"], repro_err=tr["max_err"])
    nwb_io.write_mean_image(nwbfile, ops["meanImg"])   # for the fish-outline diagnostics
    nwb_io.write_nwbfile(nwbfile, cfg.nwb_path())
    print(f"[engert] {cfg.name}: wrote {len(stat)} ROIs + "
          f"{tr['F_full'].shape[1]}-frame traces -> {cfg.nwb_path()}")
