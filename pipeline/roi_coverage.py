"""Activity coverage of a fluorescence trace: in how much of the recording is the ROI active?

Many suite2p traces in the floor-clipped recordings (0.3 Hz and 0.1 Hz zebrafish, medaka) sit
at one floor value for most of the recording. Per trace:

  floor             the trace's most common value, if it holds >= FLOOR_SHARE of frames
                    (a clipped trace); otherwise the trace has no floor. The minimum is NOT
                    used: a single outlier frame below the floor would make every floor frame
                    count as active.
  active fraction   share of frames above the floor (1 for an unclipped trace)
  coverage          share of WIN_S-second windows containing >= MIN_ACTIVE active frames

The engert/medaka analysis stages keep ROIs with coverage >= the YAML's `coverage_threshold`
(0 = no filter). Nothing here uses stimulus timing or power at any frequency, so the filter
cannot create or remove a stimulus-locked response. Background and calibration tests:
docs/nfc_finite_sample_bias/activity_coverage.py.
"""
import numpy as np

FLOOR_SHARE = 0.2
WIN_S = 60.0
MIN_ACTIVE = 3


def activity(F, T):
    """dict of per-trace arrays: clipped (has a floor), active_frac, coverage. F is
    (n_rois, n_frames); T is the sample period in seconds."""
    N = F.shape[1]
    floor = np.full(len(F), -np.inf)
    for i, row in enumerate(F):
        v, c = np.unique(row, return_counts=True)
        if c.max() >= FLOOR_SHARE * N:
            floor[i] = v[c.argmax()]
    active = F > floor[:, None]
    w = int(round(WIN_S / T))
    nw = N // w
    A = active[:, :nw * w].reshape(len(F), nw, w)
    return dict(clipped=np.isfinite(floor), active_frac=active.mean(1),
                coverage=(A.sum(2) >= MIN_ACTIVE).mean(1))


def low_coverage_traces(cfg):
    """(F, coverage) of the traces the coverage threshold drops: those passing the iscell/npix
    thresholds and the fish outline, not flatlines, with coverage < cfg.coverage_threshold.
    For the diagnostics' heatmap of kept vs dropped traces; empty if there is no threshold."""
    from magpyneto2.engert_helpers import remove_flatlines
    from pipeline import body_outline, nwb_io
    io, nwbfile = nwb_io.read_nwbfile(cfg.nwb_path())
    F, roi_df = nwb_io.read_roi_data(nwbfile)
    mean_img = nwb_io.read_mean_image(nwbfile)
    io.close()
    empty = (np.zeros((0, F.shape[1])), np.zeros(0))
    if cfg.coverage_threshold <= 0:
        return empty
    passing = ((roi_df["p_iscell"].values > cfg.iscell_threshold)
               & (roi_df["npix"].values > cfg.npix_threshold)
               & body_outline.inside_rois(roi_df, mean_img,
                                          body_outline.params_for(cfg.body_outline))[0])
    cov = activity(F[passing].astype(float), cfg.sample_period)["coverage"]
    low = cov < cfg.coverage_threshold
    if not low.any():
        return empty
    F_low, cov_low = F[passing][low], cov[low]
    keep = remove_flatlines(F_low)[3]
    return F_low[keep], cov_low[keep]
