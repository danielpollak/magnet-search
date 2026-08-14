"""Analysis stage for medaka GCaMP experiments.

Reads Suite2p F/ROI data back from this experiment's NWB file (written by
the processing stage — see pipeline/paradigms/medaka.py), no tiff slicing
needed (each session_path is its own independent trial directory). Two
Fourier frequencies are computed per trial:
  - cfg.analysis.f  (magnetic, e.g. 0.1 Hz)
  - 1/60 Hz         (visual stimulus, fixed for medaka)
"""
import os

import numpy as np
import pandas as pd

from magpyneto2.engert_helpers import fit_Fourier, remove_flatlines
from magpyneto2.statistics import corrected_pvalues
from pipeline import nwb_io

_VISUAL_FREQ = 1 / 60
_VISUAL_Q = 6


def _load_from_nwb(nwb_path, iscell_thres, npix_thres):
    """Same contract as engert's _load_from_nwb — see that module's
    docstring for the included_mask semantics."""
    if not os.path.exists(nwb_path):
        raise FileNotFoundError(
            f"{nwb_path} not found -- run `python pipeline/processing.py "
            f"--experiment <name>` first (medaka now has a real processing "
            f"stage; it's no longer a no-op).")

    io_r, nwbfile = nwb_io.read_nwbfile(nwb_path)
    F_all, roi_df = nwb_io.read_roi_data(nwbfile)
    Ly, Lx = nwb_io.get_imaging_dims(nwbfile)
    io_r.close()

    mask = (roi_df["p_iscell"].values > iscell_thres) & (roi_df["npix"].values > npix_thres)
    F_masked = F_all[mask]

    F_final, _, _, inclusion_inds = remove_flatlines(F_masked)

    included_mask = np.zeros(len(roi_df), dtype=bool)
    masked_positions = np.where(mask)[0]
    included_mask[masked_positions[inclusion_inds]] = True

    return F_final, roi_df, included_mask, (Ly, Lx)


def compute_fourier_results(cfg, verbose=True):
    """Pure computation, no I/O beyond the NWB read -- see engert.py's
    compute_fourier_results docstring for why this is factored out (lets
    verify_outputs.py independently recompute and diff against what got
    persisted, without depending on any legacy pickle)."""
    f_b = cfg.analysis.f
    Q_b = cfg.analysis.Q
    T   = cfg.sample_period

    def _p(msg):
        if verbose:
            print(msg)

    _p(f"[medaka] {cfg.name}: loading Suite2p from {cfg.nwb_path()}")
    F, roi_df, included_mask, imaging_dims = _load_from_nwb(
        cfg.nwb_path(), cfg.iscell_threshold, cfg.npix_threshold)
    _p(f"[medaka] {cfg.name}: {len(F)} cells after iscell/npix filter + flatline removal")

    # Magnetic frequency (keep intermediates for diagnostics)
    chat_b, onb, offb, freq_win_b = fit_Fourier(F, T=T, f=f_b, Q=Q_b)
    # Visual frequency
    chat_v, onv, offv, freq_win_v = fit_Fourier(F, T=T, f=_VISUAL_FREQ, Q=_VISUAL_Q)

    contingency = "positive control" if "no_magneto" in cfg.name else "mag"
    nn = int(120 * (F.shape[1] // 60))

    # Use basename + ".tif" so get_poscontrols_negresults() rec-name patterns match:
    #   magneto_0.tif → mag experiment
    #   magneto_1.tif / magneto_2.tif → fish positive control (visual freq)
    rec_name = os.path.basename(cfg.session_path.rstrip("/\\")) + ".tif"

    rows = []
    for chat_l, freq, Q in [(chat_b, f_b, Q_b), (chat_v, _VISUAL_FREQ, _VISUAL_Q)]:
        rr = np.array(chat_l)
        rows.append(pd.DataFrame({
            "id":          np.arange(len(chat_l)),
            "pp":          corrected_pvalues(rr, Q),
            "nn":          nn,
            "rr":          rr,
            "freq":        freq,
            "rec":         rec_name,
            "Q":           Q,
            "date":        cfg.date,
            "area":        "wholebrain",
            "ID":          cfg.subject_id,
            "species":     "medaka",
            "contingency": contingency,
        }))
    fourier_df = pd.concat(rows, ignore_index=True)

    return {
        "F": F, "roi_df": roi_df, "included_mask": included_mask,
        "imaging_dims": imaging_dims, "fourier_df": fourier_df,
        "f_b": f_b, "Q_b": Q_b, "T": T, "rec_name": rec_name,
        "freq_win_b": freq_win_b, "onb": onb, "offb": offb,
        "freq_win_v": freq_win_v, "onv": onv, "offv": offv,
    }


def run_analysis(cfg):
    r = compute_fourier_results(cfg)
    F, roi_df, included_mask, imaging_dims = r["F"], r["roi_df"], r["included_mask"], r["imaging_dims"]
    fourier_df, f_b, Q_b, T, rec_name = r["fourier_df"], r["f_b"], r["Q_b"], r["T"], r["rec_name"]
    freq_win_b, onb, offb = r["freq_win_b"], r["onb"], r["offb"]
    freq_win_v, onv, offv = r["freq_win_v"], r["onv"], r["offv"]

    # ── NWB write (see .claude/plans — NWB replatform, Phase 4) ────────────
    # Magnetic and visual are two INDEPENDENT groups (not 1F/2F harmonics of
    # each other), so each gets its own write_imaging_fourier_results call —
    # no onfreq_pow_2f/offfreq_pow_2f pairing, matching medaka's own
    # from-scratch (not paired-row) fourier_df construction above. The 5
    # extra medaka-only columns (date/area/ID/species/contingency) are NOT
    # persisted in the shared table (they're per-cfg constants, not Fourier
    # intermediates) -- callers needing them re-derive from cfg, same as
    # aggregate.py's ANNOT_D fallback already does for the legacy pickle.
    nwb_io.append_results(
        cfg.nwb_path(),
        lambda nwbfile: [
            nwb_io.write_imaging_fourier_results(
                nwbfile, rec=rec_name, freq=f_b, Q=Q_b,
                T_duration=F.shape[1] * T,
                fourier_df_rows=fourier_df.loc[fourier_df.freq == f_b],
                onfreq_pow=onb, offfreq_pow=offb, freq_win=freq_win_b),
            nwb_io.write_imaging_fourier_results(
                nwbfile, rec=rec_name, freq=_VISUAL_FREQ, Q=_VISUAL_Q,
                T_duration=F.shape[1] * T,
                fourier_df_rows=fourier_df.loc[fourier_df.freq == _VISUAL_FREQ],
                onfreq_pow=onv, offfreq_pow=offv, freq_win=freq_win_v),
        ],
    )
    print(f"[medaka] {cfg.name}: saved -> {cfg.nwb_path()}  ({len(fourier_df)} rows)")

    # Diagnostics — magnetic frequency only (same format as engert)
    from pathlib import Path
    from pipeline.diagnostics.engert import plot_engert_diagnostics
    diag_dir = Path(cfg.data_dir).parent / "figs" / "analysis"
    diag_dir.mkdir(parents=True, exist_ok=True)
    fourier_df_b = fourier_df.loc[fourier_df.freq == f_b].reset_index(drop=True)
    plot_engert_diagnostics(
        cfg, F, fourier_df_b, freq_win_b,
        onb, offb, diag_dir,
        roi_df=roi_df, included_mask=included_mask, imaging_dims=imaging_dims)
