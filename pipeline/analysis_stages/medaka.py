"""Analysis stage for medaka GCaMP experiments.

Each session_path is an individual trial directory containing its own suite2p/plane0/
outputs — no tiff slicing needed. Two Fourier frequencies are computed per trial:
  - cfg.analysis.f  (magnetic, e.g. 0.1 Hz)
  - 1/60 Hz         (visual stimulus, fixed for medaka)
"""
import os
import pickle

import numpy as np
import pandas as pd

from magpyneto2.engert_helpers import fit_Fourier, remove_flatlines

_VISUAL_FREQ = 1 / 60
_VISUAL_Q = 6


def _load_suite2p(session_path, iscell_thres, npix_thres):
    suite2p_dir = os.path.join(session_path, "suite2p", "plane0")

    F      = np.load(os.path.join(suite2p_dir, "F.npy"),      allow_pickle=True)
    spks   = np.load(os.path.join(suite2p_dir, "spks.npy"),   allow_pickle=True)
    stat   = np.load(os.path.join(suite2p_dir, "stat.npy"),   allow_pickle=True)
    iscell = np.load(os.path.join(suite2p_dir, "iscell.npy"), allow_pickle=True)

    mask = (iscell[:, 1] > iscell_thres) & np.array([s["npix"] > npix_thres for s in stat])
    return F[mask], spks[mask], stat[mask]


def run_analysis(cfg):
    f_b = cfg.analysis.f
    Q_b = cfg.analysis.Q
    T   = cfg.sample_period

    print(f"[medaka] {cfg.name}: loading Suite2p from {cfg.session_path}")
    F, spks, stat = _load_suite2p(cfg.session_path, cfg.iscell_threshold, cfg.npix_threshold)
    print(f"[medaka] {cfg.name}: {len(F)} cells after iscell filter")

    F, spks, stat = remove_flatlines(F, spks, stat)
    print(f"[medaka] {cfg.name}: {len(F)} cells after flatline removal")

    # Magnetic frequency (keep intermediates for diagnostics)
    chat_b, onb, offb, freq_win_b = fit_Fourier(F, T=T, f=f_b, Q=Q_b)
    # Visual frequency
    chat_v, _, _, _ = fit_Fourier(F, T=T, f=_VISUAL_FREQ, Q=_VISUAL_Q)

    contingency = "positive control" if "no_magneto" in cfg.name else "mag"
    nn = int(120 * (F.shape[1] // 60))

    # Use basename + ".tif" so get_poscontrols_negresults() rec-name patterns match:
    #   magneto_0.tif → mag experiment
    #   magneto_1.tif / magneto_2.tif → fish positive control (visual freq)
    rec_name = os.path.basename(cfg.session_path.rstrip("/\\")) + ".tif"

    rows = []
    for chat_l, freq in [(chat_b, f_b), (chat_v, _VISUAL_FREQ)]:
        rows.append(pd.DataFrame({
            "id":          np.arange(len(chat_l)),
            "pp":          np.nan,
            "nn":          nn,
            "rr":          np.array(chat_l),
            "freq":        freq,
            "rec":         rec_name,
            "date":        cfg.date,
            "area":        "wholebrain",
            "ID":          cfg.subject_id,
            "species":     "medaka",
            "contingency": contingency,
        }))
    fourier_df = pd.concat(rows, ignore_index=True)

    os.makedirs(cfg.data_dir, exist_ok=True)
    with open(cfg.analysis_path(), "wb") as fh:
        pickle.dump(fourier_df, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[medaka] {cfg.name}: saved → {cfg.analysis_path()}  ({len(fourier_df)} rows)")

    # Diagnostics — magnetic frequency only (same format as engert)
    from pathlib import Path
    from pipeline.diagnostics.engert import plot_engert_diagnostics
    diag_dir = Path(cfg.data_dir).parent / "figs" / "analysis"
    diag_dir.mkdir(parents=True, exist_ok=True)
    fourier_df_b = fourier_df.loc[fourier_df.freq == f_b].reset_index(drop=True)
    plot_engert_diagnostics(
        cfg, F, stat, fourier_df_b, freq_win_b,
        onb, offb, diag_dir)
