"""Analysis stage for Engert GCaMP experiments.

Loads Suite2p outputs (sliced to the specific tiff via len_df), runs fit_Fourier
at 1F and 2F (skipping 2F when above Nyquist), builds fourier_df, saves pickle,
and generates diagnostic PDF.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from magpyneto2.engert_helpers import fit_Fourier, remove_flatlines, get_len_df
from magpyneto2.statistics import (
    get_epsilon,
    normalized_Fourier_PDF,
    normalized_Fourier_PDF_corrected,
    normalized_Fourier_CDF_corrected,
)


def _load_suite2p_sliced(session_path, tiff_name, iscell_thres, npix_thres):
    """Load F, spks, stat from suite2p/plane0/, sliced to frames for tiff_name."""
    suite2p_dir = os.path.join(session_path, "suite2p", "plane0")

    F      = np.load(os.path.join(suite2p_dir, "F.npy"),      allow_pickle=True)
    spks   = np.load(os.path.join(suite2p_dir, "spks.npy"),   allow_pickle=True)
    stat   = np.load(os.path.join(suite2p_dir, "stat.npy"),   allow_pickle=True)
    iscell = np.load(os.path.join(suite2p_dir, "iscell.npy"), allow_pickle=True)

    mask = (iscell[:, 1] > iscell_thres) & np.array([s["npix"] > npix_thres for s in stat])
    F, spks, stat = F[mask], spks[mask], stat[mask]

    if tiff_name:
        len_df = get_len_df(session_path)
        tiff_full = os.path.join(session_path, tiff_name)
        row = len_df.loc[len_df["path"] == tiff_full]
        if len(row) == 0:
            raise ValueError(
                f"tiff_name '{tiff_name}' not found in len_df for {session_path}.\n"
                f"Available: {list(len_df['path'])}"
            )
        start, end = int(row["start"].values[0]), int(row["end"].values[0])
        F, spks = F[:, start:end], spks[:, start:end]

    return F, spks, stat


def _corrected_pvalues(rr, Q):
    eps = get_epsilon(Q)
    R, YY = normalized_Fourier_PDF()
    PDF = normalized_Fourier_PDF_corrected(R[1:], R[1:], YY[1:], eps)
    CDF = normalized_Fourier_CDF_corrected(PDF, R[1:])
    return 1 - np.interp(rr, R[2:], CDF)


def run_analysis(cfg):
    freq   = cfg.analysis.f
    Q      = cfg.analysis.Q
    T      = cfg.sample_period
    nyquist = 0.5 / T

    # ── Load ────────────────────────────────────────────────────────────────
    print(f"[engert] {cfg.name}: loading Suite2p from {cfg.session_path}"
          + (f"  [{cfg.tiff_name}]" if cfg.tiff_name else "  [all frames]"))
    F, spks, stat = _load_suite2p_sliced(
        cfg.session_path, cfg.tiff_name,
        cfg.iscell_threshold, cfg.npix_threshold)
    print(f"[engert] {cfg.name}: {len(F)} cells after iscell filter")

    F, spks, stat = remove_flatlines(F, spks, stat)
    print(f"[engert] {cfg.name}: {len(F)} cells after flatline removal")

    # ── Fourier 1F ──────────────────────────────────────────────────────────
    print(f"[engert] {cfg.name}: fit_Fourier at {freq} Hz  (T={T})")
    chat_l, onfreq_pow_l, offfreq_pow_l, freq_win = fit_Fourier(
        F, T=T, f=freq, Q=Q)

    rr        = np.array(chat_l)
    pp        = _corrected_pvalues(rr, Q)
    fou_alt   = np.array(offfreq_pow_l)                              # (C, 2Q-1) complex
    sigma_1f  = np.sqrt(0.5 * np.mean(np.abs(fou_alt) ** 2, axis=1))
    sens      = np.abs(onfreq_pow_l) / np.where(sigma_1f > 0, sigma_1f, np.nan)

    # ── Fourier 2F (skip if above Nyquist) ──────────────────────────────────
    do_2f = (2 * freq) < nyquist
    if do_2f:
        print(f"[engert] {cfg.name}: fit_Fourier at {freq*2} Hz (2F)")
        chat_2f_l, onfreq_2f_l, offfreq_2f_l, _ = fit_Fourier(
            F, T=T, f=freq * 2, Q=Q)
        rr_2f      = np.array(chat_2f_l)
        pp_2f      = _corrected_pvalues(rr_2f, Q)
        fou_alt_2f = np.array(offfreq_2f_l)
        sigma_2f   = np.sqrt(0.5 * np.mean(np.abs(fou_alt_2f) ** 2, axis=1))
        sens_2f    = np.abs(onfreq_2f_l) / np.where(sigma_2f > 0, sigma_2f, np.nan)
    else:
        print(f"[engert] {cfg.name}: skipping 2F ({freq*2:.3f} Hz >= Nyquist {nyquist:.3f} Hz)")
        rr_2f = np.full(len(rr), np.nan)
        pp_2f = np.full(len(rr), np.nan)
        sens_2f = np.full(len(rr), np.nan)

    # ── Number of frames used by fit_Fourier ────────────────────────────────
    N_frames = int(120 * (F.shape[1] // 60))
    nn = np.full(len(rr), N_frames)

    # ── Build fourier_df ─────────────────────────────────────────────────────
    fourier_df = pd.DataFrame({
        "id":      np.arange(len(rr)),
        "pp":      pp,
        "nn":      nn,
        "rr":      rr,
        "freq":    freq,
        "rec":     cfg.name,
        "2f_rr":   rr_2f,
        "2f_pp":   pp_2f,
        "sens":    sens,
        "sens_2f": sens_2f,
        "Q":       Q,
    })

    os.makedirs(cfg.data_dir, exist_ok=True)
    with open(cfg.analysis_path(), "wb") as fh:
        pickle.dump(fourier_df, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[engert] {cfg.name}: saved -> {cfg.analysis_path()}")

    # ── Diagnostics ──────────────────────────────────────────────────────────
    from pipeline.diagnostics.engert import plot_engert_diagnostics
    diag_dir = Path(cfg.data_dir).parent / "figs" / "analysis"
    diag_dir.mkdir(parents=True, exist_ok=True)
    plot_engert_diagnostics(
        cfg, F, stat, fourier_df, freq_win,
        onfreq_pow_l, offfreq_pow_l, diag_dir)
