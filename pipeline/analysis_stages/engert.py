"""Analysis stage for Engert GCaMP experiments.

Reads Suite2p F/ROI data back from this experiment's NWB file (written by
the processing stage — see pipeline/paradigms/engert.py), applies the
iscell_threshold/npix_threshold mask and flatline removal (unchanged
formulas, still deferred to analysis time so editing thresholds in the YAML
only requires re-running analysis, not reprocessing), runs fit_Fourier at 1F
and 2F (skipping 2F when above Nyquist), builds fourier_df, writes Fourier
results to NWB, and generates diagnostic PDF.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from magpyneto2.engert_helpers import fit_Fourier, remove_flatlines
from magpyneto2.statistics import corrected_pvalues
from pipeline import nwb_io


def _load_from_nwb(nwb_path, iscell_thres, npix_thres):
    """Load (F, roi_df, included_mask, imaging_dims) from the processing
    stage's NWB file, applying the iscell/npix mask (STRICT `>`, matching
    the pre-NWB `_load_suite2p_sliced`'s own convention) and flatline
    removal. `included_mask` (over the FULL, unfiltered roi_df) marks the
    exact population surviving both steps, for diagnostics."""
    if not os.path.exists(nwb_path):
        raise FileNotFoundError(
            f"{nwb_path} not found -- run `python pipeline/processing.py "
            f"--experiment <name>` first (engert now has a real processing "
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
    """Pure computation: load Suite2p data back from NWB, run 1F/2F
    fit_Fourier, build fourier_df. No I/O beyond the NWB read -- no writing,
    no diagnostics. Factored out of run_analysis so verify_outputs.py can
    call this SAME code path independently and diff the result against
    what got persisted, as a serialization-fidelity check (does
    write_imaging_fourier_results/read_fourier_results_as_full_fourier_df
    round-trip this exactly?) that doesn't depend on any legacy pickle.

    Returns a dict with everything run_analysis needs to write/plot:
    F, roi_df, included_mask, imaging_dims, fourier_df, freq, Q, Q_2f, T,
    freq_win, onfreq_pow_l, offfreq_pow_l, onfreq_2f_l, offfreq_2f_l.
    """
    freq    = cfg.analysis.f
    Q_frac  = cfg.analysis.Q_frac
    T       = cfg.sample_period
    nyquist = 0.5 / T

    def _p(msg):
        if verbose:
            print(msg)

    _p(f"[engert] {cfg.name}: loading Suite2p from {cfg.nwb_path()}")
    F, roi_df, included_mask, imaging_dims = _load_from_nwb(
        cfg.nwb_path(), cfg.iscell_threshold, cfg.npix_threshold)
    _p(f"[engert] {cfg.name}: {int((roi_df['p_iscell'].values > cfg.iscell_threshold).sum())} "
       f"cells after iscell filter (npix filter combined), {len(F)} after flatline removal")

    # ── Fourier harmonics: 1F always, 2F only if below Nyquist ─────────────
    # Both harmonics go through the identical fit_Fourier -> NFC/p_value/sens
    # derivation below; only the frequency (and whether 2F is skipped)
    # differs, so this loops over (harmonic label, frequency) instead of
    # duplicating the block per harmonic.
    harmonics = [(1, freq)]
    if (2 * freq) < nyquist:
        harmonics.append((2, freq * 2))
    else:
        _p(f"[engert] {cfg.name}: skipping 2F ({freq*2:.3f} Hz >= Nyquist {nyquist:.3f} Hz)")

    n_cells = len(F)
    empty = np.full(n_cells, np.nan)
    results = {
        1: dict(NFC=empty, p_value=empty, sens=empty, onfreq_pow_l=None,
                offfreq_pow_l=None, freq_win=None, M=None),
        2: dict(NFC=empty, p_value=empty, sens=empty, onfreq_pow_l=None,
                offfreq_pow_l=None, freq_win=None, M=None),
    }
    for h, hfreq in harmonics:
        _p(f"[engert] {cfg.name}: fit_Fourier at {hfreq} Hz" + (f" ({h}F)" if h == 2 else f"  (T={T})"))
        chat_l, onfreq_pow_l, offfreq_pow_l, freq_win, M, avg_signal_l = fit_Fourier(
            F, T=T, f=hfreq, Q_frac=Q_frac)

        NFC     = np.array(chat_l)
        p_value = corrected_pvalues(NFC, M)
        fou_alt = np.array(offfreq_pow_l)                             # (C, 2*M) complex
        sigma   = np.sqrt(0.5 * np.mean(np.abs(fou_alt) ** 2, axis=1))
        sens    = avg_signal_l / np.where(sigma > 0, 2 * sigma, np.nan)

        results[h] = dict(NFC=NFC, p_value=p_value, sens=sens, onfreq_pow_l=onfreq_pow_l,
                           offfreq_pow_l=offfreq_pow_l, freq_win=freq_win, M=M)

    r1, r2 = results[1], results[2]
    NFC, p_value, sens = r1["NFC"], r1["p_value"], r1["sens"]
    onfreq_pow_l, offfreq_pow_l = r1["onfreq_pow_l"], r1["offfreq_pow_l"]
    freq_win, M_1f = r1["freq_win"], r1["M"]
    NFC_2f, p_value_2f, sens_2f = r2["NFC"], r2["p_value"], r2["sens"]
    onfreq_2f_l, offfreq_2f_l = r2["onfreq_pow_l"], r2["offfreq_pow_l"]
    freq_win_2f, M_2f = r2["freq_win"], r2["M"]

    # ── Number of frames used by fit_Fourier ────────────────────────────────
    N_frames = int(120 * (F.shape[1] // 60))
    n_frames = np.full(len(NFC), N_frames)

    # ── Build fourier_df ─────────────────────────────────────────────────────
    fourier_df = pd.DataFrame({
        "id":        np.arange(len(NFC)),
        "p_value":   p_value,
        "n_frames":  n_frames,
        "NFC":       NFC,
        "freq":      freq,
        "rec":       cfg.name,
        "2f_NFC":    NFC_2f,
        "2f_p_value": p_value_2f,
        "sens":      sens,
        "sens_2f":   sens_2f,
        "Q":         M_1f,  # bin count behind THIS row's NFC/p_value (1F)
    })

    return {
        "F": F, "roi_df": roi_df, "included_mask": included_mask,
        "imaging_dims": imaging_dims, "fourier_df": fourier_df,
        "freq": freq, "Q": M_1f, "Q_2f": M_2f, "T": T, "freq_win": freq_win,
        "onfreq_pow_l": onfreq_pow_l, "offfreq_pow_l": offfreq_pow_l,
        "freq_win_2f": freq_win_2f,
        "onfreq_2f_l": onfreq_2f_l, "offfreq_2f_l": offfreq_2f_l,
    }


def run_analysis(cfg):
    r = compute_fourier_results(cfg)
    F, roi_df, included_mask, imaging_dims = r["F"], r["roi_df"], r["included_mask"], r["imaging_dims"]
    fourier_df, freq, Q, Q_2f, T = r["fourier_df"], r["freq"], r["Q"], r["Q_2f"], r["T"]
    freq_win = r["freq_win"]
    onfreq_pow_l, offfreq_pow_l = r["onfreq_pow_l"], r["offfreq_pow_l"]
    freq_win_2f = r["freq_win_2f"]
    onfreq_2f_l, offfreq_2f_l = r["onfreq_2f_l"], r["offfreq_2f_l"]

    # ── NWB write (see .claude/plans — NWB replatform, Phase 4) ────────────
    nwb_io.rebuild_and_replace_analysis(
        cfg.nwb_path(),
        lambda nwbfile: nwb_io.write_imaging_fourier_results(
            nwbfile, rec=cfg.name, freq=freq, Q=Q,
            T_duration=F.shape[1] * T, fourier_df_rows=fourier_df,
            onfreq_pow=onfreq_pow_l, offfreq_pow=offfreq_pow_l, freq_win=freq_win,
            onfreq_pow_2f=onfreq_2f_l, offfreq_pow_2f=offfreq_2f_l, Q_2f=Q_2f),
    )
    print(f"[engert] {cfg.name}: saved -> {cfg.nwb_path()}")

    # ── Diagnostics ──────────────────────────────────────────────────────────
    # 2F args are passed straight from this SAME run's in-memory
    # compute_fourier_results() output, not reconstructed from NWB (unlike
    # NPIX's simple.py, which reads diagnostics input back from disk) --
    # per-unit 2F Fourier coefficients are never persisted to NWB (see
    # write_imaging_fourier_results/nwb_io.read_log_dict_equivalent), so
    # this in-memory path is the only source for the 2F spectrum plot.
    from pipeline.diagnostics.engert import plot_engert_diagnostics
    diag_dir = Path(cfg.data_dir).parent / "figs" / "analysis"
    diag_dir.mkdir(parents=True, exist_ok=True)
    plot_engert_diagnostics(
        cfg, F, fourier_df, freq_win,
        onfreq_pow_l, offfreq_pow_l, diag_dir,
        roi_df=roi_df, included_mask=included_mask, imaging_dims=imaging_dims,
        freq_win_2f=freq_win_2f, onfreq_pow_2f=onfreq_2f_l,
        offfreq_pow_2f=offfreq_2f_l, Q_2f=Q_2f)
