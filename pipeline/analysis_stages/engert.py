"""Analysis stage for Engert GCaMP experiments.

Reads Suite2p F/ROI data back from this experiment's NWB file (written by
the processing stage — see pipeline/paradigms/engert.py), applies the
iscell_threshold/npix_threshold mask and flatline removal (unchanged
formulas, still deferred to analysis time so editing thresholds in the YAML
only requires re-running analysis, not reprocessing), runs fit_Fourier at 1F
and 2F (skipping 2F when above Nyquist), builds fourier_df, writes Fourier
results to NWB, and generates diagnostic PDF.

If `cfg.analysis.visual_f` is set (sentinel -1.0 = unset, the default), an
ADDITIONAL, independent Fourier fit runs at that frequency -- not a 1F/2F
harmonic of `cfg.analysis.f`, but its own separate (rec, freq) group,
mirroring how `analysis_stages/medaka.py` always fits a magnetic AND a
visual frequency per trial. This is for the handful of engert experiments
where a visual grating ran concurrently with the magnetic stimulus (see
CLAUDE.md's "Zebrafish/medaka multi-trial sessions" section for exactly
which experiments qualify) -- most engert YAMLs leave `visual_f` unset and
get only the single-frequency 1F/2F fit, unchanged from before this existed.
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


# Default off-frequency half-window for the optional `visual_f` fit, used
# unless a YAML sets its own `analysis.visual_Q_frac`. Matches medaka's own
# `_VISUAL_Q_FRAC` -- same 1/60 Hz stimulus design, same reasoning: smaller
# defaults (0.10/0.20/0.25) all yield too few off-frequency bins to clear
# MIN_FOURIER_BINS=8 at a visual frequency this low (confirmed on the actual
# 2022-09/10 zebrafish recordings, all N=1260 frames -> M=10 at 0.5, vs.
# M<8 at 0.25). Retune per-experiment via `analysis.visual_Q_frac` if a
# future visual_f experiment has a much shorter recording.
_DEFAULT_VISUAL_Q_FRAC = 0.50


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
    freq_win, onfreq_coef_l, offfreq_coef_l, onfreq_coef_2f_l, offfreq_coef_2f_l
    -- plus, only when `cfg.analysis.visual_f` is set: visual_fourier_df,
    visual_freq, Q_visual, visual_freq_win, visual_onfreq_coef_l,
    visual_offfreq_coef_l (all None/absent otherwise).
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
        1: dict(NFC=empty, p_value=empty, sens=empty, onfreq_coef_l=None,
                offfreq_coef_l=None, freq_win=None, M=None),
        2: dict(NFC=empty, p_value=empty, sens=empty, onfreq_coef_l=None,
                offfreq_coef_l=None, freq_win=None, M=None),
    }
    for h, hfreq in harmonics:
        _p(f"[engert] {cfg.name}: fit_Fourier at {hfreq} Hz" + (f" ({h}F)" if h == 2 else f"  (T={T})"))
        NFC_l, onfreq_coef_l, offfreq_coef_l, freq_win, M, avg_signal_l = fit_Fourier(
            F, T=T, f=hfreq, Q_frac=Q_frac)

        NFC     = np.array(NFC_l)
        p_value = corrected_pvalues(NFC, M)
        fou_alt = np.array(offfreq_coef_l)                            # (C, 2*M) complex
        sigma   = np.sqrt(0.5 * np.mean(np.abs(fou_alt) ** 2, axis=1))
        sens    = avg_signal_l / np.where(sigma > 0, 2 * sigma, np.nan)

        results[h] = dict(NFC=NFC, p_value=p_value, sens=sens, onfreq_coef_l=onfreq_coef_l,
                           offfreq_coef_l=offfreq_coef_l, freq_win=freq_win, M=M)

    r1, r2 = results[1], results[2]
    NFC, p_value, sens = r1["NFC"], r1["p_value"], r1["sens"]
    onfreq_coef_l, offfreq_coef_l = r1["onfreq_coef_l"], r1["offfreq_coef_l"]
    freq_win, M_1f = r1["freq_win"], r1["M"]
    NFC_2f, p_value_2f, sens_2f = r2["NFC"], r2["p_value"], r2["sens"]
    onfreq_coef_2f_l, offfreq_coef_2f_l = r2["onfreq_coef_l"], r2["offfreq_coef_l"]
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

    result = {
        "F": F, "roi_df": roi_df, "included_mask": included_mask,
        "imaging_dims": imaging_dims, "fourier_df": fourier_df,
        "freq": freq, "Q": M_1f, "Q_2f": M_2f, "T": T, "freq_win": freq_win,
        "onfreq_coef_l": onfreq_coef_l, "offfreq_coef_l": offfreq_coef_l,
        "freq_win_2f": freq_win_2f,
        "onfreq_coef_2f_l": onfreq_coef_2f_l, "offfreq_coef_2f_l": offfreq_coef_2f_l,
        "visual_freq": None, "Q_visual": None, "visual_freq_win": None,
        "visual_onfreq_coef_l": None, "visual_offfreq_coef_l": None,
    }

    # ── Optional independent visual-frequency fit (NOT a harmonic of `freq`) ──
    # Folded into the SAME `fourier_df` (via concat) rather than a separate
    # dict key -- matching medaka.py's single-fourier_df convention, which
    # verify_outputs.py and every other `["fourier_df"]` consumer expects.
    # `run_analysis` below re-derives this group's own rows via
    # `fourier_df.loc[fourier_df.freq == visual_freq]`, exactly like
    # medaka.py does for its own two frequency groups.
    visual_freq = cfg.analysis.visual_f
    if visual_freq > 0:
        visual_Q_frac = cfg.analysis.visual_Q_frac if cfg.analysis.visual_Q_frac > 0 \
            else _DEFAULT_VISUAL_Q_FRAC
        _p(f"[engert] {cfg.name}: fit_Fourier at {visual_freq} Hz (visual, independent of {freq} Hz)")
        NFC_v_l, onfreq_coef_v_l, offfreq_coef_v_l, freq_win_v, M_v, avg_signal_v_l = fit_Fourier(
            F, T=T, f=visual_freq, Q_frac=visual_Q_frac)

        NFC_v     = np.array(NFC_v_l)
        p_value_v = corrected_pvalues(NFC_v, M_v)
        fou_alt_v = np.array(offfreq_coef_v_l)
        sigma_v   = np.sqrt(0.5 * np.mean(np.abs(fou_alt_v) ** 2, axis=1))
        sens_v    = avg_signal_v_l / np.where(sigma_v > 0, 2 * sigma_v, np.nan)

        # No 2f_NFC/2f_p_value/sens_2f columns -- this is an independent
        # group (like medaka's own visual-frequency fourier_df), not a
        # 1F/2F harmonic pair; concat leaves those NaN for these rows, which
        # write_imaging_fourier_results's onfreq_coef_2f=None (below) matches.
        visual_fourier_df = pd.DataFrame({
            "id":       np.arange(len(NFC_v)),
            "p_value":  p_value_v,
            "n_frames": n_frames,
            "NFC":      NFC_v,
            "freq":     visual_freq,
            "rec":      cfg.name,
            "sens":     sens_v,
            "Q":        M_v,
        })
        result["fourier_df"] = pd.concat([fourier_df, visual_fourier_df], ignore_index=True)
        result.update({
            "visual_freq": visual_freq, "Q_visual": M_v, "visual_freq_win": freq_win_v,
            "visual_onfreq_coef_l": onfreq_coef_v_l, "visual_offfreq_coef_l": offfreq_coef_v_l,
        })

    return result


def run_analysis(cfg):
    r = compute_fourier_results(cfg)
    F, roi_df, included_mask, imaging_dims = r["F"], r["roi_df"], r["included_mask"], r["imaging_dims"]
    freq, Q, Q_2f, T = r["freq"], r["Q"], r["Q_2f"], r["T"]
    freq_win = r["freq_win"]
    onfreq_coef_l, offfreq_coef_l = r["onfreq_coef_l"], r["offfreq_coef_l"]
    freq_win_2f = r["freq_win_2f"]
    onfreq_coef_2f_l, offfreq_coef_2f_l = r["onfreq_coef_2f_l"], r["offfreq_coef_2f_l"]

    # `r["fourier_df"]` holds BOTH groups' rows concatenated when visual_f is
    # set (see compute_fourier_results) -- select each group's own rows by
    # freq before using them, exactly like medaka.py's
    # `fourier_df.loc[fourier_df.freq == f_b]` pattern, since
    # write_imaging_fourier_results/plot_engert_diagnostics both expect
    # exactly len(F) rows aligned 1:1 with onfreq_coef_l/offfreq_coef_l.
    primary_fourier_df = r["fourier_df"].loc[r["fourier_df"].freq == freq].reset_index(drop=True)

    # ── NWB write (see .claude/plans — NWB replatform, Phase 4) ────────────
    def _write(nwbfile):
        nwb_io.write_imaging_fourier_results(
            nwbfile, rec=cfg.name, freq=freq, Q=Q,
            T_duration=F.shape[1] * T, fourier_df_rows=primary_fourier_df,
            onfreq_coef=onfreq_coef_l, offfreq_coef=offfreq_coef_l, freq_win=freq_win,
            onfreq_coef_2f=onfreq_coef_2f_l, offfreq_coef_2f=offfreq_coef_2f_l, Q_2f=Q_2f)
        # Optional independent visual-frequency group (see compute_fourier_results) --
        # written as its own (rec, freq) group, no 2F pairing, matching medaka.py's
        # own visual-frequency write call.
        if r["visual_freq"] is not None:
            visual_fourier_df = r["fourier_df"].loc[
                r["fourier_df"].freq == r["visual_freq"]].reset_index(drop=True)
            nwb_io.write_imaging_fourier_results(
                nwbfile, rec=cfg.name, freq=r["visual_freq"], Q=r["Q_visual"],
                T_duration=F.shape[1] * T, fourier_df_rows=visual_fourier_df,
                onfreq_coef=r["visual_onfreq_coef_l"], offfreq_coef=r["visual_offfreq_coef_l"],
                freq_win=r["visual_freq_win"])

    nwb_io.rebuild_and_replace_analysis(cfg.nwb_path(), _write)
    print(f"[engert] {cfg.name}: saved -> {cfg.nwb_path()}")

    # ── Diagnostics (primary frequency only -- unaffected by visual_f) ──────
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
        cfg, F, primary_fourier_df, freq_win,
        onfreq_coef_l, offfreq_coef_l, diag_dir,
        roi_df=roi_df, included_mask=included_mask, imaging_dims=imaging_dims,
        freq_win_2f=freq_win_2f, onfreq_coef_2f=onfreq_coef_2f_l,
        offfreq_coef_2f=offfreq_coef_2f_l, Q_2f=Q_2f)
