"""
Paradigm: openephys_multistim

OpenEphys recordings with auxiliary stimuli (visual gratings, WN, oddball, bars).
Extends the openephys paradigm: runs process_raw_data_NPIX for magnetic trials,
then dispatches each auxiliary_stimuli entry to a kind-specific handler.
"""
import os
import pickle

import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import tqdm.auto as tqdm
import matplotlib
matplotlib.use("Agg")

from ephysio import openEphysIO
from ephysio.kilosortIO import Reader
from magpyneto2 import (
    get_cluster_info, process_raw_data_NPIX, update_MM_d_mag,
    save_MM_d_pickle, save_diagnostics_MM,
    schmitt, correct_theta, get_condition_array, get_MM_offset,
    get_concat_spks_consistent_period,
)


def _build_contingency(cfg, ksr, udf, cat_df):
    """Build contingency_d and aux_d by iterating cat_df rows."""
    label = "good" if cfg.good else None
    all_sts = ksr.spikesbycluster(label=label)
    contingency_d = {}
    aux_d = {}

    # cluster_info.tsv column name varies by kilosort version
    id_col = "cluster_id" if "cluster_id" in udf.columns else "id"

    for catrow_i, catrow in tqdm.tqdm(cat_df.iterrows(), total=len(cat_df)):
        spath = catrow.path.split(catrow.recname)
        recroot_path = spath[0] + catrow.recname

        # Per-row stream override if cfg.streams is populated
        stream_id = cfg.stream_id
        if cfg.streams:
            idx = cat_df.index.get_loc(catrow_i)
            if idx < len(cfg.streams) and cfg.streams[idx] is None:
                continue
            if idx < len(cfg.streams) and cfg.streams[idx] is not None:
                stream_id = cfg.streams[idx]

        try:
            recording = se.OpenEphysBinaryRecordingExtractor(
                recroot_path, stream_id=stream_id)
        except Exception:
            print(f"  WARNING: could not load {recroot_path}, skipping")
            continue

        ldr = openEphysIO.Loader(
            recroot_path.replace("\\", "/"), cntlbarcodes=cfg.cntlbarcodes)

        beginning_time = 0 if catrow_i == 0 else cat_df.iloc[catrow_i - 1]["cumulate"]

        aux_d[catrow.recname] = (recording, ldr)
        contingency_d[catrow.recname] = {
            getattr(cr, id_col): all_sts[getattr(cr, id_col)][
                (all_sts[getattr(cr, id_col)] > beginning_time) &
                (all_sts[getattr(cr, id_col)] < catrow.cumulate)
            ] - beginning_time
            for _, cr in udf.iterrows()
        }

    return all_sts, contingency_d, aux_d


def _handle_visual_gratings(aux_cfg, contingency_d, aux_d, udf, cat_df, df_l, MM_d):
    recname = aux_cfg.recname
    st_d = contingency_d[recname]
    recording, ldr = aux_d[recname]

    viz_trace = recording.get_traces()[:, aux_cfg.channel]

    df = pd.read_csv(aux_cfg.orientation_csv, header=None)
    trial_sequence = [str(line).split("ori")[-1][0] for line in df.values]
    orientation_d = {key: int(val) * 45 for key, val in enumerate(trial_sequence)}

    iup, idown = schmitt(
        np.array(viz_trace, dtype="float64"),
        thr_on=aux_cfg.thr_on, thr_off=aux_cfg.thr_off, starttype=0, endtype=0)

    if aux_cfg.iup_min_filter > 0:
        idown = idown[iup > aux_cfg.iup_min_filter]
        iup = iup[iup > aux_cfg.iup_min_filter]

    deststream = ldr.spikestreams()[0] if aux_cfg.deststream == "auto" else aux_cfg.deststream
    sourcestream = ldr.nidaqstream() if aux_cfg.sourcestream == "auto" else aux_cfg.sourcestream

    t_up = ldr.shifttime(
        iup, deststream=deststream,
        sourcestream=sourcestream, sourcebarcode=aux_cfg.sourcebarcode)
    t_down = ldr.shifttime(
        idown, deststream=deststream,
        sourcestream=sourcestream, sourcebarcode=aux_cfg.sourcebarcode)

    trials = np.zeros_like(t_up)
    for trial, inc in enumerate(np.where(np.diff(t_up) > aux_cfg.trial_gap_samples)[0]):
        trials[inc + 1:] = trial + 1

    theta = correct_theta(int(np.ceil(t_up[-1])), np.round(t_up).astype(int))

    for unit_id, spks in tqdm.tqdm(st_d.items()):
        trial_df_l = []
        for trial in range(aux_cfg.n_orientations):
            period_markers = t_up[trials == trial]
            if len(period_markers) == 0:
                continue
            spk = spks[(spks > period_markers[0]) & (spks < period_markers[-1])]
            if len(spk) > 50:
                phase = theta[spk.astype(int)]
                period = get_condition_array(spk, period_markers)
                trial_df_l.append(pd.DataFrame({
                    "period": period,
                    "spk": spk / 30_000,
                    "phase": phase,
                    "freq": aux_cfg.frequency,
                    "id": unit_id,
                    "rec": recname + "_" + str(orientation_d[trial]),
                }))
        if len(trial_df_l) == aux_cfg.n_orientations:
            df_l.extend(trial_df_l)

    ap_sr = ldr.samplingrate(ldr.spikestream())
    offset = get_MM_offset(cat_df, recname)
    for trial in range(aux_cfg.n_orientations):
        period_markers = t_up[trials == trial]
        if len(period_markers) == 0:
            continue
        MM_d["aux"][recname + str(orientation_d[trial])] = (
            np.array([np.min(period_markers), np.max(period_markers)]) + offset,
            aux_cfg.frequency,
        )


def _handle_white_noise(aux_cfg, contingency_d, aux_d, udf, cat_df, df_l, MM_d):
    """White-noise handler.

    Two modes, controlled by duration_s:
    - duration_s <= 10: multi-window (each t_down marks end of a short burst; iterate all)
    - duration_s >  10: single-window (300s before the last t_down)
    """
    recname = aux_cfg.recname
    st_d = contingency_d[recname]
    recording, ldr = aux_d[recname]

    aud_trace = recording.get_traces()[:, aux_cfg.channel]
    iup, idown = schmitt(
        np.array(aud_trace, dtype="float64"),
        thr_on=aux_cfg.thr_on, thr_off=aux_cfg.thr_off, starttype=0, endtype=0)

    deststream = ldr.spikestreams()[0] if aux_cfg.deststream == "auto" else aux_cfg.deststream
    sourcestream = ldr.nidaqstream() if aux_cfg.sourcestream == "auto" else aux_cfg.sourcestream

    t_up = ldr.shifttime(
        iup, deststream=deststream,
        sourcestream=sourcestream, sourcebarcode=aux_cfg.sourcebarcode)
    t_down = ldr.shifttime(
        idown, deststream=deststream,
        sourcestream=sourcestream, sourcebarcode=aux_cfg.sourcebarcode)

    ap_sr = ldr.samplingrate(ldr.spikestream())
    offset = get_MM_offset(cat_df, recname)
    freq = aux_cfg.frequency
    duration_samples = int(aux_cfg.duration_s * ap_sr)

    if aux_cfg.duration_s <= 10:
        # Multi-window: iterate t_up/t_down pairs, shift spike times
        legacy = getattr(aux_cfg, "wn_legacy_formula", False)
        period_s = 1.0 / freq
        if not legacy:
            periods_qs = [np.array([1.25, 2.5, 3.75, 5]) + aux_cfg.duration_s * i
                          for i in np.arange(len(t_down))]
        MM_d["aux"][recname] = (
            np.vstack([t_down - duration_samples, t_down]).T + offset,
            freq,
        )
        for unit_id, spks in st_d.items():
            concat_l = []
            for i in np.arange(len(t_down)):
                stim_start = t_down[i] - duration_samples
                concat_l.append(
                    spks[(spks > stim_start) & (spks < t_down[i])]
                    - stim_start + i * duration_samples)
            spk_s = np.concatenate(concat_l) / ap_sr
            if len(spk_s) > 50:
                if legacy:
                    period_col = spk_s // period_s
                    phase_col = (spk_s % period_s) / period_s * 2 * np.pi
                else:
                    period_col = [np.sum(t > periods_qs) for t in spk_s]
                    phase_col = spk_s % period_s * 2 * np.pi
                trial_df_l = [pd.DataFrame({
                    "period": period_col,
                    "spk": spk_s,
                    "phase": phase_col,
                    "freq": freq,
                    "id": unit_id,
                    "rec": recname,
                })]
                df_l.append(pd.concat(trial_df_l))
    else:
        # Single-window: 300s before last t_down
        MM_d["aux"][recname] = (
            np.vstack([t_down - duration_samples, t_down]).T + offset,
            freq,
        )
        period_freq = aux_cfg.wn_period_freq if aux_cfg.wn_period_freq > 0 else freq
        for unit_id, unit_st in tqdm.tqdm(sorted(st_d.items()), total=len(st_d)):
            concat_spks = unit_st[
                (unit_st > (t_down - duration_samples)) & (unit_st < t_down)
            ] / ap_sr
            if len(concat_spks) > 50:
                period = concat_spks // (1 / period_freq)
                phase = (concat_spks % (1 / period_freq)) / (1 / period_freq) * 2 * np.pi
                df_l.append(pd.DataFrame({
                    "period": period,
                    "spk": concat_spks,
                    "phase": phase,
                    "freq": freq,
                    "id": unit_id,
                    "rec": recname,
                }))


def _handle_oddball(aux_cfg, contingency_d, aux_d, udf, cat_df, df_l, MM_d):
    """Oddball: inter-event intervals where t_down[i] - t_down[i-1] < max_interval_s."""
    recname = aux_cfg.recname
    st_d = contingency_d[recname]
    recording, ldr = aux_d[recname]

    viz_trace = recording.get_traces()[:, aux_cfg.channel]
    iup, idown = schmitt(
        np.array(viz_trace, dtype="float64"),
        thr_on=aux_cfg.thr_on, thr_off=aux_cfg.thr_off, starttype=0, endtype=0)

    deststream = ldr.spikestreams()[0] if aux_cfg.deststream == "auto" else aux_cfg.deststream
    sourcestream = ldr.nidaqstream() if aux_cfg.sourcestream == "auto" else aux_cfg.sourcestream

    t_up = ldr.shifttime(
        iup, deststream=deststream,
        sourcestream=sourcestream, sourcebarcode=aux_cfg.sourcebarcode)
    t_down = ldr.shifttime(
        idown, deststream=deststream,
        sourcestream=sourcestream, sourcebarcode=aux_cfg.sourcebarcode)

    ap_sr = ldr.samplingrate(ldr.spikestream())
    offset = get_MM_offset(cat_df, recname)
    max_interval_s = getattr(aux_cfg, "max_interval_s", 1.2)

    # Build intervals: (curr, prev) pairs where gap < max_interval_s
    intervals = []
    for down_i in range(1, len(t_down) - 1):
        curr = t_down[down_i]
        prev = t_down[down_i - 1]
        if (curr - prev) / ap_sr < max_interval_s:
            intervals.append((curr, prev))

    if len(intervals) == 0:
        print(f"  WARNING: no oddball intervals found for {recname}")
        return

    period = np.abs(np.mean(np.diff(intervals)) / ap_sr)
    freq = 1 / period
    periods_arr = np.arange(len(intervals)) * period

    MM_d["aux"][recname] = (np.array(intervals) + offset, freq)

    trial_df_l = []
    for unit_id, st in tqdm.tqdm(st_d.items()):
        ball_spks = []
        for interval_ind, (curr, prev) in enumerate(intervals):
            shifted = (st[(st > prev) & (st < curr)] - prev
                       + interval_ind * period * ap_sr) / ap_sr
            ball_spks.append(shifted)
        ball_spks = np.concatenate(ball_spks)
        if len(ball_spks) > 50:
            trial_df_l.append(pd.DataFrame({
                "period": [np.sum(t > periods_arr) for t in ball_spks],
                "spk": ball_spks,
                "phase": ball_spks % period * 2 * np.pi,
                "freq": freq,
                "id": unit_id,
                "rec": recname,
            }))

    if trial_df_l:
        df_l.append(pd.concat(trial_df_l))


def _handle_visual_bars(aux_cfg, contingency_d, aux_d, udf, cat_df, df_l, MM_d):
    """20220916-style visual bars.

    Orientation CSV has 'trial' and 'deg' columns.
    First stimulus stamp is lost (decrement trial index by 1).
    Uses get_concat_spks_consistent_period with a fixed 4-second period.
    """
    recname = aux_cfg.recname
    st_d = contingency_d[recname]
    recording, ldr = aux_d[recname]

    viz_trace = recording.get_traces()[:, aux_cfg.channel]
    iup, idown = schmitt(
        np.array(viz_trace, dtype="float64"),
        thr_on=aux_cfg.thr_on, thr_off=aux_cfg.thr_off, starttype=0, endtype=0)

    deststream = ldr.spikestreams()[0] if aux_cfg.deststream == "auto" else aux_cfg.deststream
    sourcestream = ldr.nidaqstream() if aux_cfg.sourcestream == "auto" else aux_cfg.sourcestream

    t_up = ldr.shifttime(
        iup, deststream=deststream,
        sourcestream=sourcestream, sourcebarcode=aux_cfg.sourcebarcode)
    t_down = ldr.shifttime(
        idown, deststream=deststream,
        sourcestream=sourcestream, sourcebarcode=aux_cfg.sourcebarcode)

    ap_sr = ldr.samplingrate(ldr.spikestream())
    offset = get_MM_offset(cat_df, recname)

    df_csv = pd.read_csv(aux_cfg.orientation_csv)
    df_csv.columns = ["trial", "deg"]
    degs = df_csv.deg.unique()

    for deg in tqdm.tqdm(degs):
        deg_df = df_csv.loc[df_csv.deg == deg, :].copy()
        deg_df.trial = deg_df.trial - 1   # first stamp lost

        ons = t_up[deg_df.trial]
        offs = t_down[deg_df.trial]

        bar_freq = 1 / (np.mean(offs - ons) / ap_sr)
        period_samples = int(4 * ap_sr)

        MM_d["aux"][recname + "_" + str(deg)] = (
            np.vstack([t_down - 5 * ap_sr, t_down]).T + offset,
            bar_freq,
        )

        freq = 1 / 4  # fixed 4-second period
        for unit_id, sts in st_d.items():
            concat_spks = get_concat_spks_consistent_period(
                sts, ons, offs, period_samples) / ap_sr
            if len(concat_spks) > 50:
                period = concat_spks // (1 / freq)
                phase = (concat_spks % (1 / freq)) / (1 / freq) * 2 * np.pi
                df_l.append(pd.DataFrame({
                    "period": period,
                    "spk": concat_spks,
                    "phase": phase,
                    "freq": freq,
                    "id": unit_id,
                    "rec": recname + "_" + str(deg),
                }))


_HANDLERS = {
    "visual_gratings": _handle_visual_gratings,
    "white_noise":     _handle_white_noise,
    "oddball":         _handle_oddball,
    "visual_bars":     _handle_visual_bars,
}


def run_processing(cfg):
    cat_df = pd.read_csv(cfg.metadata_csv)
    cat_df["cumulate"] = np.cumsum(cat_df["nframes"].values)

    ksr = Reader(cfg.aggregated_path)
    udf = get_cluster_info(cfg.aggregated_path)
    if cfg.good:
        udf = udf.loc[udf.KSLabel == "good", :]

    all_sts, contingency_d, aux_d = _build_contingency(cfg, ksr, udf, cat_df)  # noqa

    # --- Magnetic trials ---
    folder_locations_freq_skips = []
    for trial in cfg.trials:
        recname = trial.recname
        if recname not in contingency_d:
            raise KeyError(f"Trial recname '{recname}' not in metadata CSV")
        folder_locations_freq_skips.append((
            trial.folder,
            trial.frequency,
            trial.skips,
            contingency_d[recname],
            *aux_d[recname],
        ))

    modulation_df, data, _ = process_raw_data_NPIX(
        folder_locations_freq_skips, THRES=cfg.threshold)

    MM_d = {"spikes": all_sts, "aux": {}}
    MM_d = update_MM_d_mag(MM_d, data, cat_df)

    df_l = [modulation_df]

    # --- Auxiliary stimuli ---
    for aux_cfg in cfg.auxiliary_stimuli:
        handler = _HANDLERS.get(aux_cfg.kind)
        if handler is None:
            raise ValueError(f"Unknown aux stimulus kind: {aux_cfg.kind}")
        if aux_cfg.recname not in contingency_d:
            print(f"  WARNING: aux recname '{aux_cfg.recname}' not in contingency_d, skipping")
            continue
        handler(aux_cfg, contingency_d, aux_d, udf, cat_df, df_l, MM_d)

    modulation_df = pd.concat(df_l).reset_index(drop=True)

    with open(cfg.processing_path(), "wb") as f:
        pickle.dump(modulation_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    mm_path = os.path.join(cfg.data_dir, f"MM_{cfg.name}.pickle")
    save_MM_d_pickle(MM_d, mm_path)
    save_diagnostics_MM(MM_d, cfg.name)

    from pathlib import Path
    from pipeline.diagnostics.processing import plot_recording_timeline
    diag_dir = Path(cfg.data_dir).parent / "figs" / "processing"
    diag_dir.mkdir(parents=True, exist_ok=True)
    plot_recording_timeline(cfg, MM_d, modulation_df, diag_dir)
