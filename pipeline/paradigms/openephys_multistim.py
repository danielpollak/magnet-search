"""
Paradigm: openephys_multistim

OpenEphys recordings with auxiliary stimuli (visual gratings, WN, oddball, bars).
Extends the openephys paradigm: runs process_raw_data_NPIX for magnetic trials,
then dispatches each auxiliary_stimuli entry to a kind-specific handler.
"""
import os

import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import tqdm.auto as tqdm
import matplotlib
matplotlib.use("Agg")

from ephysio import openEphysIO
from ephysio.kilosortIO import Reader
from magpyneto2 import (
    get_cluster_info, process_raw_data_NPIX,
    schmitt, correct_theta, get_condition_array, get_MM_offset,
    get_concat_spks_consistent_period,
)
from pipeline import nwb_io

AP_SR = 30_000


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


def _handle_visual_gratings(cfg, aux_cfg, contingency_d, aux_d, udf, cat_df, df_l, nwb_epochs):
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
    # "gratings" NWB reconstruction needs TWO marker arrays in the aggregated
    # domain: the global t_up (rounded, matching legacy's own rounding
    # before building `theta`) for phase, and each trial's own unrounded
    # period_markers subset (get_condition_array is sensitive to sub-sample
    # precision) for period -- see nwb_io.write_epochs_table's docstring.
    phase_crossings_agg = np.round(t_up).astype(np.int64) + offset
    for trial in range(aux_cfg.n_orientations):
        period_markers = t_up[trials == trial]
        if len(period_markers) == 0:
            continue
        period_markers_agg = period_markers + offset
        nwb_epochs.append({
            "rec": recname + "_" + str(orientation_d[trial]),
            "stim_type": "visual_gratings",
            "frequency": float(aux_cfg.frequency),
            "start_time": float(period_markers_agg[0]) / 30_000,
            "stop_time": float(period_markers_agg[-1]) / 30_000,
            "period_crossings": period_markers_agg,
            "phase_crossings": phase_crossings_agg,
            "phase_method": "gratings",
            "local_offset_samples": int(offset),
            # Legacy only includes a unit's data for ANY orientation of this
            # recname if it passes min_spikes in EVERY orientation (`if
            # len(trial_df_l) == aux_cfg.n_orientations: df_l.extend(...)`
            # above) -- see nwb_io._gratings_group_exclusions.
            "gratings_group": recname,
        })


def _handle_white_noise(cfg, aux_cfg, contingency_d, aux_d, udf, cat_df, df_l, nwb_epochs):
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
        window_starts = t_down - duration_samples + offset
        window_stops = t_down + offset
        window_offsets = np.arange(len(t_down)) * duration_samples
        nwb_epochs.append({
            "rec": recname,
            "stim_type": "white_noise",
            "frequency": float(freq),
            "start_time": float(window_starts.min()) / ap_sr,
            "stop_time": float(window_stops.max()) / ap_sr,
            "period_crossings": np.array([window_starts.min(), window_stops.max()]),
            "phase_method": "stitched_floor" if legacy else "stitched_crossings_unnorm",
            "window_starts": window_starts,
            "window_stops": window_stops,
            "window_offsets": window_offsets,
            **({} if legacy else {"synthetic_period_markers": np.concatenate(periods_qs)}),
        })
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
        period_freq = aux_cfg.wn_period_freq if aux_cfg.wn_period_freq > 0 else freq
        nwb_epochs.append({
            "rec": recname,
            "stim_type": "white_noise",
            "frequency": float(freq),
            "start_time": (float(t_down) - duration_samples + offset) / ap_sr,
            "stop_time": (float(t_down) + offset) / ap_sr,
            # float(t_down): t_down is a 1-element array here (same as the
            # start_time/stop_time casts above) -- without it, this produces
            # a (2, 1)-shaped array instead of (2,), which breaks the
            # period_crossings ragged column's homogeneity once concatenated
            # against other epochs' plain 1-D arrays (confirmed on real data:
            # HDMF's "inhomogeneous shape" write error).
            "period_crossings": np.array([float(t_down) - duration_samples, float(t_down)]),
            "phase_method": "arithmetic",
            "period_frequency": float(period_freq),
            "local_offset_samples": int(offset),
        })
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


_ODDBALL_CATEGORIES = ("standard", "long_on", "long_off", "long_both")
# only "standard" clears enough trials for fit_fourier_sig -- see multistim.py
_ODDBALL_FOURIER_CATEGORIES = ("standard",)


def _classify_oddball_trials(t_up, t_down, ap_sr, on_normal_max_s, on_long_min_s,
                              off_normal_max_s, off_long_min_s):
    """Classify each candidate trial (bounded by adjacent down-crossings
    down_i-1, down_i) by its own on-duration (t_up[down_i] -> t_down[down_i],
    the pulse itself) and off-duration (t_down[down_i-1] -> t_up[down_i], the
    preceding silence) SEPARATELY -- not by the combined down-to-down gap
    this function used to test as one value, which conflated two distinct
    manipulations (a long stimulus-ON period vs. a long stimulus-OFF/silence
    period before an otherwise-normal pulse) into one "deviant" bucket
    (confirmed on real data: some "deviant"-by-combined-gap trials have a
    completely standard on-duration, just preceded by an unusually long
    silence, and vice versa; a few have BOTH elevated at once). See
    MagnetSearch/code/notebooks/20230221_concatenated.ipynb's ball_d/
    oddball_d split, which the original single-gap version of this ported --
    this generalizes it along the axis the notebook never separated.

    <= *_normal_max_s -> "normal" on that axis; >= *_long_min_s -> "long";
    in between is a dead zone -- ambiguous on that axis, dropped from every
    category (same dead-zone treatment as the original ball_d/oddball_d
    split, just applied per-axis instead of to the combined gap).

    Returns (buckets, dropped_down_idx): buckets is
    {"standard"|"long_on"|"long_off"|"long_both": (intervals, down_indices)},
    each `intervals` a list of (curr, prev) sample pairs and `down_indices`
    the down_i each interval came from (parallel, same order, both empty for
    a category with no trials).
    """
    buckets = {cat: ([], []) for cat in _ODDBALL_CATEGORIES}
    dropped_down_idx = []
    for down_i in range(1, len(t_down) - 1):
        curr, prev = t_down[down_i], t_down[down_i - 1]
        on_dur_s = (t_down[down_i] - t_up[down_i]) / ap_sr
        off_dur_s = (t_up[down_i] - t_down[down_i - 1]) / ap_sr

        if on_dur_s <= on_normal_max_s:
            on_state = "normal"
        elif on_dur_s >= on_long_min_s:
            on_state = "long"
        else:
            on_state = "ambiguous"

        if off_dur_s <= off_normal_max_s:
            off_state = "normal"
        elif off_dur_s >= off_long_min_s:
            off_state = "long"
        else:
            off_state = "ambiguous"

        if on_state == "ambiguous" or off_state == "ambiguous":
            dropped_down_idx.append(down_i)
            continue

        category = {
            ("normal", "normal"): "standard",
            ("long", "normal"): "long_on",
            ("normal", "long"): "long_off",
            ("long", "long"): "long_both",
        }[(on_state, off_state)]
        buckets[category][0].append((curr, prev))
        buckets[category][1].append(down_i)

    return buckets, dropped_down_idx


def _pad_oddball_windows(down_idx, t_up, t_down, ap_sr, recname):
    """Pre/post pad each trial's window by half of THAT TRIAL'S OWN
    on-duration (t_down[down_i] - t_up[down_i]), i.e. pre = post = on_dur/2,
    ANCHORED AT THE STIMULUS'S OWN ONSET (t_up[down_i]) rather than at the
    window's `prev` (previous offset) boundary -- the evoked response is
    locked to stimulus onset, not to whenever the previous stimulus happened
    to end (confirmed: anchoring at `prev` instead made pre-pad clip to ~0
    for every trial, since the preceding trial's own on-span always ends
    exactly at `prev` with zero natural room; anchoring at onset gives every
    trial real room, bounded by ITS OWN off-duration).

    Because onset is used as the anchor, pre-pad only reaches into the
    PRECEDING trial's own stimulus-on span if the requested pad exceeds this
    trial's own (already-natural) off-duration -- normal trials (small pad,
    normal off-duration) essentially never clip; long_on trials (pad ~=
    on_dur/2, often bigger than a NORMAL off-duration) clip regularly,
    landing exactly at the preceding trial's own offset (`prev`) rather than
    reaching any further. Post-pad is symmetric: it only reaches the
    FOLLOWING trial's own onset if the requested pad exceeds THAT trial's
    own off-duration.

    down_idx : down-crossing indices for every candidate trial across ALL
        categories (standard + long_on + long_off + long_both) AND the
        dropped dead-zone ones, NOT pre-split by category -- bounds come
        directly from t_down[down_i-1]/t_up[down_i+1], so no per-category
        neighbor lookup is needed: every down_i, dropped or not, has a real
        on-span at that index that must never be contaminated.
    Returns {down_i: (padded_start, padded_stop)}, same (pre-offset) sample
    domain as t_up/t_down.
    """
    windows = {}
    for di in down_idx:
        onset, curr = t_up[di], t_down[di]
        on_dur_s = (curr - onset) / ap_sr
        pad = on_dur_s / 2 * ap_sr

        lo_bound = t_down[di - 1]                                      # preceding trial's own offset
        hi_bound = t_up[di + 1] if di + 1 < len(t_up) else np.inf       # following trial's own onset

        padded_start = onset - pad
        if padded_start < lo_bound:
            print(f"  WARNING: {recname} trial (down_i={di}) pre-pad clipped -- "
                  f"requested {pad / ap_sr:.3f}s would include the preceding "
                  f"trial's own stimulus-on period; clipped to "
                  f"{(onset - lo_bound) / ap_sr:.3f}s")
            padded_start = lo_bound

        padded_stop = curr + pad
        if padded_stop > hi_bound:
            print(f"  WARNING: {recname} trial (down_i={di}) post-pad clipped -- "
                  f"requested {pad / ap_sr:.3f}s would include the following "
                  f"trial's own stimulus-on period; clipped to "
                  f"{(hi_bound - curr) / ap_sr:.3f}s")
            padded_stop = hi_bound

        windows[di] = (padded_start, padded_stop)
    return windows


def _emit_oddball_epoch(intervals, down_idx, windows_by_di, t_up, epoch_rec, stim_type,
                         st_d, offset, ap_sr, nwb_epochs, df_l, recname_for_df):
    """Build the synthetic stitched-timeline epoch + per-unit phase rows for
    one trial category and append them.

    Phase is anchored at each trial's own onset (t_up[down_i]), NOT at the
    (possibly padded) window start -- the response is locked to stimulus
    onset, not to wherever padding happens to begin (see
    _pad_oddball_windows). window_starts/window_stops (the actual spike-
    inclusion bounds) and phase_anchors (onset) are written as separate NWB
    columns so nwb_io._stitch_synthetic_samples keeps them decoupled at
    analysis time too -- see write_epochs_table's phase_anchors docstring.

    epoch_rec is the `rec` value written to nwb_epochs/the per-spike df --
    it must be distinct per category so fit_fourier_sig's
    groupby(["rec", "freq"]) doesn't mix categories together (only relevant
    for categories in _ODDBALL_FOURIER_CATEGORIES -- multistim.py never
    calls fit_fourier_sig on the others, but the epoch/spike data is still
    written here so it's inspectable via build_modulation_frame and the
    trial diagnostics).
    """
    if len(intervals) == 0:
        print(f"  WARNING: no {stim_type} oddball intervals found for {recname_for_df}")
        return

    windows = [windows_by_di[di] for di in down_idx]
    onsets = [t_up[di] for di in down_idx]

    period = np.abs(np.mean(np.diff(intervals)) / ap_sr)
    freq = 1 / period
    periods_arr = np.arange(len(intervals)) * period

    windows_arr = np.array(windows)  # columns: (padded_start, padded_stop)
    window_starts = windows_arr[:, 0] + offset
    window_stops = windows_arr[:, 1] + offset
    phase_anchors = np.array(onsets) + offset
    window_offsets = np.arange(len(intervals)) * period * ap_sr
    nwb_epochs.append({
        "rec": epoch_rec,
        "stim_type": stim_type,
        "frequency": float(freq),
        "start_time": float(window_starts.min()) / ap_sr,
        "stop_time": float(window_stops.max()) / ap_sr,
        "period_crossings": np.array([window_starts.min(), window_stops.max()]),
        "phase_method": "stitched_crossings_unnorm",
        "window_starts": window_starts,
        "window_stops": window_stops,
        "window_offsets": window_offsets,
        "phase_anchors": phase_anchors,
        "synthetic_period_markers": periods_arr,
    })

    trial_df_l = []
    for unit_id, st in tqdm.tqdm(st_d.items()):
        ball_spks = []
        for interval_ind, ((padded_start, padded_stop), onset) in enumerate(zip(windows, onsets)):
            shifted = (st[(st > padded_start) & (st < padded_stop)] - onset
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
                "rec": epoch_rec,
            }))

    if trial_df_l:
        df_l.append(pd.concat(trial_df_l))


def _handle_oddball(cfg, aux_cfg, contingency_d, aux_d, udf, cat_df, df_l, nwb_epochs):
    """Oddball: classify each candidate trial (adjacent down-crossings) into
    standard / long_on / long_off / long_both by its own on- and off-
    durations (see _classify_oddball_trials) -- generalizes the original
    single-gap standard/deviant split (MagnetSearch/code/notebooks/
    20230221_concatenated.ipynb's ball_d/oddball_d) once real data showed
    the combined down-to-down gap conflates two distinct manipulations. Only
    "standard" is Fourier-analyzed (see multistim.py's _ODDBALL_FOURIER_CATEGORIES
    usage); the other three are written as full NWB epochs + covered by the
    trial diagnostics, but are each too sparse on their own for fit_fourier_sig."""
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

    buckets, dropped_down_idx = _classify_oddball_trials(
        t_up, t_down, ap_sr,
        getattr(aux_cfg, "on_normal_max_s", 0.7), getattr(aux_cfg, "on_long_min_s", 0.9),
        getattr(aux_cfg, "off_normal_max_s", 0.7), getattr(aux_cfg, "off_long_min_s", 0.9))

    all_down_idx = sorted(
        [di for _, idx in buckets.values() for di in idx] + dropped_down_idx)
    windows_by_di = _pad_oddball_windows(all_down_idx, t_up, t_down, ap_sr, recname)

    # Raw-TTL diagnostic -- plotted regardless of whether any intervals were
    # found (empty-interval sessions are exactly when seeing the raw trace
    # matters most for debugging). t_down/idown share indexing 1:1 (t_down is
    # just idown run through ldr.shifttime()), so down_i applies directly.
    # A crossing that ends one candidate trial also STARTS the next one, so
    # it can only carry one category label here -- later assignments win;
    # this is a rough visual aid, not authoritative (the NWB epochs/
    # diagnostics below are keyed by down_i, never ambiguous).
    try:
        from pathlib import Path
        from pipeline.diagnostics.oddball import plot_oddball_raw_ttl
        category = np.zeros(len(idown), dtype=int)  # 0=dropped, 1..4=standard/long_on/long_off/long_both
        code_of = {"standard": 1, "long_on": 2, "long_off": 3, "long_both": 4}
        for cat, (_, idx) in buckets.items():
            if not idx:
                continue
            code = code_of[cat]
            category[sorted(idx)] = code
            category[sorted(i - 1 for i in idx if i - 1 >= 0)] = code
        diag_dir = Path(cfg.data_dir).parent / "figs" / "processing"
        diag_dir.mkdir(parents=True, exist_ok=True)
        plot_oddball_raw_ttl(
            cfg, recname, viz_trace, recording.get_sampling_frequency(),
            aux_cfg.thr_on, aux_cfg.thr_off, iup, idown, category, diag_dir)
    except Exception as exc:
        print(f"  WARNING: oddball raw-TTL diagnostics failed ({exc})")

    # Trial-level diagnostic -- stimulus/baseline spans, dropped-trial
    # markers, and the actual spike raster together, against the padded
    # windows that really feed the analysis (see plot_oddball_trial_diagnostics).
    try:
        from pathlib import Path
        from pipeline.diagnostics.oddball import plot_oddball_trial_diagnostics
        diag_dir = Path(cfg.data_dir).parent / "figs" / "processing"
        diag_dir.mkdir(parents=True, exist_ok=True)
        plot_oddball_trial_diagnostics(
            cfg, recname, st_d, t_up, t_down, ap_sr,
            buckets, windows_by_di, dropped_down_idx, diag_dir)
    except Exception as exc:
        print(f"  WARNING: oddball trial diagnostics failed ({exc})")

    _emit_oddball_epoch(
        buckets["standard"][0], buckets["standard"][1], windows_by_di, t_up,
        recname, "oddball", st_d, offset, ap_sr, nwb_epochs, df_l, recname)
    for cat in ("long_on", "long_off", "long_both"):
        _emit_oddball_epoch(
            buckets[cat][0], buckets[cat][1], windows_by_di, t_up,
            f"{recname}_{cat}", f"oddball_{cat}", st_d, offset, ap_sr,
            nwb_epochs, df_l, recname)

def _handle_visual_bars(cfg, aux_cfg, contingency_d, aux_d, udf, cat_df, df_l, nwb_epochs):
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

        period_samples = int(4 * ap_sr)

        freq = 1 / 4  # fixed 4-second period

        # Reproduce get_concat_spks_consistent_period's own windowing
        # exactly. Its `last_off` is NOT a simple i*period_samples running
        # offset (confirmed wrong on real data -- every window past the
        # first landed at the wrong synthetic position): each window's
        # shift is the RAW absolute `off + BL` of the *previous* window
        # (0 for the first window), i.e. window i's shift is
        # `offs[i-1] + BL[i-1]` for i>=1, NOT `i * period_samples`. Each
        # window's own span still has length == period_samples (by
        # construction of BL), so this only differs from i*period_samples
        # when consecutive windows aren't perfectly back-to-back in real
        # time -- which is the normal case.
        ons_arr = np.asarray(ons, dtype=np.float64)
        offs_arr = np.asarray(offs, dtype=np.float64)
        BL = (period_samples - (offs_arr - ons_arr)) / 2
        window_starts = ons_arr - BL + offset
        window_stops = offs_arr + BL + offset
        window_offsets = np.zeros(len(ons_arr))
        window_offsets[1:] = offs_arr[:-1] + BL[:-1]
        nwb_epochs.append({
            "rec": recname + "_" + str(deg),
            "stim_type": "visual_bars",
            "frequency": float(freq),
            "start_time": float(window_starts.min()) / ap_sr,
            "stop_time": float(window_stops.max()) / ap_sr,
            "period_crossings": np.array([window_starts.min(), window_stops.max()]),
            "phase_method": "stitched_floor",
            "window_starts": window_starts,
            "window_stops": window_stops,
            "window_offsets": window_offsets,
        })

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

    # NWB Units table gets these SAME cfg.good-filtered spike trains (same
    # reasoning as openephys.py) -- only `good` units are written by
    # default; set `good: False` and reprocess to recover MUA. `all_sts`
    # here is already good-filtered (_build_contingency's own `label`
    # matches cfg.good identically), so no separate unfiltered computation
    # is needed -- see .claude/plans, NWB replatform "good-only" cutover.
    all_sts_nwb = all_sts
    udf_nwb = udf

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

    df_l = [modulation_df]

    # --- NWB dual-write prep (see .claude/plans — NWB replatform, Phase 3) ---
    # Units are a single shared/concatenated catalog across every rec in this
    # file (rec="", same as openephys) -- unlike gutfreund's per-recording
    # independent sortings.
    nwb_epochs = []
    # `folder`'s own basename is what process_raw_data_NPIX itself uses for
    # modulation_df's "rec" column (confirmed on real data -- legacy's own
    # rec label does NOT reflect a trial's `recname:` override), but
    # `cat_df`/metadata-CSV lookups (get_MM_offset, skips) key on the
    # metadata CSV's own recname column, which for a trial with an explicit
    # override is DIFFERENT from the folder's basename (that mismatch is
    # exactly why the override field exists -- e.g. 20230413_firstsite's
    # Mag2_inclined trial). Using the folder-basename for those lookups
    # instead of `trial.recname` silently fetched the wrong/default
    # offset, shifting this epoch's window and both dropping and admitting
    # the wrong units (confirmed on real data).
    folder_to_trial_recname = {t.folder: t.recname for t in cfg.trials}
    for freq in data.keys():
        for folder in data[freq].keys():
            rec_label = os.path.basename(folder)
            metadata_recname = folder_to_trial_recname.get(folder, rec_label)
            offset = get_MM_offset(cat_df, metadata_recname)
            _, period_crossings = data[freq][folder]
            full_crossings = (np.asarray(period_crossings) + offset).astype(np.int64)
            trial_skips = next((t.skips for t in cfg.trials if t.recname == metadata_recname), 0)
            nwb_epochs.append({
                "rec": rec_label,
                "stim_type": "magnetic",
                "frequency": float(freq),
                "start_time": float(full_crossings[0]) / AP_SR,
                "stop_time": float(full_crossings[-1]) / AP_SR,
                "period_crossings": full_crossings,
                "skips": trial_skips,
                "local_offset_samples": int(offset),
            })

    # --- Auxiliary stimuli ---
    for aux_cfg in cfg.auxiliary_stimuli:
        handler = _HANDLERS.get(aux_cfg.kind)
        if handler is None:
            raise ValueError(f"Unknown aux stimulus kind: {aux_cfg.kind}")
        if aux_cfg.recname not in contingency_d:
            print(f"  WARNING: aux recname '{aux_cfg.recname}' not in contingency_d, skipping")
            continue
        handler(cfg, aux_cfg, contingency_d, aux_d, udf, cat_df, df_l, nwb_epochs)

    modulation_df = pd.concat(df_l).reset_index(drop=True)

    nwbfile = nwb_io.create_nwbfile(cfg)
    # label_column="KSLabel": matches this paradigm's own good-filter
    # (`udf.KSLabel == "good"` above) -- see write_units_and_spikes's
    # label_column docstring for why this must be stated explicitly rather
    # than auto-detected.
    nwb_io.write_units_and_spikes(
        nwbfile, all_sts_nwb, udf_nwb, sampling_rate=AP_SR, label_column="KSLabel")
    nwb_io.write_epochs_table(nwbfile, nwb_epochs, sampling_rate=AP_SR)
    nwb_io.write_nwbfile(nwbfile, cfg.nwb_path())

    from pathlib import Path
    from pipeline.diagnostics.processing import plot_recording_timeline
    diag_dir = Path(cfg.data_dir).parent / "figs" / "processing"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Read diagnostics input back from the just-written NWB file rather than
    # the in-memory modulation_df above -- same "traceability without
    # recomputation" check as openephys.py. A diagnostics-plotting bug must
    # never block the processing stage itself, so any failure here is
    # logged and skipped rather than raised.
    try:
        io_r, nwbfile_r = nwb_io.read_nwbfile(cfg.nwb_path())
        modulation_df_nwb = nwb_io.build_modulation_frame(nwbfile_r, good_only=cfg.good)
        plot_recording_timeline(cfg, nwbfile_r, modulation_df_nwb, diag_dir, cat_df=cat_df)
        io_r.close()
    except Exception as exc:
        print(f"  WARNING: NWB-sourced diagnostics failed ({exc}); skipping timeline plot")
