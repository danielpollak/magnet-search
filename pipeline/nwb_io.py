"""
Core NWB read/write library for the magnet_search pipeline.

Replaces the per-experiment trio of pickles (`{name}_processing.pickle`,
`MM_{name}.pickle`, `{name}_analysis.pickle`) with a single `{name}.nwb`
file. See .claude/plans (NWB replatform plan) for the full design rationale.

Phase 1/2 scope (this file): `Units` + spikes, a `stimulus_epochs`
`TimeIntervals` table (holding Schmitt-trigger period-crossing arrays so
phase/period are derived on demand instead of being stored per spike), and
Fourier-analysis results tables. Phase 4 adds the ophys (engert/medaka)
branch: `PlaneSegmentation` (all suite2p ROIs, unfiltered) + `RoiResponseSeries`
(fluorescence traces), and a `fit_Fourier()`-sourced analog of the Fourier
results tables (see `write_imaging_fourier_results` — `write_fourier_results`
itself is tightly coupled to `fit_fourier_sig()`'s `log_dict` shape, which
`fit_Fourier()` has no equivalent of, so imaging gets its own writer
targeting the *same* table schema; `read_fourier_results_as_full_fourier_df`
is shared unchanged).

Key design decision — sample-domain round-tripping
----------------------------------------------------
`Units.spike_times` and `stimulus_epochs.period_crossings` are stored in
seconds (the NWB convention), but the legacy `period`/`phase` arithmetic
(`np.sum(spk > period_crossings)`, `correct_theta`) is integer sample-domain
arithmetic. To reproduce it bit-for-bit, `build_modulation_frame` converts
back to samples via `round(seconds * sampling_rate)` before doing any
period/phase math — never in the float-seconds domain directly. This
round-trip is exact for any realistic recording length (float64 represents
integers exactly up to 2**53 samples, i.e. tens of thousands of hours at
30 kHz), so no precision is lost relative to keeping the original integer
sample arrays.
"""
import glob
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dateutil.tz import tzlocal

from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable
from pynwb import NWBFile, NWBHDF5IO
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
from pynwb.ophys import ImagingPlane, ImageSegmentation, PlaneSegmentation, OpticalChannel, Fluorescence, RoiResponseSeries

AP_SR = 30_000  # default NPIX/spike sampling rate (matches fit_fourier_sig's sr=30_000 default)

GZIP_LEVEL = 4  # empirically: level 4 vs 9 made no measurable difference on
                # real spike_times/fluorescence data (both float64/float32
                # arrays with little byte-level redundancy for gzip to
                # exploit) -- 4 is the cheaper choice for identical output.


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

def create_nwbfile(cfg, session_start_time=None):
    """Build a fresh, empty NWBFile for `cfg`, filling required metadata.

    A `Subject` is attached only if `cfg.subject_id` is set — kept optional
    so the pilot doesn't require editing every experiment YAML up front
    (subject/date backfill from `aggregate.py`'s ANNOT_D is a later, separate
    task; NWBFile itself doesn't require a Subject).
    """
    if session_start_time is None:
        session_start_time = _infer_session_start_time(cfg)

    nwbfile = NWBFile(
        session_description=f"{cfg.paradigm} experiment {cfg.name}",
        identifier=cfg.name,
        session_start_time=session_start_time,
        lab="Wagenaar Lab",
        institution="Caltech",
    )

    subject_id = getattr(cfg, "subject_id", "") or ""
    if subject_id:
        species = getattr(cfg, "species", "") or None
        nwbfile.subject = Subject(subject_id=subject_id, species=species)

    return nwbfile


def _infer_session_start_time(cfg):
    """Best-effort session start time from an OpenEphys settings.xml, falling
    back to a fixed placeholder. Exact wall-clock time is metadata only — it
    is never used in any numeric computation downstream — so a placeholder
    is safe when settings.xml can't be found.
    """
    try:
        from magpyneto2.nwb_utils import get_openephys_start_time
        search_root = getattr(cfg, "aggregated_path", "") or getattr(cfg, "session_path", "")
        candidates = []
        if search_root and os.path.isdir(search_root):
            for settings_path in glob.glob(
                    os.path.join(search_root, "**", "settings.xml"), recursive=True):
                dt = get_openephys_start_time(settings_path)
                if dt is not None:
                    candidates.append(dt)
        if candidates:
            dt = min(candidates)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tzlocal())
            return dt
    except Exception:
        pass
    return datetime(2000, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Sampling-rate bookkeeping (scratch, so period_crossings/spike_times can be
# exactly round-tripped back to integer sample indices)
# ---------------------------------------------------------------------------

def _set_sampling_rate(nwbfile, sampling_rate):
    if "sampling_rate" not in nwbfile.scratch:
        nwbfile.add_scratch(
            np.array([sampling_rate], dtype=np.float64),
            name="sampling_rate",
            notes="Samples/second used for Units.spike_times and "
                  "stimulus_epochs.period_crossings sample-domain round-trip",
        )


def get_sampling_rate(nwbfile):
    if "sampling_rate" in nwbfile.scratch:
        return float(np.atleast_1d(np.asarray(nwbfile.scratch["sampling_rate"].data))[0])
    return AP_SR


# ---------------------------------------------------------------------------
# Units + spikes
# ---------------------------------------------------------------------------

def write_units_and_spikes(nwbfile, all_sts, cluster_info=None, sampling_rate=AP_SR, rec="",
                            label_column=None):
    """Add one row per Kilosort cluster to `nwbfile.units`.

    Parameters
    ----------
    all_sts : dict {cluster_id (int): np.ndarray of spike sample indices}
        Full-session spike trains, in the same sample-index domain used by
        `process_raw_data_NPIX`. Pass ALL clusters (e.g.
        `ksr.spikesbycluster(label=None)`), not just `good` ones — the
        `kilosort_label` column lets `cfg.good` filtering happen at analysis
        time instead of being baked in at write time, so no unit's spike
        train is silently dropped before ever reaching the NWB file.
    cluster_info : pd.DataFrame, optional
        Output of `magpyneto2.get_cluster_info()` — used to attach
        `kilosort_label`/`channel` columns.
    sampling_rate : float
        Samples/second for `all_sts`; recorded in scratch (see
        `get_sampling_rate`) so it can be inverted exactly later.
    label_column : str, optional
        Which `cluster_info` column is authoritative for `kilosort_label`.
        A real cluster_info.tsv commonly has BOTH `KSLabel` (raw classifier
        output) and `group` (phy-curated, defaults to a copy of `KSLabel`
        until a human overrides it) — after curation the two can genuinely
        disagree per-cluster, so which one is "the" good/mua label is a
        property of what that paradigm's OWN legacy code filtered on, not
        something safely auto-detectable. Pass the column explicitly:
        `"KSLabel"` for openephys/openephys_multistim (matches their
        `udf.KSLabel == "good"`); `"group"` for gutfreund (matches
        `get_all_spiketrains`'s `unit_df.group == label`, which mirrors
        `ephysio.kilosortIO.Reader`'s own group-tsv-overrides-KSLabel
        precedence). Leave as `None` to fall back to the old auto-detect
        (KSLabel if present, else group) — kept only for callers that
        haven't been updated to pass this explicitly.
    rec : str
        Which recording this Kilosort catalog belongs to. Leave as "" (the
        default) for paradigms with ONE shared/concatenated catalog across
        all recs in the file (openephys, openephys_multistim,
        spikeglx_direct) — such units apply to every epoch that time-window-
        matches, exactly like Phase 2. Pass an actual rec name for paradigms
        where EACH recording has its OWN independent Kilosort sorting
        (gutfreund) — cluster ids are then only unique *within* a rec, so
        `build_modulation_frame` must scope unit-lookup by rec as well as by
        time window, and this function may be called multiple times (once
        per recording) to add each one's units to the same file. The NWB row
        `id` is always auto-assigned (not `cluster_id`) so repeated calls
        with overlapping cluster ids across different recs never collide;
        the original Kilosort id is preserved in the `cluster_id` column.
    """
    if nwbfile.units is None:
        nwbfile.add_unit_column(
            "kilosort_label", "Kilosort cluster_info.tsv KSLabel/group (good/mua/noise)")
        nwbfile.add_unit_column(
            "channel", "probe channel from cluster_info.tsv (NaN if unknown)")
        nwbfile.add_unit_column(
            "cluster_id", "original Kilosort cluster id (unique only within `rec` "
                           "if `rec` is non-empty; matches legacy modulation_df.id)")
        nwbfile.add_unit_column(
            "rec", "recording this unit's sorting belongs to; '' if the unit is "
                   "shared across every rec in this file (concatenated-catalog "
                   "paradigms) rather than scoped to one independent sorting")

    info_by_id = {}
    if cluster_info is not None:
        id_col = "cluster_id" if "cluster_id" in cluster_info.columns else "id"
        for _, row in cluster_info.iterrows():
            info_by_id[int(row[id_col])] = row

    label_col = None
    if cluster_info is not None:
        if label_column is not None:
            label_col = label_column if label_column in cluster_info.columns else None
        elif "KSLabel" in cluster_info.columns:
            label_col = "KSLabel"
        elif "group" in cluster_info.columns:
            label_col = "group"

    for cluster_id, st_samples in all_sts.items():
        st_samples = np.asarray(st_samples)
        info = info_by_id.get(int(cluster_id))
        label = str(info[label_col]) if info is not None and label_col is not None else ""
        channel = float(info["ch"]) if info is not None and "ch" in info else np.nan
        nwbfile.add_unit(
            spike_times=(st_samples / sampling_rate).astype(np.float64),
            kilosort_label=label,
            channel=channel,
            cluster_id=int(cluster_id),
            rec=rec,
        )

    _set_sampling_rate(nwbfile, sampling_rate)
    return nwbfile.units


# ---------------------------------------------------------------------------
# Stimulus epochs (rec/frequency/window plus the full period-crossing array,
# not just [start, end])
# ---------------------------------------------------------------------------

def write_epochs_table(nwbfile, epochs, sampling_rate=AP_SR):
    """Add one row per stimulus epoch ("rec") to a custom `stimulus_epochs`
    TimeIntervals table.

    Parameters
    ----------
    epochs : list of dict, each with keys:
        rec              : str
        stim_type        : str  ("magnetic" | "visual_gratings" |
                                  "white_noise" | "oddball" | "visual_bars")
        frequency        : float
        start_time       : float (s)
        stop_time        : float (s)
        period_crossings : np.ndarray[int]  Schmitt-trigger crossing sample
                            indices, same sample domain as Units.spike_times
                            pre-division (this function divides by
                            `sampling_rate` before storing).
        orientation_deg  : float, optional (default NaN)
        skips            : int, optional (default 0)
        local_offset_samples : int, optional (default 0)
            The constant added to convert this epoch's *recording-local*
            period-crossing/spike samples into the *aggregated* (whole-
            catalog) domain that `Units.spike_times`/`period_crossings` are
            stored in (for openephys, this is `get_MM_offset(cat_df, rec)` —
            i.e. `beginning_time`). `build_modulation_frame` subtracts it
            back out so its `spk` column matches the legacy
            `modulation_df`'s per-recording-local convention exactly, even
            though `period`/`phase` (shift-invariant) never needed it.
        phase_method : str, optional (default "crossings")
            "crossings" (default): period/phase are derived from
            `period_crossings` via `compute_period_at_samples`/
            `correct_theta_at_samples` (openephys, spikeglx_direct — anywhere
            the legacy code itself detected real Schmitt-trigger crossings at
            integer sample resolution).
            "arithmetic": period/phase are derived directly from
            `floor(spk_seconds / period_s)` / `(spk_seconds % period_s) /
            period_s * 2pi` where `period_s = 1/frequency` (gutfreund,
            openephys_multistim white_noise single-window) — used when the
            legacy formula operates on continuous float seconds rather than
            integer sample crossings, so routing it through the sample-
            quantized `period_crossings` machinery would introduce rounding
            drift the original computation never had (accumulating over many
            periods, e.g. for frequencies that don't evenly divide the
            sampling rate). `period_crossings` is still stored (as
            `[start_time, stop_time]` in samples) for dual-write
            completeness, but is not used for reconstruction in this mode.
            "gratings" (openephys_multistim visual_gratings): period is
            `magpyneto2.utils.get_condition_array(spk, period_crossings)`
            (called directly, not reimplemented, since it has quirky
            insertion-collision behavior under `np.searchsorted` that isn't
            safely reproducible via a generic crossings-count formula);
            `period_crossings` here holds the *per-trial* markers. Phase is
            `correct_theta_at_samples` using a SEPARATE, wider
            `phase_crossings` array (the *global*, whole-grating-stimulus
            marker sequence) rather than `period_crossings`.
            "stitched_floor" (openephys_multistim white_noise multi-window
            legacy formula, visual_bars): spikes are first re-indexed from
            multiple disjoint real windows (`window_starts`/`window_stops`/
            `window_offsets`) onto one synthetic timeline, then period/phase
            use the same floor/modulo formulas as "arithmetic" on that
            synthetic time.
            "stitched_crossings_unnorm" (openephys_multistim white_noise
            multi-window non-legacy formula, oddball): same synthetic-
            timeline stitching as "stitched_floor", but period is a
            crossings-style count against `synthetic_period_markers` (a
            purely synthetic marker sequence, unrelated to any real
            detected crossing) and phase is the *unnormalized*
            `(synthetic_t % period_s) * 2pi` (no `/period_s`) — both quirks
            of the original scripts, reproduced as-is rather than "fixed".
        period_frequency : float, optional
            Overrides `frequency` as the divisor for `period_s = 1/
            period_frequency` in "arithmetic"/"stitched_floor" modes only
            (frequency itself is still stored/reported unchanged in the
            `freq` column) — bug-compat for `wn_period_freq` (see
            CLAUDE.md's schema notes on white_noise).
        phase_crossings : np.ndarray[int], optional
            Only used by "gratings" — the global marker sequence for phase
            reconstruction (see above). Same sample-domain convention as
            `period_crossings`.
        window_starts, window_stops, window_offsets : np.ndarray[int], optional
            Only used by "stitched_*" modes — parallel arrays (same length),
            all in the same sample domain as Units.spike_times, describing
            each disjoint real window to pull spikes from
            (`window_starts[i]`, `window_stops[i]`, both strict/exclusive
            bounds matching the legacy `>`/`<` comparisons) and the sample
            offset added after subtracting `window_starts[i]` to place that
            window's spikes on the shared synthetic timeline.
        synthetic_period_markers : np.ndarray[float], optional
            Only used by "stitched_crossings_unnorm" — a marker sequence
            already in the *synthetic*-timeline seconds domain (not a real
            recording time), counted via `compute_period_at_samples`-style
            `sum(t > markers)`.
    """
    table = TimeIntervals(
        name="stimulus_epochs",
        description="One row per recording/trial epoch. period_crossings "
                     "lets phase/period be derived on demand (see "
                     "correct_theta_at_samples/compute_period_at_samples) "
                     "instead of being stored per spike.",
    )
    table.add_column("rec", "recording/trial-block name (join key, matches legacy `rec`)")
    table.add_column("stim_type", "magnetic | visual_gratings | white_noise | oddball | visual_bars")
    table.add_column("frequency", "stimulus frequency, Hz")
    table.add_column("orientation_deg", "grating/bar orientation in degrees, NaN if n/a")
    table.add_column("skips", "number of leading period crossings skipped")
    table.add_column(
        "local_offset_seconds",
        "constant subtracted from Units.spike_times (after converting to "
        "samples) to recover recording-local spike times matching the "
        "legacy modulation_df.spk convention; 0.0 if spikes are already "
        "recording-local for this paradigm",
    )
    table.add_column(
        "period_crossings",
        "Schmitt-trigger period-crossing times (s); multiply by the file's "
        "sampling_rate scratch value and round to int64 to recover exact "
        "sample indices for phase/period arithmetic. Not used for "
        "reconstruction when phase_method=='arithmetic' (still populated, "
        "with just [start_time, stop_time], for dual-write completeness).",
        index=True,
    )
    table.add_column(
        "phase_method",
        "'crossings' (default) | 'arithmetic' | 'gratings' | 'stitched_floor' "
        "| 'stitched_crossings_unnorm' -- see write_epochs_table's docstring",
    )
    table.add_column(
        "period_frequency",
        "overrides `frequency` as the period_s=1/period_frequency divisor "
        "for 'arithmetic'/'stitched_floor' modes; NaN means 'use frequency'",
    )
    table.add_column(
        "min_spikes_on_full_session",
        "if True, min_spikes is checked against the unit's full-session "
        "spike count rather than its windowed count (gutfreund only -- "
        "see build_modulation_frame's docstring); the windowed count then "
        "only needs to be non-empty",
    )
    table.add_column(
        "truncate_spike_samples",
        "if True, spike sample indices are computed via truncation "
        "(np.int64 cast) instead of rounding -- reproduces "
        "spikeglx_direct.py's own `(st * AP_sr).astype(int)`, which can be "
        "off by one sample vs. rounding when AP_sr isn't an exact integer "
        "(see build_modulation_frame's docstring)",
    )
    table.add_column(
        "period_crossing_inclusive",
        "if True, period = count of crossings <= spike sample (matches "
        "spikeglx_direct's `periods[prev:curr] = i` slice loop) instead of "
        "the default count of crossings < spike sample -- only differs "
        "when a spike sample lands exactly on a crossing value (see "
        "compute_period_at_samples's docstring)",
    )
    table.add_column(
        "gratings_group",
        "'gratings'-only: groups all orientation-epochs from one "
        "visual_gratings recname -- a unit is only included in ANY epoch "
        "of the group if it passes min_spikes in EVERY epoch of the group "
        "(matches legacy's all-or-nothing per-recname filter; see "
        "_gratings_group_exclusions). '' for every other phase_method.",
    )
    # These four are only meaningful for a subset of phase_methods
    # ("gratings"/"stitched_*") -- added only if at least one epoch in THIS
    # file actually uses them. If every epoch left them at the empty-array
    # default (e.g. a pure openephys file with no aux stimuli), the whole
    # ragged column would be empty across every row and HDMF can't infer a
    # dtype for it ("Cannot infer dtype of empty list or tuple") -- adding
    # the column unconditionally would break every paradigm that doesn't use
    # a given mode, not just leave it harmlessly unused.
    def _any_nonempty(key):
        return any(len(ep[key]) for ep in epochs if ep.get(key) is not None)

    has_phase_crossings = _any_nonempty("phase_crossings")
    has_window_starts = _any_nonempty("window_starts")
    has_synthetic_markers = _any_nonempty("synthetic_period_markers")

    if has_phase_crossings:
        table.add_column(
            "phase_crossings",
            "'gratings'-only: global marker sequence (s) for phase "
            "reconstruction, separate from period_crossings' per-trial markers",
            index=True,
        )
    if has_window_starts:
        table.add_column(
            "window_starts", "'stitched_*'-only: per-window start sample (s), strict/exclusive lower bound", index=True)
        table.add_column(
            "window_stops", "'stitched_*'-only: per-window stop sample (s), strict/exclusive upper bound", index=True)
        table.add_column(
            "window_offsets",
            "'stitched_*'-only: sample offset (s-equivalent, i.e. divided by "
            "sampling_rate like everything else here) added after subtracting "
            "window_starts[i], placing that window's spikes on the shared "
            "synthetic timeline", index=True)
    if has_synthetic_markers:
        table.add_column(
            "synthetic_period_markers",
            "'stitched_crossings_unnorm'-only: marker sequence already in the "
            "synthetic-timeline seconds domain, counted via sum(t > markers)",
            index=True)

    for ep in epochs:
        # NOT forced to int64 here: "gratings" mode's period_crossings are
        # genuine sub-sample-precision floats (from a cross-clock
        # interpolation, `ldr.shifttime`) and legacy's `get_condition_array`
        # call uses them unrounded -- truncating to int64 at write time would
        # lose exactly the precision that matters for its searchsorted-based
        # insertion-order counting (verified empirically to change real
        # counts, not just a theoretical rare edge case). Every other mode's
        # crossings genuinely are integers already, so this is a no-op there.
        crossings_sec = np.asarray(ep["period_crossings"], dtype=np.float64) / sampling_rate
        row_kwargs = dict(
            start_time=float(ep["start_time"]),
            stop_time=float(ep["stop_time"]),
            rec=ep["rec"],
            stim_type=ep.get("stim_type", "magnetic"),
            frequency=float(ep["frequency"]),
            orientation_deg=float(ep.get("orientation_deg", np.nan)),
            skips=int(ep.get("skips", 0)),
            local_offset_seconds=float(ep.get("local_offset_samples", 0)) / sampling_rate,
            period_crossings=crossings_sec,
            phase_method=ep.get("phase_method", "crossings"),
            period_frequency=float(ep.get("period_frequency", np.nan)),
            min_spikes_on_full_session=bool(ep.get("min_spikes_on_full_session", False)),
            truncate_spike_samples=bool(ep.get("truncate_spike_samples", False)),
            period_crossing_inclusive=bool(ep.get("period_crossing_inclusive", False)),
            gratings_group=ep.get("gratings_group", ""),
        )
        if has_phase_crossings:
            row_kwargs["phase_crossings"] = (
                np.asarray(ep.get("phase_crossings", []), dtype=np.int64) / sampling_rate)
        if has_window_starts:
            # NOT forced to int64: window_starts/stops (e.g. `t_down` from a
            # cross-clock `ldr.shifttime` interpolation) can be genuinely
            # fractional, and legacy's stitching arithmetic keeps that
            # fractional part through to the synthetic timeline rather than
            # rounding it away first (see _ragged_samples' round_to_int=False).
            row_kwargs["window_starts"] = np.asarray(ep.get("window_starts", []), dtype=np.float64) / sampling_rate
            row_kwargs["window_stops"] = np.asarray(ep.get("window_stops", []), dtype=np.float64) / sampling_rate
            row_kwargs["window_offsets"] = np.asarray(ep.get("window_offsets", []), dtype=np.float64) / sampling_rate
        if has_synthetic_markers:
            row_kwargs["synthetic_period_markers"] = np.asarray(
                ep.get("synthetic_period_markers", []), dtype=np.float64)
        # Ragged columns must be 1-D per row -- a stray (2, 1)-shaped array
        # (e.g. from forgetting to `float()` a 1-element numpy array before
        # building it) writes fine here but fails later, cryptically, when
        # HDMF concatenates it against other rows' plain 1-D arrays
        # ("inhomogeneous shape" -- confirmed on real data). Catch it at the
        # source epoch instead.
        for _ragged_col in ("period_crossings", "phase_crossings", "window_starts",
                             "window_stops", "window_offsets", "synthetic_period_markers"):
            if _ragged_col in row_kwargs and np.asarray(row_kwargs[_ragged_col]).ndim != 1:
                raise ValueError(
                    f"epoch rec={ep.get('rec')!r}: '{_ragged_col}' must be 1-D, got shape "
                    f"{np.asarray(row_kwargs[_ragged_col]).shape} -- check for an un-`float()`ed "
                    f"1-element array in the paradigm module's epoch dict construction")
        table.add_row(**row_kwargs)

    nwbfile.add_time_intervals(table)
    _set_sampling_rate(nwbfile, sampling_rate)
    return table


# ---------------------------------------------------------------------------
# On-demand phase/period reconstruction (replaces per-spike storage)
# ---------------------------------------------------------------------------

def _period_crossings_samples(epoch_row, sampling_rate):
    crossings_sec = np.asarray(epoch_row["period_crossings"])
    return np.round(crossings_sec * sampling_rate).astype(np.int64)


def _period_crossings_float_samples(epoch_row, sampling_rate):
    """Unrounded counterpart of `_period_crossings_samples` -- needed by
    "gratings" mode, whose `period_crossings` carry genuine sub-sample
    precision that `get_condition_array`'s insertion-order counting is
    sensitive to (unlike every other mode, whose crossings are already
    exact integers, for which rounding here is a no-op)."""
    return np.asarray(epoch_row["period_crossings"]) * sampling_rate


def _ragged_samples(epoch_row, column, sampling_rate, round_to_int=True):
    """Read one of the optional per-epoch ragged sample-domain columns.
    Returns an empty array if the epoch has none.

    `round_to_int=True` (default) for `phase_crossings` -- legacy itself
    rounds its analogous marker array (`np.round(t_up).astype(int)`) before
    building the dense theta array it indexes, so rounding here is faithful,
    not lossy. `round_to_int=False` for `window_starts`/`window_stops`/
    `window_offsets` -- these can be genuinely fractional (e.g. `t_down` from
    a cross-clock `ldr.shifttime` interpolation), and legacy's own stitching
    arithmetic (`spikes - window_start + offset`) keeps that fractional part
    all the way through rather than rounding it away first.
    """
    values = np.asarray(epoch_row.get(column, []))
    if round_to_int:
        return np.round(values * sampling_rate).astype(np.int64)
    return values * sampling_rate


def _stitch_synthetic_samples(spike_samples, window_starts, window_stops, window_offsets):
    """Re-index spikes from N disjoint real windows onto one synthetic
    timeline: for window i, spikes in (window_starts[i], window_stops[i])
    (strict/exclusive, matching every "stitched" legacy handler's `>`/`<`)
    map to `spike - window_starts[i] + window_offsets[i]`. Windows are
    processed in order and concatenated, matching legacy's per-window-loop-
    then-concatenate order (`np.concatenate([...])`) exactly -- callers
    should not assume the result is sorted.
    """
    if len(window_starts) == 0:
        return np.array([], dtype=np.int64)
    out = []
    for w_start, w_stop, w_offset in zip(window_starts, window_stops, window_offsets):
        in_window = spike_samples[(spike_samples > w_start) & (spike_samples < w_stop)]
        out.append(in_window - w_start + w_offset)
    return np.concatenate(out)


def correct_theta_at_samples(spike_samples, period_crossings):
    """Vectorized equivalent of `magpyneto2.utils.correct_theta`, evaluated
    only at `spike_samples` (never materializes the full dense theta array).

    Exactly reproduces:
        theta_corrected = np.zeros(length)
        for i in range(1, len(period_crossings)):
            last, curr = period_crossings[i-1], period_crossings[i]
            theta_corrected[last:curr] = np.linspace(0, 2*np.pi, num=curr-last)
        return theta_corrected[spike_samples]

    Spikes outside [period_crossings[0], period_crossings[-1]) get phase 0.0,
    matching the legacy dense array's untouched-zeros behavior there.
    """
    spike_samples = np.asarray(spike_samples)
    crossings = np.asarray(period_crossings)
    phase = np.zeros(len(spike_samples), dtype=np.float64)
    if len(crossings) < 2:
        return phase

    # bin_idx = i such that crossings[i-1] <= s < crossings[i]
    bin_idx = np.searchsorted(crossings, spike_samples, side="right")
    valid = (bin_idx >= 1) & (bin_idx < len(crossings))
    last = crossings[bin_idx[valid] - 1]
    curr = crossings[bin_idx[valid]]
    seg_len = curr - last
    # np.linspace(0, 2*pi, num=seg_len)[k] == k * (2*pi / (seg_len - 1)) for seg_len > 1
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where(seg_len > 1, 2 * np.pi / np.maximum(seg_len - 1, 1), 0.0)
    offset = spike_samples[valid] - last
    phase[valid] = offset * step
    return phase


def compute_period_at_samples(spike_samples, period_crossings, inclusive=False):
    """Vectorized equivalent of `np.sum(spk > period_crossings)` per spike
    (count of crossings strictly less than each spike sample) when
    `inclusive=False` (the default, matching openephys/openephys_multistim's
    own period computation).

    `inclusive=True` instead counts crossings <= each spike sample, matching
    spikeglx_direct's own `periods[prev:curr] = i` slice-assignment loop
    (index `prev` -- i.e. == crossings[i-1] exactly -- falls INSIDE bucket i,
    not bucket i-1). The two conventions only disagree when a spike sample
    lands EXACTLY on a crossing value -- vanishingly rare for real
    high-resolution analog crossings, but confirmed on real data to matter
    for spikeglx_direct once `truncate_spike_samples` (see
    build_modulation_frame) makes exact integer collisions non-negligible.
    """
    spike_samples = np.asarray(spike_samples)
    crossings = np.asarray(period_crossings)
    side = "right" if inclusive else "left"
    return np.searchsorted(crossings, spike_samples, side=side).astype(np.int64)


def _gratings_group_exclusions(epochs, units_df, sr, min_spikes):
    """visual_gratings' own legacy loop
    (`if len(trial_df_l) == aux_cfg.n_orientations: df_l.extend(trial_df_l)`)
    only includes a unit's data for ANY orientation of one grating recname
    if it passed `len(spk) > 50` in EVERY orientation of that recname --
    confirmed on real data: per-orientation-independent filtering (what a
    naive per-epoch loop does) silently included extra units that legacy
    excludes as a block. Pre-compute, per `gratings_group` (shared across
    all orientation-epochs of one recname), the set of unit cluster_ids
    that fail this all-or-nothing criterion, so the main per-epoch loop
    below can additionally exclude them.

    Returns {gratings_group: set of excluded cluster_ids}, empty if this
    file has no "gratings" epochs.
    """
    exclusions = {}
    if "phase_method" not in epochs.columns or "gratings_group" not in epochs.columns:
        return exclusions
    gratings_epochs = epochs[epochs["phase_method"] == "gratings"]
    if len(gratings_epochs) == 0:
        return exclusions

    id_col = "cluster_id" if "cluster_id" in units_df.columns else None
    unit_ids = (units_df[id_col].values if id_col else units_df.index.values)
    spike_sample_arrays = [
        np.round(np.asarray(st) * sr).astype(np.int64) for st in units_df["spike_times"].values
    ]

    for group, group_epochs in gratings_epochs.groupby("gratings_group"):
        qualifies = np.ones(len(units_df), dtype=bool)
        for _, ep in group_epochs.iterrows():
            c_lo = float(ep["start_time"]) * sr
            c_hi = float(ep["stop_time"]) * sr
            counts = np.array([
                int(np.sum((s > c_lo) & (s < c_hi))) for s in spike_sample_arrays
            ])
            qualifies &= counts > min_spikes  # gratings: exclusive-of-min_spikes
        exclusions[group] = set(int(u) for u in unit_ids[~qualifies])
    return exclusions


def build_modulation_frame(nwbfile, rec=None, good_only=True, min_spikes=50):
    """Reconstruct a `modulation_df`-shaped, in-memory-only DataFrame
    (columns: period, spk, phase, freq, id, rec) from the NWB file's `Units`
    + `stimulus_epochs` tables — exactly the input `fit_fourier_sig()` expects.

    This is never persisted; it is a transient adapter so the (unmodified)
    `fit_fourier_sig()` machinery keeps working against NWB-backed data. See
    the NWB replatform design doc, "Is modulation_df eliminated?".
    """
    sr = get_sampling_rate(nwbfile)
    epochs = nwbfile.intervals["stimulus_epochs"].to_dataframe()
    if rec is not None:
        epochs = epochs[epochs["rec"] == rec]

    units_df = nwbfile.units.to_dataframe()
    if good_only and "kilosort_label" in units_df.columns:
        units_df = units_df[units_df["kilosort_label"] == "good"]

    has_rec_scoping = "rec" in units_df.columns
    gratings_exclusions = _gratings_group_exclusions(epochs, units_df, sr, min_spikes)

    rows = []
    for _, epoch in epochs.iterrows():
        phase_method = epoch.get("phase_method", "crossings")

        # local_offset: Units.spike_times/period_crossings are stored in the
        # *aggregated* (whole-catalog) domain; legacy modulation_df.spk is
        # *recording-local* (all_sts - beginning_time). period/phase are
        # shift-invariant so they're fine either way, but spk itself must be
        # converted back — see write_epochs_table's local_offset_seconds.
        # Not used by "stitched_*" modes: their `spk` values are already a
        # synthetic re-indexed timeline unrelated to the aggregation offset.
        local_offset = int(round(float(epoch.get("local_offset_seconds", 0.0)) * sr))

        # Scope candidate units to this epoch's recording. "" means "shared
        # across every rec in this file" (concatenated-catalog paradigms,
        # e.g. openephys) — those units are always candidates. A non-empty
        # rec means "this unit's Kilosort sorting is local to that one
        # recording" (e.g. gutfreund) — only match epochs for that same rec,
        # since cluster ids (and even the time-window numeric ranges, which
        # restart near 0 for every independent recording) are NOT globally
        # unique/disjoint across such recordings.
        if has_rec_scoping:
            epoch_units = units_df[
                (units_df["rec"] == "") | (units_df["rec"] == epoch["rec"])]
        else:
            epoch_units = units_df

        def period_s_of(epoch):
            override = epoch.get("period_frequency", np.nan)
            freq = override if (override is not None and not np.isnan(override)) else epoch["frequency"]
            return 1.0 / float(freq)

        # min_spikes-threshold convention differs by phase_method, matching
        # each paradigm's own legacy code exactly (confirmed on real data --
        # see the NWB replatform plan/session notes): openephys/
        # openephys_multistim magnetic trials (via process_raw_data_NPIX)
        # and spikeglx_direct both use `if len(st) < 50: continue` --
        # INCLUSIVE of exactly `min_spikes`. gutfreund (`if len(st) <= 50:
        # continue`) and every openephys_multistim aux-stimulus handler
        # (`if len(concat_spks) > 50:`) instead require STRICTLY more than
        # `min_spikes` -- EXCLUSIVE of exactly `min_spikes`. Getting this
        # wrong silently included/excluded whole units whose window spike
        # count landed exactly on the boundary.
        exclusive_min_spikes = phase_method != "crossings"

        # gutfreund.py's own min_spikes check is on the unit's FULL-SESSION
        # spike count (`all_sts_d.items()`'s `st`, BEFORE windowing to
        # [on, off]) -- `if len(st) <= 50: continue` -- with the post-window
        # count only checked for non-emptiness (`if len(st_samples) == 0:
        # continue`), not against min_spikes at all. Every other paradigm
        # using this same "single real window" branch (openephys_multistim's
        # WN single-window, also phase_method="arithmetic") instead checks
        # min_spikes on the WINDOWED count -- confirmed on real data that
        # these two "arithmetic" callers genuinely differ, so this can't be
        # inferred from phase_method alone; each epoch states it explicitly.
        min_spikes_on_full_session = bool(epoch.get("min_spikes_on_full_session", False))

        # --- "stitched_*": spikes come from N disjoint real windows,
        # re-indexed onto one synthetic timeline before any period/phase math
        # (openephys_multistim white_noise multi-window, oddball, visual_bars)
        if phase_method in ("stitched_floor", "stitched_crossings_unnorm"):
            # round_to_int=False: these can be genuinely fractional (see
            # write_epochs_table's docstring) -- do NOT round away precision
            # legacy's own stitching arithmetic never rounded away either.
            window_starts = _ragged_samples(epoch, "window_starts", sr, round_to_int=False)
            window_stops = _ragged_samples(epoch, "window_stops", sr, round_to_int=False)
            window_offsets = _ragged_samples(epoch, "window_offsets", sr, round_to_int=False)
            if len(window_starts) == 0:
                continue
            period_s = period_s_of(epoch)
            markers = np.asarray(epoch.get("synthetic_period_markers", []), dtype=np.float64)

            for unit_id, unit in epoch_units.iterrows():
                out_id = int(unit["cluster_id"]) if "cluster_id" in unit else int(unit_id)
                spike_samples = np.round(np.asarray(unit["spike_times"]) * sr).astype(np.int64)
                synthetic = _stitch_synthetic_samples(
                    spike_samples, window_starts, window_stops, window_offsets)
                if (len(synthetic) <= min_spikes if exclusive_min_spikes
                        else len(synthetic) < min_spikes):
                    continue
                spk_seconds = synthetic / sr

                if phase_method == "stitched_floor":
                    period_vals = np.floor(spk_seconds / period_s)
                    phase_vals = (spk_seconds % period_s) / period_s * 2 * np.pi
                else:
                    # "stitched_crossings_unnorm": period = sum(t > markers)
                    # (same convention as compute_period_at_samples, just on
                    # a synthetic marker sequence rather than real crossings);
                    # phase is deliberately UNNORMALIZED (no /period_s) --
                    # both reproduce quirks of the original scripts as-is.
                    period_vals = np.searchsorted(markers, spk_seconds, side="left").astype(np.int64)
                    phase_vals = (spk_seconds % period_s) * 2 * np.pi

                rows.append(pd.DataFrame({
                    "period": period_vals, "spk": spk_seconds, "phase": phase_vals,
                    "freq": epoch["frequency"], "id": out_id, "rec": epoch["rec"],
                }))
            continue

        # --- "crossings" / "arithmetic" / "gratings": single real window ---
        crossings = _period_crossings_samples(epoch, sr)
        if len(crossings) < 2:
            continue

        # Spike-inclusion window is the epoch's own start_time/stop_time (by
        # construction == crossings[0]/crossings[-1] for "crossings" and
        # "gratings", since that's what those paradigms set them to). For
        # openephys/spikeglx_direct, start_time/stop_time are themselves
        # exact integer-sample/sr values, so rounding after the seconds
        # round-trip is the precision-safe move (see this module's docstring
        # on sample-domain round-tripping). gutfreund's on/off and gratings'
        # period_markers[0]/[-1] are NOT integer sample counts (float
        # NIDAQ->AP / cross-clock-interpolated values) -- rounding them would
        # risk an off-by-one at spikes within half a sample of that boundary,
        # so both use the unrounded float threshold instead, exactly
        # reproducing legacy's `st[(st > on) & (st < off)]`-style
        # integer-vs-float comparisons.
        if phase_method in ("arithmetic", "gratings"):
            c_lo = float(epoch["start_time"]) * sr
            c_hi = float(epoch["stop_time"]) * sr
        else:
            c_lo = int(round(float(epoch["start_time"]) * sr))
            c_hi = int(round(float(epoch["stop_time"]) * sr))

        if phase_method == "gratings":
            phase_crossings = _ragged_samples(epoch, "phase_crossings", sr)
            period_markers_float = _period_crossings_float_samples(epoch, sr)

        # spikeglx_direct's own theta/periods-array lookup computes
        # `(st * AP_sr).astype(int)` -- TRUNCATION, not rounding -- on `st`
        # values that already round-tripped through a division by a
        # non-integer AP_sr (e.g. 30000.3). That round-trip can land a hair
        # below the true integer (e.g. 261935.99999999997), which truncation
        # then drops to 261935 instead of 261936 -- a real, reproducible
        # off-by-one in legacy's own output for ~3% of spikes on real data.
        # Every other "crossings" user has an exact-integer AP_sr (no float
        # error to begin with) so this never affects them either way -- but
        # must be truncated here, not rounded, to match spikeglx_direct bit
        # for bit rather than "fixing" its arithmetic.
        truncate = bool(epoch.get("truncate_spike_samples", False))
        excluded_ids = gratings_exclusions.get(epoch.get("gratings_group"), None) \
            if phase_method == "gratings" else None
        for unit_id, unit in epoch_units.iterrows():
            out_id = int(unit["cluster_id"]) if "cluster_id" in unit else int(unit_id)
            if excluded_ids is not None and out_id in excluded_ids:
                continue
            spike_times_s = np.asarray(unit["spike_times"])
            raw_samples = spike_times_s * sr
            spike_samples = (raw_samples.astype(np.int64) if truncate
                              else np.round(raw_samples).astype(np.int64))
            mask = (spike_samples > c_lo) & (spike_samples < c_hi)
            st = spike_samples[mask]
            # Reported `spk` must come from the ORIGINAL float seconds value,
            # not by re-deriving seconds from `st` (the sample-domain array
            # used for period/phase lookup) -- spikeglx_direct's own legacy
            # code reports `st_v` (untouched float division) separately from
            # `ap_indices` (the truncated lookup index); conflating the two
            # reintroduces the exact truncation error `truncate_spike_samples`
            # exists to reproduce ONLY in the lookup, not in `spk` itself.
            st_seconds_precise = spike_times_s[mask] - local_offset / sr
            if min_spikes_on_full_session:
                # min_spikes applies to the unit's full-session count, not
                # the windowed one -- windowed count only needs to be
                # non-empty (matches gutfreund.py's `if len(st_samples) ==
                # 0: continue` and spikeglx_direct.py's equivalent `.dropna()`
                # after indexing NaN-filled theta/periods arrays outside the
                # window -- same effect, no explicit count check needed
                # there). Exclusive/inclusive convention still follows
                # exclusive_min_spikes -- spikeglx_direct's own full-session
                # check (`if len(st) < 50: continue`) is INCLUSIVE, same as
                # "crossings", unlike gutfreund's EXCLUSIVE `<= 50`.
                full_excluded = (len(spike_samples) <= min_spikes if exclusive_min_spikes
                                 else len(spike_samples) < min_spikes)
                if full_excluded or len(st) == 0:
                    continue
            elif (len(st) <= min_spikes if exclusive_min_spikes
                    else len(st) < min_spikes):
                continue

            if phase_method == "arithmetic":
                # Bypass sample quantization entirely -- exactly reproduces
                # gutfreund.py's own `t // period_s` / `(t % period_s) /
                # period_s * 2pi`, operating on the same float seconds values
                # in the same order (see write_epochs_table's phase_method
                # docstring for why: routing this through integer-sample
                # period_crossings would accumulate rounding drift the
                # legacy computation never had).
                period_s = period_s_of(epoch)
                spk_seconds = st_seconds_precise
                period_vals = np.floor(spk_seconds / period_s)
                phase_vals = (spk_seconds % period_s) / period_s * 2 * np.pi
            elif phase_method == "gratings":
                # period: magpyneto2.utils.get_condition_array called
                # directly (not reimplemented -- see write_epochs_table's
                # docstring) on the per-trial period_markers_float (UNROUNDED
                # -- its searchsorted-based insertion-order counting is
                # sensitive to the sub-sample precision legacy's raw
                # `ldr.shifttime` markers actually have).
                # phase: correct_theta_at_samples over the *global*
                # phase_crossings (rounded, matching legacy's own
                # `np.round(t_up).astype(int)` before building its theta
                # array), not the per-trial markers. Both st and the marker
                # arrays are consistently in the aggregated domain, and both
                # computations are shift-invariant (ordering/relative-
                # position based), so no local_offset subtraction is needed
                # before calling them -- only the reported `spk` column
                # needs it, same as "crossings" mode.
                from magpyneto2.utils import get_condition_array
                period_vals = get_condition_array(st.astype(np.float64), period_markers_float)
                phase_vals = correct_theta_at_samples(st, phase_crossings)
                spk_seconds = st_seconds_precise
            else:
                period_vals = compute_period_at_samples(
                    st, crossings, inclusive=bool(epoch.get("period_crossing_inclusive", False)))
                phase_vals = correct_theta_at_samples(st, crossings)
                spk_seconds = st_seconds_precise

            rows.append(pd.DataFrame({
                "period": period_vals,
                "spk": spk_seconds,
                "phase": phase_vals,
                "freq": epoch["frequency"],
                "id": out_id,
                "rec": epoch["rec"],
            }))

    if not rows:
        return pd.DataFrame(columns=["period", "spk", "phase", "freq", "id", "rec"])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Fourier-analysis results (persists fit_fourier_sig()'s intermediates, which
# today are computed, used for diagnostics, and discarded every single run)
# ---------------------------------------------------------------------------

def write_fourier_results(nwbfile, fourier_df, log_dict):
    """Persist `fit_fourier_sig()`'s scalar results *and* its previously-
    discarded per-group/per-unit intermediates (ff_alt, fou0, fou_alt) into
    three linked DynamicTables under `nwbfile.processing["analysis"]`.

    No changes to `fit_fourier_sig()` itself are required: `log_dict` already
    carries every per-group intermediate needed, INCLUDING each group's own
    actual off-frequency bin count (`entry["M"]`) -- under the Q_frac (bin
    fraction) policy, a single `fit_fourier_sig()` call already produces
    different bin counts for different `(rec, freq)` groups and for the 1F
    vs. 2F harmonic of the same group (since the fraction is applied
    independently at each analyzed frequency), so there is no longer one
    scalar `Q` to pass in — each group's own `M` is used instead. `sigma`
    (per unit) and the null-distribution PDF/CDF/eps (Q-only-dependent) are
    cheaply recomputed here from public magpyneto2.statistics helpers,
    deduplicated per distinct `M` rather than duplicated per group/unit.

    Complex Fourier coefficients are split into real/imag columns (HDF5
    complex-dtype support is inconsistent across NWB client libraries).

    Safe to call multiple times on the same `nwbfile` with different
    (fourier_df, log_dict) — e.g. openephys_multistim's per-stimulus-type
    Q_frac overrides (mag_Q_frac/visual_Q_frac/WN_Q_frac): existing tables
    are reused/appended to rather than recreated, and an M already present
    in `null_distribution_models` from an earlier call is not duplicated.
    """
    from magpyneto2.statistics import (
        get_epsilon, get_sgm, normalized_Fourier_PDF,
        normalized_Fourier_PDF_corrected, normalized_Fourier_CDF_corrected,
    )

    if "analysis" in nwbfile.processing:
        module = nwbfile.processing["analysis"]
    else:
        module = nwbfile.create_processing_module(
            "analysis", "Fourier-analysis results (fit_fourier_sig intermediates)")

    # -- null-distribution model: one row per distinct M (shared, not
    #    duplicated per group or per unit, and not duplicated across
    #    multiple write_fourier_results calls either) --
    if "null_distribution_models" in module.data_interfaces:
        null_table = module["null_distribution_models"]
        already_have = set(int(v) for v in null_table["Q"][:])
    else:
        null_table = DynamicTable(
            name="null_distribution_models",
            description="Corrected null-distribution PDF/CDF grid, one row per "
                         "distinct Q actually used in this file.",
        )
        null_table.add_column("Q", "off-frequency half-window size")
        null_table.add_column("eps", "get_epsilon(Q) correction factor")
        null_table.add_column("support_r", "PDF/CDF support grid (NFC values)", index=True)
        null_table.add_column("pdf_corrected", "corrected null PDF, same grid as support_r", index=True)
        null_table.add_column("cdf_corrected", "corrected null CDF, same grid as support_r", index=True)
        already_have = set()

    distinct_Ms = sorted({int(entry["M"]) for entry in log_dict.values()})
    R, YY_uncorrected = normalized_Fourier_PDF()
    for M in distinct_Ms:
        if M in already_have:
            continue
        eps = get_epsilon(M)
        PDF = normalized_Fourier_PDF_corrected(R[1:], R[1:], YY_uncorrected[1:], eps)
        CDF = normalized_Fourier_CDF_corrected(PDF, R[1:])
        null_table.add_row(
            Q=int(M), eps=float(eps), support_r=R[1:], pdf_corrected=PDF, cdf_corrected=CDF)
        already_have.add(M)

    # -- one row per (rec, freq, harmonic) group --
    if "fourier_group_results" in module.data_interfaces:
        group_table = module["fourier_group_results"]
    else:
        group_table = DynamicTable(
            name="fourier_group_results",
            description="One row per (rec, freq, harmonic) analysis group — "
                         "shared-across-units Fourier intermediates.",
        )
        group_table.add_column("rec", "recording/trial-block name")
        group_table.add_column("frequency", "stimulus frequency, Hz")
        group_table.add_column("harmonic", "1F or 2F")
        group_table.add_column("Q", "off-frequency half-window size")
        group_table.add_column("T", "analysis duration, s")
        group_table.add_column("C", "number of units in this group")
        group_table.add_column("ff_alt", "off-frequencies analyzed, Hz", index=True)

    # NB: 2F log_dict keys are ("twoF_"+rec, "twoF_"+str(frq)) where `frq` is
    # the *base* (1F) frequency — see fit_fourier_sig, `log_dict[("twoF_" +
    # rec, "twoF_"+str(frq))] = {..., "args": (spks, frq * 2, M_2f)}` — so the
    # base frequency, not frq*2, is what's embedded in the dict key string.
    group_row_by_key = {}  # (rec, base_freq) -> {"1F": (idx, entry), "2F": (idx, entry)}
    for (rec, frq), entry in log_dict.items():
        is_2f = isinstance(rec, str) and rec.startswith("twoF_")
        harmonic = "2F" if is_2f else "1F"
        real_rec = rec[len("twoF_"):] if is_2f else rec
        base_freq = float(str(frq)[len("twoF_"):]) if isinstance(frq, str) and frq.startswith("twoF_") else float(frq)
        analyzed_freq = base_freq * 2 if is_2f else base_freq

        group_table.add_row(
            rec=real_rec, frequency=analyzed_freq, harmonic=harmonic,
            Q=int(entry["M"]), T=float(entry["T"]), C=int(entry["C"]),
            ff_alt=np.asarray(entry["ff_alt"], dtype=float),
        )
        group_idx = len(group_table) - 1
        group_row_by_key.setdefault((real_rec, base_freq), {})[harmonic] = (group_idx, entry)

    # -- one row per unit per group --
    if "per_unit_fourier_results" in module.data_interfaces:
        unit_table = module["per_unit_fourier_results"]
    else:
        unit_table = DynamicTable(
            name="per_unit_fourier_results",
            description="One row per unit per group — today's full_fourier_df "
                         "row plus persisted per-unit Fourier intermediates.",
        )
        unit_table.add_column("group_1f_index", "row index into fourier_group_results, 1F")
        unit_table.add_column("group_2f_index", "row index into fourier_group_results, 2F (-1 if none)")
        unit_table.add_column("unit_id", "Kilosort cluster id (matches Units table id)")
        unit_table.add_column("p_value", "p-value, 1F")
        unit_table.add_column("spk_count", "spike count")
        unit_table.add_column("NFC", "NFC, 1F")
        unit_table.add_column("2f_NFC", "NFC, 2F")
        unit_table.add_column("2f_p_value", "p-value, 2F")
        unit_table.add_column("sens", "detection sensitivity, 1F")
        unit_table.add_column("sens_2f", "detection sensitivity, 2F")
        unit_table.add_column("fou0_real", "on-freq coefficient, real part, 1F")
        unit_table.add_column("fou0_imag", "on-freq coefficient, imag part, 1F")
        unit_table.add_column("fou_alt_real", "off-freq coefficients, real parts, 1F", index=True)
        unit_table.add_column("fou_alt_imag", "off-freq coefficients, imag parts, 1F", index=True)
        unit_table.add_column("sigma", "sqrt(0.5 * mean(|off-freq coeffs|^2)), 1F")

    # positional alignment: within a (rec, freq) group, fit_fourier_sig builds
    # ids/fou0/fou_alt/fourier_df rows all from the same per-unit list, in
    # the same order — so position i in a fourier_df group == position i in
    # log_dict[(rec, freq)]'s arrays. No id-based lookup needed.
    for (rec, freq), gdf in fourier_df.groupby(["rec", "freq"], sort=False):
        key = (rec, float(freq))
        groups = group_row_by_key.get(key, {})
        if "1F" not in groups:
            continue
        idx_1f, entry_1f = groups["1F"]
        idx_2f, _entry_2f = groups.get("2F", (-1, None))

        fou0 = np.asarray(entry_1f["fou0"]).reshape(-1)
        fou_alt = np.asarray(entry_1f["fou_alt"])
        sigma_arr = get_sgm(entry_1f["fou_alt_c"])

        for pos, (_, row) in enumerate(gdf.iterrows()):
            # 2f_NFC/2f_p_value aren't valid Python identifiers, so they can't
            # be passed as literal add_row(2f_NFC=...) keyword args -- build a
            # dict and unpack it instead (DynamicTable.add_row's (*args,
            # **kwargs) signature accepts non-identifier string keys this way
            # even though the literal keyword syntax doesn't parse for them).
            row_kwargs = dict(
                group_1f_index=idx_1f, group_2f_index=idx_2f, unit_id=int(row["id"]),
                p_value=float(row["p_value"]), spk_count=int(row["spk_count"]), NFC=float(row["NFC"]),
                sens=float(row.get("sens", np.nan)), sens_2f=float(row.get("sens_2f", np.nan)),
                fou0_real=float(np.real(fou0[pos])), fou0_imag=float(np.imag(fou0[pos])),
                fou_alt_real=np.real(fou_alt[pos]), fou_alt_imag=np.imag(fou_alt[pos]),
                sigma=float(sigma_arr[pos]),
            )
            row_kwargs["2f_NFC"] = float(row.get("2f_NFC", np.nan))
            row_kwargs["2f_p_value"] = float(row.get("2f_p_value", np.nan))
            unit_table.add_row(**row_kwargs)

    # module.add() on a table that's already a data_interface of this module
    # raises -- only add each table the first time it's created.
    if "null_distribution_models" not in module.data_interfaces:
        module.add(null_table)
    if "fourier_group_results" not in module.data_interfaces:
        module.add(group_table)
    if "per_unit_fourier_results" not in module.data_interfaces:
        module.add(unit_table)
    return module


def read_fourier_results_as_full_fourier_df(nwbfile):
    """Reconstruct a `full_fourier_df`-shaped DataFrame (columns matching
    `fit_fourier_sig()`'s legacy output) from the persisted results tables —
    used by `verify_outputs.py` and `aggregate.py`-equivalents so nothing
    needs to be recomputed to compare against/consume legacy pickles."""
    unit_df = nwbfile.processing["analysis"]["per_unit_fourier_results"].to_dataframe()
    group_df = nwbfile.processing["analysis"]["fourier_group_results"].to_dataframe()

    # Per-row, not a single file-wide scalar: openephys_multistim can apply
    # different Q per stimulus type (mag_Q/visual_Q/WN_Q), and legacy
    # fit_fourier_sig() stamps its own Q argument onto every row of *its*
    # call's result, so the concatenated full_fourier_df genuinely varies by
    # group -- matching that here, not just reading group_df["Q"].iloc[0].
    rec = group_df.loc[unit_df["group_1f_index"], "rec"].values
    freq = group_df.loc[unit_df["group_1f_index"], "frequency"].values
    Q = (group_df.loc[unit_df["group_1f_index"], "Q"].values
         if len(group_df) else np.nan)

    # group_2f_index is -1 for any unit with no paired 2F group (medaka's
    # independent mag/visual groups, or a 1F-only analysis) -- group_df.loc
    # with a -1 label would raise (or silently mis-index), so mask those out
    # rather than indexing group_df with the raw column. This surfaces the
    # 2F bin count that write_fourier_results/write_imaging_fourier_results
    # already persist per-harmonic in fourier_group_results, but which this
    # reader previously dropped on the floor -- callers that need eps for
    # the 2F harmonic's null distribution (e.g. suspect_count_significance)
    # had no way to reconstruct it from all_fourier_df before this.
    group_2f_idx = unit_df["group_2f_index"].values
    Q_2f = np.full(len(unit_df), np.nan)
    has_2f = group_2f_idx >= 0
    if len(group_df) and has_2f.any():
        Q_2f[has_2f] = group_df.loc[group_2f_idx[has_2f], "Q"].values

    result = {
        "id": unit_df["unit_id"].values,
        "p_value": unit_df["p_value"].values,
        "NFC": unit_df["NFC"].values,
        "freq": freq,
        "rec": rec,
        "2f_NFC": unit_df["2f_NFC"].values,
        "2f_p_value": unit_df["2f_p_value"].values,
        "sens": unit_df["sens"].values,
        "sens_2f": unit_df["sens_2f"].values,
        "Q": Q,
        "Q_2f": Q_2f,
    }
    # write_fourier_results (ephys) persists spk_count; write_imaging_fourier_results
    # (engert/medaka) persists n_frames -- this reader is shared across both, so it
    # must emit whichever one this file's per_unit_fourier_results table actually has,
    # rather than forcing both pathways onto one name.
    if "spk_count" in unit_df.columns:
        result["spk_count"] = unit_df["spk_count"].values
    else:
        result["n_frames"] = unit_df["n_frames"].values
    return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# Ophys (engert/medaka): suite2p ROI segmentation + fluorescence traces
# ---------------------------------------------------------------------------
#
# Design mirrors the ephys Units pattern: write ALL suite2p ROIs (not just
# ones passing iscell_threshold/npix_threshold) with `p_iscell`/`npix`
# columns, so that threshold is applied at analysis time (deferred
# filtering) instead of being baked in at processing time -- matching
# CLAUDE.md's documented tutorial workflow of editing iscell_threshold/
# npix_threshold and re-running ONLY the analysis stage, no reprocessing.
# `remove_flatlines` (which depends on the specific frame-slice being
# analyzed) still runs at analysis time for the same reason.

def write_imaging_plane_and_rois(nwbfile, stat, iscell, ops, sampling_rate,
                                  name="PlaneSegmentation", indicator="GCaMP"):
    """Write ALL suite2p ROIs (ImagingPlane + PlaneSegmentation), unfiltered.

    Parameters
    ----------
    stat, iscell : np.ndarray (object / float), suite2p's own `stat.npy`/
        `iscell.npy` arrays, unfiltered (every ROI suite2p found).
    ops : dict, suite2p's `ops.npy` (`.item()`-unwrapped) -- only `Ly`/`Lx`
        (imaging plane pixel dimensions) are used.
    sampling_rate : float, frames/second (1/T, matches fit_Fourier's `T`
        sample-period parameter) -- recorded in scratch like the ephys
        Units table's sampling_rate, so downstream readers don't need `cfg`.

    Columns added: `p_iscell` (suite2p iscell.npy column 1 — the
    classifier's probability, not the binary column 0 the manuscript
    figures use), `npix`, `x`, `y` (`stat["med"]` — suite2p's convention is
    `med = [y, x]`, i.e. row-major; swapped here to the more common x-then-y
    ordering), and a `pixel_mask` (suite2p's `xpix`/`ypix`/`lam`, in NWB's
    `(x, y, weight)` triple convention -- also swapped from suite2p's
    row-major `ypix`/`xpix` storage order).
    """
    device = nwbfile.create_device(
        name="suite2p", description="Suite2p-segmented 2-photon imaging (device details not tracked upstream)")
    optical_channel = OpticalChannel(
        name="OpticalChannel", description=f"{indicator} (emission_lambda not tracked upstream; "
                                            "placeholder value, not a measured wavelength)",
        emission_lambda=509.0)
    imaging_plane = nwbfile.create_imaging_plane(
        name="ImagingPlane",
        optical_channel=optical_channel,
        description="suite2p-segmented imaging plane (all ROIs, unfiltered)",
        device=device,
        excitation_lambda=488.0,  # not tracked upstream; placeholder (common GCaMP excitation)
        indicator=indicator,
        location="unknown",
        imaging_rate=float(sampling_rate),
        grid_spacing=[1.0, 1.0],
        grid_spacing_unit="pixels",
    )

    img_seg = ImageSegmentation()
    ps = img_seg.create_plane_segmentation(
        name=name,
        description="All suite2p ROIs (not pre-filtered by iscell_threshold/"
                     "npix_threshold -- see p_iscell/npix columns; apply "
                     "thresholds at analysis time, matching cfg.good's "
                     "deferred-filtering pattern for ephys Units).",
        imaging_plane=imaging_plane,
    )
    ps.add_column("p_iscell", "suite2p classifier probability (iscell.npy column 1)")
    ps.add_column("npix", "number of pixels in this ROI")
    ps.add_column("x", "ROI centroid x (stat['med'][1])")
    ps.add_column("y", "ROI centroid y (stat['med'][0])")

    for i, s in enumerate(stat):
        pixel_mask = list(zip(
            np.asarray(s["xpix"]).astype(np.uint32).tolist(),
            np.asarray(s["ypix"]).astype(np.uint32).tolist(),
            np.asarray(s["lam"]).astype(np.float32).tolist(),
        ))
        ps.add_roi(
            pixel_mask=pixel_mask,
            p_iscell=float(iscell[i, 1]),
            npix=int(s["npix"]),
            x=int(s["med"][1]),
            y=int(s["med"][0]),
        )

    if "ophys" in nwbfile.processing:
        ophys_module = nwbfile.processing["ophys"]
    else:
        ophys_module = nwbfile.create_processing_module(
            "ophys", "suite2p ROI segmentation + fluorescence traces")
    ophys_module.add(img_seg)

    _set_sampling_rate(nwbfile, sampling_rate)
    _set_imaging_dims(nwbfile, int(ops["Ly"]), int(ops["Lx"]))
    return ps


def _set_imaging_dims(nwbfile, Ly, Lx):
    if "imaging_dims" not in nwbfile.scratch:
        nwbfile.add_scratch(
            np.array([Ly, Lx], dtype=np.int64),
            name="imaging_dims",
            notes="[Ly, Lx] pixel dimensions of the imaging plane (suite2p ops.npy)",
        )


def get_imaging_dims(nwbfile):
    if "imaging_dims" in nwbfile.scratch:
        arr = np.asarray(nwbfile.scratch["imaging_dims"].data)
        return int(arr[0]), int(arr[1])
    return None, None


def write_roi_response_series(nwbfile, F, plane_segmentation, sampling_rate):
    """Write suite2p's `F.npy` (ALL rois, already sliced to this experiment's
    own frame range for engert's shared-session tiffs -- see
    `pipeline/paradigms/engert.py`) as a `RoiResponseSeries` inside a
    `Fluorescence` container (both use fixed names -- `read_roi_data` reads
    them back by those same fixed names).

    `F` : np.ndarray, shape (n_rois, n_frames) -- suite2p's own convention
    (roi-major). NWB's `TimeSeries.data` convention is time-major, so this
    is transposed before writing.

    Stores ALL ROIs regardless of p_iscell/npix -- unlike Units'
    good-only cutover, iscell_threshold/npix_threshold tuning is an
    actively-used workflow (see CLAUDE.md's tutorial), so trimming
    non-passing ROIs here would break "retune the threshold without
    reprocessing." Compression (see GZIP_LEVEL) is applied instead, for a
    smaller but still free and lossless size reduction (~24% measured on
    real data).
    """
    rois_region = plane_segmentation.create_roi_table_region(
        description="all suite2p ROIs (see PlaneSegmentation's p_iscell/npix "
                     "columns for deferred threshold filtering)",
        region=list(range(len(plane_segmentation))),
    )
    roi_series = RoiResponseSeries(
        name="RoiResponseSeries",
        data=H5DataIO(
            data=np.asarray(F, dtype=np.float32).T,  # (n_frames, n_rois)
            compression="gzip", compression_opts=GZIP_LEVEL, chunks=True),
        unit="a.u.",
        rois=rois_region,
        rate=float(sampling_rate),
        starting_time=0.0,
    )
    fluor = Fluorescence(roi_response_series=roi_series, name="Fluorescence")

    ophys_module = nwbfile.processing["ophys"]
    ophys_module.add(fluor)
    return roi_series


def read_roi_data(nwbfile):
    """Reconstruct suite2p-shaped `(F, roi_df)` from the NWB file.

    `F` : np.ndarray, shape (n_rois, n_frames) -- suite2p convention,
    transposed back from NWB's time-major storage.
    `roi_df` : pd.DataFrame, one row per ROI (same order as F's rows),
    columns `p_iscell`, `npix`, `x`, `y`, `pixel_mask` (list of (x, y,
    weight) triples -- lets diagnostics draw the segmentation image without
    a separate stat.npy re-read) -- apply iscell_threshold/npix_threshold at
    the caller (this function returns everything unfiltered, matching how
    the PlaneSegmentation was written).
    """
    ophys_module = nwbfile.processing["ophys"]
    roi_df = ophys_module["ImageSegmentation"]["PlaneSegmentation"].to_dataframe()
    roi_series = ophys_module["Fluorescence"].roi_response_series["RoiResponseSeries"]
    F = np.asarray(roi_series.data).T  # back to (n_rois, n_frames)
    return F, roi_df[["p_iscell", "npix", "x", "y", "pixel_mask"]]


def write_imaging_fourier_results(nwbfile, rec, freq, Q, T_duration, fourier_df_rows,
                                   onfreq_coef, offfreq_coef, freq_win,
                                   onfreq_coef_2f=None, offfreq_coef_2f=None, Q_2f=None):
    """Persist `fit_Fourier()`'s per-cell results into the SAME
    null_distribution_models/fourier_group_results/per_unit_fourier_results
    schema `write_fourier_results` (ephys) uses, so
    `read_fourier_results_as_full_fourier_df` works unchanged for
    engert/medaka too. See this module's docstring for why this is a
    separate writer rather than reusing `write_fourier_results` directly:
    `fit_Fourier()` has no `log_dict` equivalent (no `fou_alt_c`/`get_sgm`
    3-D trial-averaging machinery).

    Mirrors write_fourier_results's own contract: `p_value`/`n_frames`/`NFC`/
    `2f_NFC`/`2f_p_value`/`sens`/`sens_2f`/`id` are trusted AS-IS from
    `fourier_df_rows` (the caller's own already-computed, legacy-pickle-
    matching DataFrame slice for this group) rather than recomputed here --
    recomputing them independently risks a subtly different formula (e.g.
    `fit_Fourier`'s own `compute_NFC` has no epsilon guard against
    `sigma==0`, while the analysis stage's `sens` computation does;
    duplicating that logic here would be a second place for the two to
    drift apart). Only the previously-discarded intermediates (`fou0`,
    `fou_alt`, `sigma`) are computed fresh here, from onfreq_coef/offfreq_coef.

    Parameters
    ----------
    rec, freq, Q, T_duration : this (rec, freq) group's identity/duration (s).
    fourier_df_rows : pd.DataFrame, this group's rows exactly as the caller
        already built them (same row order as onfreq_coef/offfreq_coef),
        columns id/p_value/n_frames/NFC/2f_NFC/2f_p_value/sens/sens_2f --
        NOT the original suite2p ROI index (see module design notes);
        verify_outputs.py parity depends on reusing these columns unchanged.
    onfreq_coef, offfreq_coef : `fit_Fourier`'s `onfreq_coef_l`/`offfreq_coef_l`
        for the base (1F) call, used only to persist fou0/fou_alt/sigma.
    onfreq_coef_2f, offfreq_coef_2f : optional, the SAME for a paired 2F call
        (engert only) -- when given, this call's unit-table rows link both
        1F and 2F groups (matching write_fourier_results's group_1f_index/
        group_2f_index pairing); when omitted, group_2f_index is -1 (no 2F),
        matching medaka's independent-group (no-harmonic-pairing) usage.
    Q_2f : optional, the off-frequency bin count for the paired 2F group
        (only meaningful when onfreq_coef_2f is given). Under the Q_frac
        (bin fraction) policy the fraction is applied independently at each
        analyzed frequency, so the 2F group generally has a DIFFERENT bin
        count than the 1F group's `Q` (M_2f ~= 2*M_1f) -- falls back to `Q`
        only if left None, for defensive backward-compatibility.
    """
    if "analysis" in nwbfile.processing:
        module = nwbfile.processing["analysis"]
    else:
        module = nwbfile.create_processing_module(
            "analysis", "Fourier-analysis results (fit_Fourier intermediates)")

    from magpyneto2.statistics import (
        get_epsilon, normalized_Fourier_PDF,
        normalized_Fourier_PDF_corrected, normalized_Fourier_CDF_corrected,
    )

    if Q_2f is None:
        Q_2f = Q

    if "null_distribution_models" in module.data_interfaces:
        null_table = module["null_distribution_models"]
        already_have = set(int(v) for v in null_table["Q"][:])
    else:
        null_table = DynamicTable(
            name="null_distribution_models",
            description="Corrected null-distribution PDF/CDF grid, one row per "
                         "distinct Q actually used in this file.",
        )
        null_table.add_column("Q", "off-frequency half-window size")
        null_table.add_column("eps", "get_epsilon(Q) correction factor")
        null_table.add_column("support_r", "PDF/CDF support grid (NFC values)", index=True)
        null_table.add_column("pdf_corrected", "corrected null PDF, same grid as support_r", index=True)
        null_table.add_column("cdf_corrected", "corrected null CDF, same grid as support_r", index=True)
        already_have = set()

    distinct_Ms = sorted({int(Q)} | ({int(Q_2f)} if onfreq_coef_2f is not None else set()))
    R, YY_uncorrected = normalized_Fourier_PDF()
    for M in distinct_Ms:
        if M in already_have:
            continue
        eps = get_epsilon(M)
        PDF = normalized_Fourier_PDF_corrected(R[1:], R[1:], YY_uncorrected[1:], eps)
        CDF = normalized_Fourier_CDF_corrected(PDF, R[1:])
        null_table.add_row(
            Q=int(M), eps=float(eps), support_r=R[1:], pdf_corrected=PDF, cdf_corrected=CDF)
        already_have.add(M)

    if "fourier_group_results" in module.data_interfaces:
        group_table = module["fourier_group_results"]
    else:
        group_table = DynamicTable(
            name="fourier_group_results",
            description="One row per (rec, freq, harmonic) analysis group — "
                         "shared-across-units Fourier intermediates.",
        )
        group_table.add_column("rec", "recording/trial-block name")
        group_table.add_column("frequency", "stimulus frequency, Hz")
        group_table.add_column("harmonic", "1F or 2F")
        group_table.add_column("Q", "off-frequency half-window size")
        group_table.add_column("T", "analysis duration, s")
        group_table.add_column("C", "number of units in this group")
        group_table.add_column("ff_alt", "off-frequencies analyzed, Hz", index=True)

    C = len(fourier_df_rows)
    assert C == len(onfreq_coef), (
        f"fourier_df_rows ({C} rows) and onfreq_coef ({len(onfreq_coef)}) must "
        f"describe the same cells in the same order")
    group_table.add_row(
        rec=rec, frequency=float(freq), harmonic="1F", Q=int(Q),
        T=float(T_duration), C=int(C), ff_alt=np.asarray(freq_win, dtype=float))
    idx_1f = len(group_table) - 1

    idx_2f = -1
    if onfreq_coef_2f is not None:
        group_table.add_row(
            rec=rec, frequency=float(freq) * 2, harmonic="2F", Q=int(Q_2f),
            T=float(T_duration), C=int(len(onfreq_coef_2f)), ff_alt=np.asarray(freq_win, dtype=float))
        idx_2f = len(group_table) - 1

    if "per_unit_fourier_results" in module.data_interfaces:
        unit_table = module["per_unit_fourier_results"]
    else:
        unit_table = DynamicTable(
            name="per_unit_fourier_results",
            description="One row per unit per group — today's full_fourier_df "
                         "row plus persisted per-unit Fourier intermediates.",
        )
        unit_table.add_column("group_1f_index", "row index into fourier_group_results, 1F")
        unit_table.add_column("group_2f_index", "row index into fourier_group_results, 2F (-1 if none)")
        unit_table.add_column("unit_id", "sequential id matching legacy full_fourier_df's id column")
        unit_table.add_column("p_value", "p-value, 1F")
        unit_table.add_column("n_frames", "frame count")
        unit_table.add_column("NFC", "NFC, 1F")
        unit_table.add_column("2f_NFC", "NFC, 2F")
        unit_table.add_column("2f_p_value", "p-value, 2F")
        unit_table.add_column("sens", "detection sensitivity, 1F")
        unit_table.add_column("sens_2f", "detection sensitivity, 2F")
        unit_table.add_column("fou0_real", "on-freq coefficient, real part, 1F")
        unit_table.add_column("fou0_imag", "on-freq coefficient, imag part, 1F")
        unit_table.add_column("fou_alt_real", "off-freq coefficients, real parts, 1F", index=True)
        unit_table.add_column("fou_alt_imag", "off-freq coefficients, imag parts, 1F", index=True)
        unit_table.add_column("sigma", "sqrt(0.5 * mean(|off-freq coeffs|^2)), 1F")

    onfreq_coef = np.asarray(onfreq_coef)
    sigma = np.array([
        np.sqrt(0.5 * np.mean(np.abs(np.asarray(offfreq_coef[i])) ** 2))
        for i in range(C)
    ])

    fourier_df_rows = fourier_df_rows.reset_index(drop=True)
    has_2f_cols = "2f_NFC" in fourier_df_rows.columns
    for i in range(C):
        row = fourier_df_rows.iloc[i]
        # 2f_NFC/2f_p_value aren't valid Python identifiers, so they can't be
        # passed as literal add_row(2f_NFC=...) keyword args -- build a dict
        # and unpack it instead (see write_fourier_results for the same
        # pattern/rationale).
        row_kwargs = dict(
            group_1f_index=idx_1f, group_2f_index=idx_2f, unit_id=int(row["id"]),
            p_value=float(row["p_value"]), n_frames=int(row["n_frames"]), NFC=float(row["NFC"]),
            sens=float(row.get("sens", np.nan)), sens_2f=float(row.get("sens_2f", np.nan)),
            fou0_real=float(np.real(onfreq_coef[i])), fou0_imag=float(np.imag(onfreq_coef[i])),
            fou_alt_real=np.real(np.asarray(offfreq_coef[i])),
            fou_alt_imag=np.imag(np.asarray(offfreq_coef[i])),
            sigma=float(sigma[i]),
        )
        row_kwargs["2f_NFC"] = float(row["2f_NFC"]) if has_2f_cols else np.nan
        row_kwargs["2f_p_value"] = float(row["2f_p_value"]) if has_2f_cols else np.nan
        unit_table.add_row(**row_kwargs)

    if "null_distribution_models" not in module.data_interfaces:
        module.add(null_table)
    if "fourier_group_results" not in module.data_interfaces:
        module.add(group_table)
    if "per_unit_fourier_results" not in module.data_interfaces:
        module.add(unit_table)
    return module


# ---------------------------------------------------------------------------
# File-level I/O helpers
# ---------------------------------------------------------------------------

def _compress_units_spike_times(nwbfile):
    """Wrap Units.spike_times' underlying flattened data array in gzip
    compression before writing. `spike_times` is built incrementally across
    many `add_unit()` calls (one per unit, sometimes across multiple
    write_units_and_spikes calls for gutfreund's per-recording sortings),
    so there's no single array to wrap until every unit has been added --
    this must run once, right before the final write, not inside
    write_units_and_spikes itself.

    Empirically measured ~33% size reduction on real data (spike times
    don't have much byte-level redundancy for gzip to exploit, but it's
    free and fully lossless/transparent -- pynwb decompresses
    automatically on read, no changes needed anywhere else).

    Uses the private `_Data__data` attribute because hdmf's VectorData.data
    property has no public setter (confirmed directly: "AttributeError:
    property 'data' of 'VectorData' object has no setter"). This is the
    same workaround hdmf's own internal code uses for exactly this
    situation -- see hdmf/common/table.py's ElementIdentifiers class,
    which does the identical `self._Data__data = ...` reassignment with a
    comment citing the same restriction.
    """
    if nwbfile.units is None:
        return
    vd = nwbfile.units["spike_times"].target
    if isinstance(vd.data, H5DataIO):
        return  # already wrapped (e.g. re-entering on an already-prepped file)
    vd._Data__data = H5DataIO(
        data=vd.data, compression="gzip", compression_opts=GZIP_LEVEL, chunks=True)


def write_nwbfile(nwbfile, nwb_path):
    """Write a freshly-built NWBFile to disk (overwrites any existing file).

    Applies gzip compression to Units.spike_times (the dominant cost in
    every ephys NWB file's size -- see _compress_units_spike_times) right
    before writing. RoiResponseSeries.data is compressed at construction
    time instead (see write_roi_response_series), since it's set once
    rather than built incrementally.
    """
    _compress_units_spike_times(nwbfile)
    with NWBHDF5IO(nwb_path, mode="w") as io:
        io.write(nwbfile)


def rebuild_and_replace_analysis(nwb_path, build_results_fn):
    """Read `nwb_path`, call `build_results_fn(nwbfile)` to add new
    containers (e.g. via `write_fourier_results`), and export the result to
    a fresh copy that atomically replaces the original.

    Fails loudly if the file is missing the processing-stage containers
    this pipeline always writes first — mirroring today's
    `os.path.exists(cfg.nwb_path())` gate in `pipeline/analysis.py`.
    Ephys paradigms write Units/stimulus_epochs; ophys (engert/medaka)
    instead write an "ophys" processing module (PlaneSegmentation +
    RoiResponseSeries) and have neither Units nor stimulus_epochs — accept
    either shape rather than assuming the ephys one.

    Rewrites the WHOLE file (via `NWBHDF5IO.export`) rather than mutating
    the existing file in place (`mode="r+"`, as this used to do). This is
    what lets the analysis stage be re-run standalone as many times as
    needed -- e.g. while retuning Q_frac against the diagnostic PDFs, per
    CLAUDE.md's documented "edit the YAML, re-run analysis only" workflow --
    without reprocessing. Once a DynamicTable's columns are written to disk,
    HDF5 only allows resizing datasets that were created with chunking
    enabled, and the null_distribution_models/fourier_group_results/
    per_unit_fourier_results tables aren't created that way -- a second
    `add_row()` call against an already-persisted table raises `TypeError:
    Only chunked datasets can be resized` (confirmed on real data: this
    fires on literally any second standalone analysis run against an
    already-analyzed file, independent of what changed). Any PRIOR
    "analysis" processing module is dropped in-memory before
    `build_results_fn` runs, so the tables it builds are always freshly
    created (never touching an on-disk dataset that would need resizing),
    and the export writes the whole file out complete rather than
    incrementally.
    """
    if not os.path.exists(nwb_path):
        raise FileNotFoundError(
            f"{nwb_path} does not exist — run the processing stage first.")

    tmp_path = os.path.splitext(nwb_path)[0] + ".tmp.nwb"  # keep the .nwb suffix pynwb expects
    with NWBHDF5IO(nwb_path, mode="r") as io_in:
        nwbfile = io_in.read()
        has_ephys_processing = nwbfile.units is not None and "stimulus_epochs" in nwbfile.intervals
        has_ophys_processing = "ophys" in nwbfile.processing
        if not (has_ephys_processing or has_ophys_processing):
            raise ValueError(
                f"{nwb_path} is missing both Units/stimulus_epochs (ephys) "
                f"and an 'ophys' processing module (imaging) — the "
                f"processing stage for this experiment did not complete "
                f"successfully.")

        if "analysis" in nwbfile.processing:
            del nwbfile.processing["analysis"]
            nwbfile.set_modified()

        build_results_fn(nwbfile)

        try:
            with NWBHDF5IO(tmp_path, mode="w") as io_out:
                io_out.export(src_io=io_in, nwbfile=nwbfile)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    os.replace(tmp_path, nwb_path)


def read_log_dict_equivalent(nwbfile):
    """Reconstruct a `log_dict`-shaped dict (same keys/array shapes
    `fit_fourier_sig()` returns) from the persisted results tables, for
    feeding `plot_analysis_diagnostics` without recomputing anything.

    1F entries get the full shape `fit_fourier_sig()` produces (C/T/ff_alt/M/
    spk_count/fou0/fou_alt/fou_alt_c). 2F entries get only the GROUP-level
    intermediates (C/T/ff_alt/M) -- `per_unit_fourier_results` only
    persists PER-UNIT Fourier coefficients (fou_alt_real/imag, fou0_real/
    imag) for the 1F harmonic (see `write_fourier_results`'s unit_table
    columns), so a reconstructed 2F entry has no `fou_alt`/`fou0`/
    `fou_alt_c` keys. This is enough for a 2F NFC histogram (only needs
    the bin count `M`, via the `"Q"` column -- which always means "bin
    count," not the config fraction, see .claude/plans), but not for a
    per-unit 2F Fourier-coefficient spectrum plot; `plot_analysis_diagnostics`
    skips that panel gracefully when it's missing.
    """
    module = nwbfile.processing["analysis"]
    group_df = module["fourier_group_results"].to_dataframe()
    unit_df = module["per_unit_fourier_results"].to_dataframe()

    log_dict = {}
    for group_idx, group_row in group_df.iterrows():
        is_2f = group_row["harmonic"] == "2F"
        M = int(group_row["Q"])

        if is_2f:
            if len(unit_df[unit_df["group_2f_index"] == group_idx]) == 0:
                continue
            base_freq = float(group_row["frequency"]) / 2
            key = ("twoF_" + group_row["rec"], "twoF_" + str(base_freq))
            log_dict[key] = {
                "C": int(group_row["C"]), "T": float(group_row["T"]),
                "ff_alt": np.asarray(group_row["ff_alt"]), "M": M,
            }
            continue

        rows = unit_df[unit_df["group_1f_index"] == group_idx]
        if len(rows) == 0:
            continue

        spk_count = rows["spk_count"].values.astype(np.int64)
        fou0 = (rows["fou0_real"].values + 1j * rows["fou0_imag"].values).reshape(-1, 1)
        fou_alt_real = np.stack(rows["fou_alt_real"].values)
        fou_alt_imag = np.stack(rows["fou_alt_imag"].values)
        fou_alt = fou_alt_real + 1j * fou_alt_imag
        fou_alt_c = np.dstack((fou_alt_real, fou_alt_imag))

        key = (group_row["rec"], float(group_row["frequency"]))
        log_dict[key] = {
            "C": int(group_row["C"]), "T": float(group_row["T"]), "spk_count": spk_count,
            "ff_alt": np.asarray(group_row["ff_alt"]), "fou0": fou0,
            "fou_alt": fou_alt, "fou_alt_c": fou_alt_c, "M": M,
        }

    return log_dict


def read_nwbfile(nwb_path):
    """Open `nwb_path` read-only and return `(io, nwbfile)`.

    Caller owns the returned `io` and must close it (or use it as a context
    manager) — pynwb keeps the underlying HDF5 file open for lazy dataset
    access, so closing it too early will invalidate any not-yet-materialized
    arrays still referenced from `nwbfile`.
    """
    io = NWBHDF5IO(nwb_path, mode="r")
    return io, io.read()
