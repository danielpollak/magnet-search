"""Fig 4 — modulation strength vs firing rate and q-value responder counts.

Pools a pigeon-HP "pseudopopulation" -- real ("mag" contingency) spike
trains from every experiment YAML with species=="Pigeon" and area=="HP" --
then synthetically modulates the pooled units to show detection thresholds.
Pooling this way (instead of a single recording) keeps Fig4's population
consistent with the "Pigeon HP" population Fig2/Fig3 already report on.

Each pigeon-HP site contains 6-10 separate magnetic recordings, which are
CONCATENATED end-to-end into one non-overlapping timeline per site before a
unit's spikes are pooled (see concat_mag_recs, and EQUAL_WINDOW_S for why
this is not optional). Every unit is then truncated to the first
EQUAL_WINDOW_S (350 s) of its site's timeline (see
truncate_pseudopop_to_window), so the four sites -- whose full timelines run
352.5-784.8 s -- contribute equal-length observations. This applies to the
whole file, not just one panel: the same `spks` list feeds panel A's example
unit, panel B's scatter, compute_sensitivity's top-decile ranking and panels
C/D's responder sweep, so doing either step anywhere but at the loader would
leave those panels describing different populations.
Panel C ports fig4_pilot.py's Storey q-value/FDR responder-count heatmap
onto this file's full pseudopopulation. Panel D repeats that exact
analysis restricted to the top decile (>=90th percentile) by a synthetic-
baseline sensitivity proxy (spk_count/T/2/sigma, computed at this file's
own simulation frequency/Q -- see compute_sensitivity) -- deliberately NOT
the REAL per-unit sens column in data/manuscript/all_fourier_df.parquet
that fig4_pilot.py itself used to select its own (single, subsampled)
population, so this file's population stays fully self-contained from
local NWBs with no aggregate.py/parquet dependency. Panel B (FR vs NFC
scatter) marks those same top-decile-sensitivity units with a black
outline. The old excess-suspects-vs-modulation-amplitude panel (formerly
"B", a single % modulated slice, long before today's B/C/D letters were
reassigned to their current panels) was dropped once the responder-count
heatmap's 2D amplitude x participation sweep made it a strict
generalization -- every "% modulated" line the old panel plotted is one
row of that heatmap's grid.

Panels C/D average N_REPEATS (10) independent random draws of which units
are in the modulated pool at each (participation, amplitude) grid cell,
rather than the single deterministic evenly-rank-spaced pool used
previously -- this washes out a banding/staircase artifact the fixed pool
selection produced. This is affordable despite the added repeats because
compute_responder_df builds a per-unit NFC lookup table ONCE per amplitude
value (a unit's NFC doesn't depend on which other units are also
modulated), so the repeats only add cheap FDR-recombination work, not
more Fourier computation -- see compute_responder_df's docstring for the
full mechanism. compute_responder_df returns every raw per-repeat row
(tagged by a `repeat` column); plot_fig4 does the actual averaging via an
explicit, commented groupby-mean step immediately before each panel's
pivot() call, so the load -> average -> plot flow is visible directly in
the plotting code.

Requires:
  data/{experiment}.nwb for every discovered pigeon-HP experiment
  seaborn

Cached simulation results are saved to data/manuscript/ on first run and
reused on subsequent runs (pass --recompute to force a fresh simulation).
Every sweep (panels B, C, D) is embarrassingly parallel across its grid
cells -- pass --workers N (N>1) to fan them out across a multiprocessing
Pool, same convention as pipeline/processing.py's/analysis.py's --workers.
With N>1, each worker renders its own tqdm progress bar (see
_run_parallel) instead of one pooled counter, so progress is visible
per-worker -- in a terminal or a notebook cell alike.

At the current grid resolution --workers is worth setting for exactly one
step: compute_responder_df's per-amplitude NFC lookup table, which is
~50 min single-process and ~2.5 min at --workers 24. Panels C/D's responder
sweep itself, once the dominant cost, is now seconds -- every cell shares
one pinned null distribution instead of rebuilding it (see
compute_responder_df's `pvalue_upper`) -- so it deliberately stays
in-process below _SWEEP_PARALLEL_MIN_CELLS rather than paying Windows
pool-spawn overhead that exceeds the work itself.

Usage:
    python pipeline/manuscript/fig4.py
    python pipeline/manuscript/fig4.py --workers 8
    python pipeline/manuscript/fig4.py --experiments 20230413_firstsite 20230415
    python pipeline/manuscript/fig4.py --recompute
"""
import argparse
import os
from multiprocessing import Pool, RLock
from pathlib import Path

# Detect if running in Jupyter notebook (must do this before matplotlib.use)
try:
    from IPython import get_ipython
    in_notebook = get_ipython() is not None
except ImportError:
    in_notebook = False

import matplotlib
if not in_notebook:
    matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm.auto as tqdm

from magpyneto2 import statistics
from pipeline import nwb_io
from pipeline.schema import load_experiment

import format_parameters as FP

# Anchored to this file's own location (repo_root/pipeline/manuscript/fig4.py
# -> repo_root), not the process's CWD -- previously these were plain
# CWD-relative strings ("data/manuscript/..."), which silently wrote (and
# once got committed) a stray duplicate cache under
# pipeline/manuscript/data/manuscript/ when this script was run with CWD set
# to its own directory (e.g. from a Jupyter notebook there) instead of the
# repo root the docstring's own usage examples assume.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EXPERIMENTS_DIR = _REPO_ROOT / "experiments"

# Every pooled unit is observed over the SAME window: the first
# EQUAL_WINDOW_S seconds of its experiment's own CONCATENATED timeline (see
# concat_mag_recs and truncate_pseudopop_to_window).
#
# Why a concatenated timeline at all: each pigeon-HP site contributes 6-10
# separate magnetic recordings, and nwb_io.build_modulation_frame hands back
# RECORDING-LOCAL spike times (`all_sts - beginning_time`, see the
# local_offset comment there) because that is what fit_fourier_sig wants --
# it groups by ("rec", "freq"), so a group is always one rec and local times
# are exactly right. Pooling a unit's recs, as this file does, is the one
# thing that convention cannot survive: every rec's magnetic epoch starts
# ~30-60 s into that rec, so a naive groupby("id") OVERLAYS all 6-10 recs on
# one ~30-90 s axis instead of laying them end to end. That inflated every
# firing rate by however many recs the unit appeared in (6-13x, varying by
# site) and left fourier_analysis's T at ~60 s when 350-785 s had really been
# observed. concat_mag_recs shifts each rec onto a single non-overlapping
# timeline instead.
#
# 350 s is the largest round window every site can supply -- the four sites'
# concatenated durations are 544.8, 465.8, 784.8 and 352.5 s. Because a unit
# is recorded for its site's whole timeline whether or not it fires in every
# rec, nothing needs dropping for spanning too little: the old
# first-to-last-spike span test (and the "duration banding" it was written to
# fix, which was itself an artifact of the overlay) is gone, and only a
# min-spike floor remains.
EQUAL_WINDOW_S = 350.0
MIN_SPIKES = 6  # same floor load_unit_spks_for_experiment applies at load time
# Two tag components beyond the window length, because two changes have each
# altered what a given nominal window MEANS:
#   "concat"  -- every cache predating it was computed on the OVERLAID
#                timeline, where the same nominal window was a completely
#                different observation of every unit.
#   "allspk"  -- min_spikes=0 on the build_modulation_frame call (see
#                load_unit_spks_for_experiment): restores the ~4% of
#                in-window spikes the per-rec 50-spike threshold discarded,
#                and admits units that cleared it in no single rec, so the
#                pooled population itself is larger.
# Neither an old "..._60s.pkl" nor an old "..._concat350s.pkl" may be
# silently reloaded as if it matched the current definition.
_WINDOW_TAG = f"concat{EQUAL_WINDOW_S:g}s_allspk"

# _pigeon_hp_pseudopop suffix: deliberately distinct from the old
# single-recording cache filenames (modulation_strength_vs_excess_count.pkl /
# modulation_strength_vs_FR.pkl) so a stale single-session cache can never be
# silently reused now that the population is pooled across experiments. The
# old files, if present, are simply orphaned -- harmless, untracked. (The
# excess-count cache itself -- CI_DF_CACHE -- was dropped along with the
# panel it fed; qvalue_responder_df_pigeon_hp_pseudopop.pkl, if a stale copy
# lingers, is unaffected.) FR_DF_CACHE additionally carries a "_v3" suffix:
# compute_fr_df's FR values have changed meaning twice now -- v2 normalized
# by each unit's own experiment's mag-trial window T; v3 bounded T by each
# unit's own first-to-last-spike span, a strictly tighter per-unit bound;
# v3_{_WINDOW_TAG} (current) divides instead by the one common observation
# window every unit now shares, see unit_firing_rates -- so any older
# _pigeon_hp_pseudopop.pkl / "_v2".pkl / unstamped "_v3".pkl left on disk is
# stale and must not be silently reloaded. Every cache path below
# is additionally stamped with _WINDOW_TAG, so changing EQUAL_WINDOW_S (which
# changes both which units are pooled and how much of each one is kept, hence
# every NFC in every panel) can never silently reload a pickle computed for a
# different window.
FR_DF_CACHE   = _REPO_ROOT / "data" / "manuscript" / f"modulation_strength_vs_FR_pigeon_hp_pseudopop_v3_{_WINDOW_TAG}.pkl"
# "_lineargrid" suffix: AMPLITUDES_RESP/PARTICIPATION_RESP went through a
# log-spaced detour (several different ranges) before settling back on a
# plain linear arange -- any older qvalue_responder_df*.pkl (any "_log*"
# suffix) holds responder counts indexed by one of those OLD grids and must
# not be silently reloaded as if it matched the current one.
# "_halfpart_10rep" suffix (added on top of "_lineargrid"): marks two
# coupled changes made together -- (a) PARTICIPATION_RESP's resolution was
# halved (100 -> 50 points, see its definition below) and (b) each grid
# cell now holds N_REPEATS randomly-pooled repeat rows (a new `repeat`
# column) instead of a single deterministic rank-spaced pool -- see
# compute_responder_df. An old "..._lineargrid.pkl" (one pool per cell, no
# `repeat` column, 100-point participation axis) must never be silently
# reloaded as if it already had repeats or the new grid: either difference
# would otherwise corrupt plot_fig4's pivot()/groupby() silently rather
# than raising a loud error.
RESP_DF_CACHE = _REPO_ROOT / "data" / "manuscript" / f"qvalue_responder_df_pigeon_hp_pseudopop_lineargrid_halfpart_10rep_{_WINDOW_TAG}.pkl"
RESP_DF_TOP_DECILE_CACHE = _REPO_ROOT / "data" / "manuscript" / f"qvalue_responder_df_pigeon_hp_pseudopop_top_decile_lineargrid_halfpart_10rep_{_WINDOW_TAG}.pkl"

SENSITIVITY_PERCENTILE = 90.0  # panel C / panel D outline: top decile by compute_sensitivity

FREQ        = 5

# Panel A palette. Everything in this panel is achromatic on purpose: Fig4 is
# a SIMULATION, and the manuscript reserves blue/orange (FP.COLOR_MAG /
# FP.COLOR_VIS) for real magnetic vs. visual stimulation -- using either here
# would imply a stimulus contingency these synthetic spike trains don't have.
# Within the panel, darkness encodes what's primary: the on-frequency stem
# (the quantity the panel is about) is black, and the off-frequency |c_n|
# cloud and the raster ticks are mid-grey. The PSTH curve is black too --
# it plays no part in the Fourier/NFC computation, but it's the only thing
# in the raster that shows the modulation as a shape rather than as tick
# density, so it reads as a summary of the grey ticks rather than as
# something subordinate to them.
SPECTRUM_DOT_COLOR = "0.45"
SIGMA_COLOR        = "0.6"
RASTER_COLOR       = "0.45"
PSTH_COLOR         = "black"

# PSTH smoothing: finer bins than the helper's 36 default, with a
# proportionally wider Gaussian (sigma 6/72 bins = 30 deg of phase, vs the
# default's 20 deg), so the curve reads as a smooth modulation envelope
# rather than a 36-segment polyline tracing bin-to-bin Poisson noise.
PSTH_N_BINS      = 72
PSTH_SMOOTH_BINS = 6.0

# Padding on panel A's rasters, as a fraction of the phase span / cycle
# count. Without it the axes limits snap exactly to the data, so spikes at
# phase 0/2*pi and on the first/last cycle are drawn half-under the spines,
# and the PSTH curve's troughs sit right on the bottom spine.
# Panel A's rasters keep only the endpoint/midpoint phase ticks rather than
# statistics._PHASE_TICKS' full five. At FIGSIZE_FIG4's width each raster is
# under an inch wide, and five LaTeX-rendered labels ("0", "$\pi/2$", ...)
# run into each other. Overridden here rather than in the shared helper
# because fig1's rasters are wide enough to keep all five.
RASTER_PHASE_TICKS      = [0, np.pi, 2 * np.pi]
RASTER_PHASE_TICKLABELS = ["0", r"$\pi$", r"$2\pi$"]

RASTER_PAD_X = 0.03
RASTER_PAD_Y = 0.03
PSTH_PAD_Y   = 0.06

# Panel B mod-condition hues. Dark2 rather than seaborn's Set1 default, whose
# first two entries are a red and a blue close enough to the manuscript's own
# stimulus colors to be misread as them.
PALETTE_FIG4B = "Dark2"

# Panel B (ported from fig4_pilot.py) sweep grids -- matches Markus Meister's
# MM_Analysis_3.ipynb notebook's own grids. Named "_RESP" (for the
# responder-count sweep computed below), not "_B" -- the panel letter this
# maps to is a display choice made once, in plot_fig4, not baked into
# identifiers here.
#
# Linear grid, 0.0 through 0.9 (np.arange's stop is exclusive, so this
# does NOT include 1.0) -- reverted back to this after a log-spaced detour
# (logspace(-2,0), then logspace(-1.2,0)) meant to reveal more of the
# detectability contour's shape near small amplitude/participation values;
# on balance this plain linear grid read better. Unlike the previous
# linear grid this file had (AMPLITUDES_RESP starting at 0.1, with
# amplitude=0.0 handled as a separate baseline case -- see
# compute_responder_df), AMPLITUDES_RESP now includes 0.0 directly, so
# compute_responder_df's own amplitude=0.0 baseline-row shortcut skips
# building a redundant (and pivot()-breaking, since it'd duplicate the
# (participation, amplitude) index) task for amplitude==0.0 explicitly.
#
# PARTICIPATION_RESP's resolution was later halved (100 -> 50 points) to
# make room for N_REPEATS-fold repetition (see below) at a comparable
# overall grid cell count: 50 x 10 x 99 = 49,500 cells vs. the old
# 100 x 99 = 9,900 -- a 5x increase, not 10x, since the y-axis got coarser
# at the same time repeats were added. This is affordable because
# compute_responder_df no longer recomputes a Fourier transform per grid
# cell (see its docstring): the actually expensive step now runs only
# ~len(AMPLITUDES_RESP) times total (an NFC lookup table, built once),
# regardless of participation resolution or repeat count.
#
# Each grid cell's modulated pool used to be a single DETERMINISTIC
# rank-spaced subset (evenly spaced by baseline sensitivity rank) -- fully
# reproducible, but this produced a visible banding/staircase artifact in
# panels C/D from pool_size = int(participation * n) truncation interacting
# with that fixed selection. Pool membership is now drawn RANDOMLY per
# repeat (see compute_responder_df), and N_REPEATS independent draws are
# averaged (see plot_fig4's explicit groupby-mean step) specifically to
# wash out that artifact.
AMPLITUDES_RESP    = np.arange(0, 1, 0.01)
PARTICIPATION_RESP = np.arange(0, 1, 0.02)
QVALUE_FDR         = 0.05
# Responder-sweep cell count above which compute_responder_df bothers with a
# worker pool -- at ~0.1 ms/cell this is roughly a minute of serial work,
# comfortably more than pool startup costs. Well above the current grid's
# 49,500 cells; it only trips if PARTICIPATION_RESP/AMPLITUDES_RESP/N_REPEATS
# are refined by an order of magnitude.
_SWEEP_PARALLEL_MIN_CELLS = 500_000
N_REPEATS          = 10  # independent random-pool draws averaged per (participation, amplitude) cell


def discover_pigeon_hp_experiments(experiments_dir: Path):
    """Auto-discover every experiment YAML with species=="Pigeon" and
    area=="HP" -- the same predicate fig3.py uses to build its "Pigeon HP"
    panels off all_fourier_df.parquet -- so Fig4's pseudopopulation tracks
    that set automatically instead of a hardcoded experiment list.
    """
    names = []
    for yml_path in sorted(experiments_dir.glob("*.yml")):
        cfg = load_experiment(yml_path)
        if cfg.species == "Pigeon" and cfg.area == "HP":
            names.append(cfg.name)
    if not names:
        raise ValueError(f"No species=='Pigeon' & area=='HP' experiment YAMLs found in {experiments_dir}")
    return names


def concat_mag_recs(nwbfile, mag_substr):
    """Lay one experiment's magnetic recordings end-to-end onto a single
    non-overlapping timeline.

    Returns (rec_map, total_duration) where rec_map maps a rec name to
    (local_start, duration, offset): subtract `local_start` from that rec's
    recording-local spike times to re-zero them on its own stimulus epoch,
    then add `offset` to place it after every preceding rec. See
    EQUAL_WINDOW_S for why this is necessary at all.

    Both quantities come from the NWB `stimulus_epochs` table rather than
    from the spikes:

      duration    = stop_time - start_time. This is the same T
                    fit_fourier_sig derives for the rec from the period
                    column ((1/freq) * max(period)) -- checked on all 32
                    pigeon-HP mag recs, where the two agree to <= 0.0036 s
                    -- so the concatenated timeline is tiled by exactly the
                    windows the real per-rec analysis uses. Taking it from
                    the epoch table instead of the periods keeps it
                    independent of which units happened to fire.
      local_start = start_time - local_offset_seconds, i.e. the epoch's
                    start expressed in the same recording-local frame as
                    `spk` (write_epochs_table stores start/stop in the
                    aggregated whole-catalog domain, and
                    build_modulation_frame subtracts local_offset_seconds
                    from the spike times only).

    Recs are ordered by `start_time`, i.e. the order they sit in the
    concatenated catalog Kilosort actually sorted. That is deliberately NOT
    the same as sorting by the wall-clock timestamp in the rec NAME --
    20230413_firstsite's catalog runs 15-08, then 15-29 down to 15-11 -- but
    correctness doesn't depend on the choice (any consistent order yields a
    valid non-overlapping timeline, and warp_mod imposes its synthetic
    modulation coherently on whatever timeline it is given), and catalog
    order needs no filename parsing.
    """
    epochs = nwbfile.intervals["stimulus_epochs"].to_dataframe()
    epochs = epochs.loc[[mag_substr in rec for rec in epochs["rec"]]]
    if epochs.empty:
        raise ValueError(
            f"No stimulus_epochs rows matching mag_rec_substring={mag_substr!r}"
        )
    # One epoch per mag rec holds for every pigeon-HP experiment. Guarded
    # rather than assumed: with two windows for one rec, a single
    # (local_start, offset) pair per rec would silently misplace one of them.
    multi = epochs["rec"].value_counts()
    if (multi > 1).any():
        raise ValueError(
            f"Expected one stimulus epoch per mag rec, got "
            f"{multi[multi > 1].to_dict()} -- concat_mag_recs needs extending "
            f"to place each window separately."
        )

    epochs = epochs.sort_values("start_time")
    durations = (epochs["stop_time"] - epochs["start_time"]).values
    local_starts = (epochs["start_time"] - epochs["local_offset_seconds"]).values
    # Exclusive cumulative sum: rec 0 lands at 0, rec i after every rec
    # before it. NOT i * duration[i-1] -- these recs run 47.8-139.0 s within
    # a single site, so a uniform stride would both overlap and leave gaps.
    offsets = np.concatenate([[0.0], np.cumsum(durations)[:-1]])
    rec_map = {rec: (float(ls), float(d), float(o))
               for rec, ls, d, o in zip(epochs["rec"], local_starts, durations, offsets)}
    return rec_map, float(durations.sum())


def load_unit_spks_for_experiment(data_dir: str, experiment: str):
    """Load one pigeon-HP experiment's "mag"-contingency spike trains, one
    spike array per real unit, with that experiment's magnetic recordings
    CONCATENATED end-to-end onto a single non-overlapping timeline (see
    concat_mag_recs).

    Returns (unit_spks, Q_frac, total_duration) where unit_spks maps
    f"{experiment}:{cluster_id}" -> spike-time array on that timeline. The
    namespaced key only exists to keep this experiment's units distinct when
    pooled with other experiments' units downstream -- cluster_id alone is
    not globally unique (see fig2.py/fig3.py's species/date/id dedup key).

    Pooling a unit's recs is meaningful because they share a single Kilosort
    sort. It is NOT what fit_fourier_sig does, though -- that groups by
    ("rec", "freq") and never mixes recs, which is why the concatenation
    above has to be done explicitly here rather than inherited from the real
    analysis path.

    `min_spikes=0` on the build_modulation_frame call, against its default of
    50: that default is a faithful port of legacy process_raw_data_NPIX's
    `if len(st) < 50: continue`, applied PER (unit, rec). It is the right
    convention for the real per-rec analysis -- a 50-spike Fourier estimate
    from one rec is not worth having -- but it is wrong for a pooled,
    concatenated timeline, where a unit's spikes from every rec are counted
    together against one duration. Under the default, a unit is simply absent
    from any rec it fired under 50 times in, so those real spikes vanish
    while the rec's seconds still count toward its firing rate. Measured on
    the four pigeon-HP sites: 2612 of 6200 (unit, rec) pairs were dropped in
    20230414_firstsite alone (every one holding 0-49 real spikes, verified
    against Units.spike_times, max 49 -- versus a minimum of 50 in every
    retained pair), costing 109,563 of 2,891,431 in-window spikes (3.8%)
    across all sites, with 1020 of 1555 units losing some. The bias is
    strongly rate-dependent, so it lands hardest exactly where it matters
    most for a detectability simulation: units firing under 0.5 Hz kept a
    median of just 48% of their spikes.

    Note that this admits units the default would have excluded from every
    rec (fewer than 50 spikes in each), so the pooled population is somewhat
    larger than the real analysis's. That is the intended trade: this file
    simulates detectability as a function of firing rate, so it must see each
    unit's actual rate rather than a per-rec-thresholded view of it.
    """
    # Reads the pipeline's own {experiment}.nwb (Phase 7 cutover -- no
    # paradigm writes the legacy {experiment}_processing.pickle anymore),
    # same reproducible-from-NWB principle applied to fig1.py.
    cfg = load_experiment(_EXPERIMENTS_DIR / f"{experiment}.yml")
    nwb_path = Path(data_dir) / f"{experiment}.nwb"
    io_r, nwbfile = nwb_io.read_nwbfile(str(nwb_path))
    # min_spikes=0, NOT the default 50 -- see this function's docstring. For
    # phase_method="crossings" the filter is `len(st) < min_spikes`, so 0
    # disables it outright (a unit with no spikes in a window contributes an
    # empty frame, which concats away harmlessly).
    modulation_df = nwb_io.build_modulation_frame(nwbfile, good_only=cfg.good,
                                                  min_spikes=0)
    # Read from the same open file -- the epoch table this uses is the one
    # whose local_offset_seconds build_modulation_frame just applied.
    mag_substr = cfg.analysis.mag_rec_substring
    rec_map, total_T = concat_mag_recs(nwbfile, mag_substr)
    io_r.close()

    # Same plain substring test analysis_stages/multistim.py uses to build
    # its mag_mask -- "mag" contingency is the real magnetic-trial recs,
    # excluding auxiliary_stimuli (visual/WN/oddball) recs, which are always
    # "positive control".
    mag_df = modulation_df.loc[[mag_substr in rec for rec in modulation_df.rec]]
    if mag_df.empty:
        available = modulation_df.rec.unique().tolist()
        raise ValueError(
            f"No recs matching mag_rec_substring={mag_substr!r} in "
            f"{nwb_path.name}. Available recs: {available}"
        )
    missing = set(mag_df.rec.unique()) - set(rec_map)
    if missing:
        raise ValueError(
            f"{nwb_path.name}: recs present in modulation_df with no stimulus "
            f"epoch to place them on the concatenated timeline: {sorted(missing)}"
        )

    unit_spks = {}
    for uid, g in mag_df.groupby("id"):
        parts = []
        for rec, g_rec in g.groupby("rec"):
            local_start, _dur, offset = rec_map[rec]
            parts.append(np.asarray(g_rec.spk.values, dtype=float) - local_start + offset)
        # Sorted because the pooled frame's row order is dict/table iteration
        # order across recs, not time -- fit_fourier_sig sorts its own
        # per-unit arrays for the same reason (np.sort(id_subdf.spk.values)).
        spkt = np.sort(np.concatenate(parts))
        if len(spkt) > 5:
            unit_spks[f"{experiment}:{uid}"] = spkt

    # Cheap check that the tiling really is non-overlapping: every spike must
    # land inside [0, total_T). A violation means a spike outside its own
    # epoch window, which is exactly what the offsets assume away.
    for key, spkt in unit_spks.items():
        if spkt[0] < -1e-6 or spkt[-1] >= total_T + 1e-6:
            raise ValueError(
                f"{key}: concatenated spike times span [{spkt[0]:.3f}, "
                f"{spkt[-1]:.3f}] s, outside the [0, {total_T:.3f}) s timeline "
                f"-- a spike falls outside its own rec's stimulus epoch."
            )

    print(f"  {experiment}: {len(unit_spks)} mag units with >5 spikes, "
          f"{len(rec_map)} mag recs concatenated -> {total_T:.1f} s timeline")

    Q_frac = cfg.analysis.mag_Q_frac if cfg.analysis.mag_Q_frac > 0 else cfg.analysis.Q_frac
    if Q_frac <= 0:
        raise ValueError(
            f"{experiment}.yml has no analysis.mag_Q_frac/Q_frac set -- "
            f"Fig4's simulation derives its off-frequency bin count from "
            f"this experiment's own Q_frac, same as the real per-experiment "
            f"analysis."
        )
    return unit_spks, Q_frac, total_T


def truncate_pseudopop_to_window(unit_spks, window_s=EQUAL_WINDOW_S,
                                 min_spikes=MIN_SPIKES):
    """Put every unit on a common observation window: keep only the first
    `window_s` seconds of its experiment's concatenated timeline. Returns a
    new dict; the input is not modified.

    Anchored at t=0 of that timeline (the first mag rec's epoch start), NOT
    at each unit's own first spike. On a real timeline the electrode was
    recording for the whole window regardless of when a given unit happened
    to fire, so t=0 is the honest observation start and every kept unit
    shares exactly `window_s` seconds of it -- which is also what makes
    fourier_analysis's single population-wide T correct for all of them.

    This replaces the old equalize_unit_windows, which measured each unit's
    first-to-last-spike SPAN and dropped anything under the window. That test
    only made sense on the overlaid timeline, where there was no real
    duration to appeal to; it dropped 249 of 1555 units purely for firing
    sparsely near the edges of a ~60 s overlay (see the fig4B_fano_by_rec
    diagnostic). Nothing is dropped for duration now -- only the same
    min_spikes floor load_unit_spks_for_experiment applies at load time,
    re-applied because truncation can push a sparse unit below it.
    """
    kept, sparse = {}, 0
    for key, spkt in unit_spks.items():
        trunc = spkt[spkt < window_s]
        if len(trunc) < min_spikes:
            sparse += 1
            continue
        kept[key] = trunc
    print(f"  truncated to a common {window_s:g} s window: {len(kept)} units kept, "
          f"{sparse} dropped for having <{min_spikes} spikes left after truncation")
    if not kept:
        raise ValueError(
            f"No units survive the {window_s:g} s common window. "
            f"Lower EQUAL_WINDOW_S."
        )
    return kept


def load_pseudopopulation_spks(data_dir: str, experiments):
    """Pool load_unit_spks_for_experiment() across every given experiment
    into one pigeon-HP pseudopopulation, put every unit on the same
    EQUAL_WINDOW_S observation window (see truncate_pseudopop_to_window),
    then sort by spike count ascending (plot_fig4 picks its example unit via
    spks[len(spks) // 2], i.e. the median-by-spike-count unit -- same
    convention the old single-recording load_spks used).

    Each experiment's units arrive on that experiment's OWN concatenated
    timeline, all starting at t=0, so truncating the merged dict is
    equivalent to truncating each experiment separately -- there is no
    cross-experiment timeline and none is implied.
    """
    all_units = {}
    q_fracs = {}
    total_Ts = {}
    for experiment in experiments:
        unit_spks, Q_frac, total_T = load_unit_spks_for_experiment(data_dir, experiment)
        all_units.update(unit_spks)
        q_fracs[experiment] = Q_frac
        total_Ts[experiment] = total_T

    # Checked before truncating rather than after: an experiment whose whole
    # concatenated timeline is shorter than the window would silently
    # contribute units observed for less than every other experiment's,
    # which is the exact defect the common window exists to prevent.
    too_short = {e: T for e, T in total_Ts.items() if T < EQUAL_WINDOW_S}
    if too_short:
        raise ValueError(
            f"EQUAL_WINDOW_S={EQUAL_WINDOW_S:g} s exceeds the total concatenated "
            f"mag duration of {too_short} -- lower it to at most "
            f"{min(total_Ts.values()):.1f} s (the shortest pooled experiment)."
        )
    all_units = truncate_pseudopop_to_window(all_units)

    unique_q_fracs = set(q_fracs.values())
    if len(unique_q_fracs) > 1:
        raise ValueError(
            f"mag_Q_frac/Q_frac differs across pooled experiments -- "
            f"pooling only has a well-defined meaning when every pooled "
            f"experiment shares one Q_frac: {q_fracs}"
        )
    Q_frac = unique_q_fracs.pop()

    keys_sorted = sorted(all_units, key=lambda k: len(all_units[k]))
    spks = [all_units[k] for k in keys_sorted]
    print(f"  {len(spks)} pigeon-HP pseudopopulation units pooled from "
          f"{len(experiments)} experiments "
          f"(concatenated timelines: "
          f"{', '.join(f'{e}={T:.0f}s' for e, T in total_Ts.items())})")
    return spks, Q_frac


def fourier_Q_from_frac(spks, freq, Q_frac, context=""):
    """Convert `Q_frac` (fraction of `freq`) into the raw off-frequency bin
    count `fourier_analysis` takes as `Q` -- same `bins_for_fraction` formula
    `fit_fourier_sig` uses for a real experiment's analysis, so Fig4's
    synthetic-modulation simulation is expressed in the same Q_frac units as
    every other figure instead of a literal bin count. `T` is computed the
    same way `fourier_analysis` computes it internally when `T` isn't passed
    explicitly (`ceil(latesttime - earliesttime)`), so the resulting M
    matches what `fourier_analysis` would derive on its own at this Q.
    """
    T = np.ceil(statistics.latesttime(spks) - statistics.earliesttime(spks))
    resolution = 1 / T
    return statistics.bins_for_fraction(freq, Q_frac, resolution, context=context)


# ---------------------------------------------------------------------------
# Parallel sweep machinery.
#
# Every grid cell below (one A for panel B, one (participation, amplitude)
# pair for panels C/D) is an independent call to statistics.fourier_analysis
# over the SAME pooled `spks` -- embarrassingly parallel. _init_worker
# stashes the shared, possibly-large `spks` list once per worker process
# (via Pool's initializer) instead of re-pickling it into every task, and
# the worker functions below read it back out of that per-process global.
# With --workers 1 (the default) _run_parallel skips Pool entirely and calls
# _init_worker/the worker function directly in this process, so the single-
# and multi-process code paths share one implementation of each sweep's
# actual math.
# ---------------------------------------------------------------------------
_worker_state = {}


def _init_worker(spks, freq, Q, extra=None):
    # Cleared, not merged into: with --workers 1 every sweep re-inits this
    # same module-level dict in this same process, so leftover keys from a
    # previous sweep's `extra` would otherwise still be visible to the next
    # one (which reads them by name and can't tell they're stale).
    _worker_state.clear()
    _worker_state["spks"] = spks
    _worker_state["freq"] = freq
    _worker_state["Q"] = Q
    _worker_state["eps"] = statistics.get_epsilon(Q)
    # Optional extra per-sweep payload (e.g. compute_responder_df's NFC
    # lookup table) that doesn't fit the spks/freq/Q shape every sweep
    # shares -- merged in verbatim so worker functions can pull additional
    # keys out of _worker_state without changing this function's signature
    # per sweep. None (the default) leaves _worker_state exactly as before,
    # so sweeps that don't need it (e.g. compute_fr_df's _fr_cell) are
    # unaffected.
    if extra:
        _worker_state.update(extra)
    # If this sweep pins a shared null-distribution support (see
    # compute_responder_df's `pvalue_upper`), build it now rather than
    # letting the first task pay for it: statistics.corrected_null_grid is
    # memoized per process, so this is one ~1s convolution per worker at
    # pool startup instead of a stall part-way into that worker's own
    # progress bar, which would otherwise make its first-iteration rate
    # look far worse than the sweep's real per-cell cost.
    if _worker_state.get("pvalue_upper") is not None:
        statistics.corrected_null_grid(_worker_state["eps"],
                                       _worker_state["pvalue_upper"])


def _init_worker_bars(spks, freq, Q, worker_fn, lock, extra=None):
    """Same as _init_worker, plus stashing worker_fn itself (so _run_chunk
    below can look it up per-process) and re-installing the shared tqdm
    lock -- tqdm's own recipe for rendering multiple bars from separate
    processes without their line-redraws stomping on each other.
    """
    _init_worker(spks, freq, Q, extra=extra)
    _worker_state["worker_fn"] = worker_fn
    tqdm.tqdm.set_lock(lock)


def _run_chunk(indexed_chunk):
    idx, chunk = indexed_chunk
    worker_fn = _worker_state["worker_fn"]
    bar = tqdm.tqdm(chunk, desc=f"worker {idx}", position=idx, leave=True)
    return idx, [worker_fn(t) for t in bar]


def _run_parallel(tasks, worker_fn, spks, freq, Q, workers, desc="", extra=None):
    """With workers<=1, runs every task in-process behind a single tqdm bar.
    With workers>1, splits tasks round-robin into `workers` chunks -- one
    per pool worker -- and has each worker render its OWN tqdm bar (stacked
    via `position=`) tracking just its own chunk, instead of one bar
    counting completions pooled across all of them. This makes load
    imbalance visible (useful once a grid this large, e.g. a 10x-finer
    AMPLITUDES_RESP/PARTICIPATION_RESP, takes long enough to want a progress
    read per-worker rather than an aggregate one). tqdm.auto resolves to
    plain-text bars inside each worker process automatically -- a spawned
    worker has no IPython kernel of its own even when the parent process is
    a notebook -- so this renders correctly both from a terminal and from a
    notebook cell; see tqdm's own parallel-bars recipe (tqdm/examples in the
    tqdm repo) for the set_lock/get_lock pattern this mirrors.

    `extra`, if given, is an additional dict merged into each worker
    process's _worker_state (see _init_worker) -- e.g. compute_responder_df
    passes its precomputed NFC lookup table this way instead of via the
    task tuples themselves, since it's shared read-only state common to
    every task rather than something that varies per task.
    """
    if workers <= 1:
        _init_worker(spks, freq, Q, extra=extra)
        return [worker_fn(t) for t in tqdm.tqdm(tasks, desc=desc)]
    workers = min(workers, len(tasks)) or 1
    chunks = [tasks[i::workers] for i in range(workers)]
    lock = RLock()
    with Pool(workers, initializer=_init_worker_bars,
              initargs=(spks, freq, Q, worker_fn, lock, extra)) as pool:
        chunk_results = pool.map(_run_chunk, list(enumerate(chunks)))
    chunk_results.sort(key=lambda ic: ic[0])
    return [r for _, chunk in chunk_results for r in chunk]


def _nfc_table_cell(task):
    """One amplitude's row of compute_responder_df's NFC lookup table:
    warp every unit at this amplitude, then one batched fourier_analysis
    over the whole population.

    `T` is read from _worker_state rather than recomputed, because it must
    be the value pinned by the caller's first (unmodulated) pass -- see
    compute_responder_df's docstring for why the table can't tolerate a
    per-call T.
    """
    (amplitude,) = task
    spks = _worker_state["spks"]
    freq = _worker_state["freq"]
    Q = _worker_state["Q"]
    T = _worker_state["T"]
    modulated = [statistics.warp_mod(spkt, amplitude, 1 / freq, 0) for spkt in spks]
    (_, _, _, _, _, _, _, _, _, NFC) = statistics.fourier_analysis(modulated, freq, Q=Q, T=T)
    return amplitude, NFC


def _fr_cell(task):
    (A,) = task
    spks = _worker_state["spks"]
    freq = _worker_state["freq"]
    Q = _worker_state["Q"]
    modulated = [statistics.modulate(spkt, freq, A) for spkt in spks]
    (C, _, spk_count, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, NFCs) = \
        statistics.fourier_analysis(modulated, freq, Q=Q)
    return A, NFCs


def unit_firing_rates(spks, window_s=EQUAL_WINDOW_S):
    """Each unit's spike count over the common observation window every
    pooled unit shares (see truncate_pseudopop_to_window).

    `window_s` is a real duration now. Before the recs were concatenated
    (see EQUAL_WINDOW_S) this same division was badly wrong: a unit's pooled
    array held all 6-10 of its recs OVERLAID on one ~60 s axis, so dividing
    by 60 s counted every rec's spikes against a single rec's worth of time
    and inflated every rate by roughly the number of recs the unit appeared
    in -- 6-13x, and by a different factor per site, so not even a constant
    that could be divided out afterwards. One unit came out at 556 Hz.

    Dividing by the window rather than by each unit's own first-to-last-spike
    span is deliberate and, on a concatenated timeline, unambiguous: the
    electrode was recording for the whole window whether or not a given unit
    fired across all of it, and fourier_analysis derives one
    population-wide T, so a span-based denominator would plot a unit against
    a duration the y axis doesn't use.
    """
    return np.array([len(spkt) / window_s for spkt in spks])


def compute_fr_df(spks, FOURIER_Q, workers=1):
    """Panel B's data: each unit's NFC at each modulation amplitude, against
    its firing rate over the common window (see unit_firing_rates).
    """
    tasks = [(A,) for A in [0, 0.3, 0.6]]
    results = _run_parallel(tasks, _fr_cell, spks, FREQ, FOURIER_Q, workers,
                             desc=f"FR vs NFC ({len(tasks)} cells)")
    rows = []
    for A, NFCs in results:
        rows.append(pd.DataFrame({
            "mod": A,
            "FR": unit_firing_rates(spks),
            "NFC": NFCs,
            "id": np.arange(len(spks)),
        }))
    return pd.concat(rows)


def compute_sensitivity(spks, FOURIER_Q, freq=FREQ):
    """Per-unit detectability proxy (spk_count / T / 2 / sigma) from a
    baseline (unmodulated) Fourier pass at this simulation's own freq/Q --
    the same formula compute_responder_df already computes internally to
    rank/pool units for its participation sweep, factored out here so
    "top decile of sensitive neurons" (panel C's reduced population, panel
    D's black-outlined scatter points) is defined the same way, from one
    shared per-unit ranking. This is a synthetic-baseline metric, NOT the
    REAL per-unit sens column in data/manuscript/all_fourier_df.parquet
    that fig4_pilot.py used to select its own population (computed at the
    real magnetic stimulus frequency) -- reusing that would pull in an
    aggregate.py/parquet dependency this file otherwise doesn't have.
    """
    (C, T, spk_count, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, NFC0) = \
        statistics.fourier_analysis(spks, freq, Q=FOURIER_Q)
    sigma = statistics.get_sgm(fou_alt_c)
    return spk_count / T / 2 / sigma


def _responder_cell(task):
    """Composes one grid cell's NFC vector from compute_responder_df's
    precomputed lookup table (_worker_state["nfc_table"]/["baseline_NFC"])
    instead of calling warp_mod/fourier_analysis directly -- see that
    function's docstring for why this is exact, not an approximation: a
    unit's NFC at a given amplitude doesn't depend on which OTHER units are
    simultaneously modulated, so the table already holds, for every unit,
    exactly the value it would have gotten from a direct per-cell
    computation. Only the FDR step below -- which genuinely does depend on
    the whole cell's composed vector -- still runs per cell.

    `upper=` pins corrected_pvalues onto the sweep's single shared null
    distribution (see compute_responder_df), so the per-cell cost here is
    an interpolation plus a sort rather than a fresh O(len(support)^2)
    convolution -- which, left per-cell, was ~1.5 s and accounted for
    essentially the entire runtime of this sweep.
    """
    participation, pool_tuple, amplitude, repeat = task
    Q = _worker_state["Q"]
    nfc_table = _worker_state["nfc_table"]
    baseline_NFC = _worker_state["baseline_NFC"]
    NFC = baseline_NFC.copy()
    if pool_tuple:
        pool_idx = np.array(pool_tuple, dtype=int)
        NFC[pool_idx] = nfc_table[amplitude][pool_idx]
    pvals = statistics.corrected_pvalues(NFC, Q, upper=_worker_state["pvalue_upper"])
    qvals, pi0 = statistics.storey_qvalues(pvals, lambda_=0.5)
    return {
        "amplitude": amplitude,
        "participation": participation,
        "repeat": repeat,
        "responders": int(np.sum(qvals < QVALUE_FDR)),
        "pi0": pi0,
    }


def compute_responder_df(spks, FOURIER_Q, workers=1, freq=FREQ, n_repeats=N_REPEATS):
    """Ports fig4_pilot.py's compute_responder_df onto this file's own
    (full, un-subsampled) pseudopopulation -- see module docstring.

    Returns ALL `n_repeats` raw random-pool-draw rows per (participation,
    amplitude) cell, tagged by a `repeat` column -- NOT pre-averaged.
    Averaging over `repeat` is deliberately left to the caller (see
    plot_fig4's explicit groupby-mean step) so the load -> average -> plot
    data flow stays visible from the plotting code alone, rather than being
    baked silently into this function's return value.

    NFC lookup-table optimization: fourier_analysis's per-unit NFC value
    (see allfourier/get_sgm/get_NFC in magpyneto2/statistics.py) depends
    only on that unit's own (possibly warped) spike train, `freq`, `Q`, and
    `T` -- NOT on which other units are simultaneously in the modulated
    pool. So instead of re-running warp_mod+fourier_analysis once per grid
    cell (up to len(PARTICIPATION_RESP) * n_repeats * len(AMPLITUDES_RESP)
    times), we run it once per AMPLITUDES_RESP value -- warping EVERY unit
    at that amplitude in one batched call -- to build `nfc_table[amplitude]`,
    a length-n vector holding each unit's "if modulated at this amplitude"
    NFC. Composing a specific grid cell's NFC vector is then just a cheap
    array gather from this table (see _responder_cell), and only the
    genuinely cell-dependent FDR step (corrected_pvalues/storey_qvalues --
    Storey's FDR is a whole-vector order statistic, not cacheable) still
    runs per cell. That FDR step is only cheap because `pvalue_upper`
    below pins one shared null distribution across the sweep; see there.

    `T` (fourier_analysis's own latest-minus-earliest-spike-time span) is a
    population-wide quantity, so it's pinned once from the very first
    (fully unmodulated) call and passed explicitly (T=T) to every other
    table-building call -- otherwise fourier_analysis would implicitly
    recompute T per call from whichever specific subset happens to be
    warped, which the table's cross-amplitude composition can't tolerate
    (a unit's table entry at one amplitude must sit on the same frequency
    grid as its entry at every other amplitude, and as the baseline). In
    practice this pinning changes nothing: warp_mod only shifts a spike
    within its own stimulus period, so it essentially never moves the
    population's overall earliest/latest spike enough to change T's
    ceil()'d integer value.

    Pool membership is drawn randomly per repeat via one
    `np.random.default_rng(repeat)` Generator per repeat index (seeds
    0..n_repeats-1), each reused/advanced across ascending `participation`
    values within its own repeat (not re-seeded per participation, which
    would otherwise correlate pools across participation levels within the
    same repeat). This replaces the old deterministic evenly-rank-spaced
    pool selection, which produced a banding/staircase artifact in panels
    C/D; averaging over independently-drawn repeats (see plot_fig4) is what
    washes that artifact out.
    """
    n = len(spks)

    # NFC lookup table -- see docstring above. This first, unmodulated call
    # also pins T for every table entry built below.
    #
    # The table is where --workers earns its keep: one amplitude means
    # warping every unit and then a full-population fourier_analysis (~29 s
    # on the 1384-unit pigeon-HP pseudopopulation at FOURIER_Q=262), so the
    # ~100 amplitudes run ~50 min single-process and ~2.5 min at
    # --workers 24 -- while the responder sweep that consumes the table
    # takes seconds. The amplitudes are
    # mutually independent (each warps the same unmodulated `spks`, and its
    # NFC vector depends on nothing but its own amplitude), so they fan out
    # cleanly, each worker rendering its own progress bar.
    (C, T, spk_count, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, NFC0) = \
        statistics.fourier_analysis(spks, freq, Q=FOURIER_Q)
    # warp_mod at 0 is a no-op, so NFC0 above already covers that entry.
    table_tasks = [(float(amplitude),) for amplitude in AMPLITUDES_RESP if amplitude != 0.0]
    table_rows = _run_parallel(table_tasks, _nfc_table_cell, spks, freq, FOURIER_Q, workers,
                               desc=f"NFC lookup table ({len(table_tasks)} amplitudes)",
                               extra={"T": T})
    nfc_table = {0.0: NFC0}
    nfc_table.update(table_rows)
    baseline_NFC = NFC0

    # Shared null distribution for the whole sweep. corrected_pvalues would
    # otherwise re-derive its support from each cell's own max NFC and
    # rebuild the eps-corrected null from scratch every time -- a ~1.5 s
    # convolution per cell, i.e. essentially 100% of this function's
    # runtime, for a distribution that (eps being fixed by FOURIER_Q) is
    # the same one every cell needs. Pinning `upper` to cover the largest
    # NFC anywhere in the lookup table -- so no cell's units can be pushed
    # off the end of the grid -- lets statistics.corrected_null_grid's memo
    # hit on every call, and additionally makes every cell in the sweep
    # comparable against one identical null rather than each against its
    # own slightly-different grid.
    table_NFC = np.concatenate([np.asarray(v, dtype=float).ravel()
                                for v in nfc_table.values()])
    table_NFC = table_NFC[np.isfinite(table_NFC)]
    pvalue_upper = max(6.0, float(table_NFC.max()) * 1.05) if table_NFC.size else 6.0

    pvals0 = statistics.corrected_pvalues(baseline_NFC, FOURIER_Q, upper=pvalue_upper)
    qvals0, pi00 = statistics.storey_qvalues(pvals0, lambda_=0.5)
    baseline_responders = int(np.sum(qvals0 < QVALUE_FDR))

    tasks = []
    baseline_rows = []
    # One persistent Generator per repeat, seeded by its own repeat index and
    # advanced across ascending participation levels (not re-seeded per
    # participation) -- see docstring above.
    rngs = [np.random.default_rng(repeat) for repeat in range(n_repeats)]
    for participation in PARTICIPATION_RESP:
        pool_size = int(participation * n)  # truncate, matching the notebook
        for repeat in range(n_repeats):
            if pool_size == 0:
                pool = np.array([], dtype=int)
            else:
                pool = rngs[repeat].choice(n, size=pool_size, replace=False)
            # Baseline (amplitude=0.0) is bit-identical regardless of pool
            # composition (warp_mod at amplitude 0 is a no-op) -- replicated
            # flatly across repeats anyway so every (participation,
            # amplitude) pair, baseline included, has exactly n_repeats rows,
            # keeping the schema uniform for the downstream groupby-mean.
            baseline_rows.append({
                "amplitude": 0.0, "participation": participation,
                "repeat": repeat,
                "responders": baseline_responders, "pi0": pi00,
            })
            pool_tuple = tuple(pool.tolist())
            for amplitude in AMPLITUDES_RESP:
                if amplitude == 0.0:
                    # Already covered by baseline_rows above -- AMPLITUDES_RESP
                    # may or may not include a literal 0.0 entry itself, so
                    # guard against building a second, redundant row for the
                    # same (participation, 0.0, repeat), which would make the
                    # groupby/pivot() below raise on a duplicate index.
                    continue
                tasks.append((participation, pool_tuple, amplitude, repeat))

    # The sweep itself is ~0.1 ms/cell now that every cell shares one null
    # distribution (see _responder_cell), so at this grid resolution it
    # finishes in seconds in-process -- less than a Pool costs to spawn on
    # Windows, and every worker would need its own copy of the table on top
    # of that. Fan it out only once the grid is big enough to pay for it.
    sweep_workers = workers if len(tasks) >= _SWEEP_PARALLEL_MIN_CELLS else 1
    rows = _run_parallel(tasks, _responder_cell, spks, freq, FOURIER_Q, sweep_workers,
                          desc=f"responder sweep ({len(tasks)} cells, {n_repeats} repeats)",
                          extra={"nfc_table": nfc_table, "baseline_NFC": baseline_NFC,
                                 "pvalue_upper": pvalue_upper})
    return pd.DataFrame(baseline_rows + rows)


def plot_fig4(NFC_modulation_FR_df, resp_df, resp_df_top, top_decile_mask, spks, FOURIER_Q, out_dir: Path):
    example_spk = spks[len(spks) // 2]
    n_top = int(np.sum(top_decile_mask))

    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY_XL}
    matplotlib.rc("font", **font)

    fig = plt.figure(figsize=FP.FIGSIZE_FIG4)
    # Two row-bands, each with its own 1x2 column split, rather than one
    # 3x6 grid. The top band is panel A (three mod-condition columns, its
    # own nested gs_A below) beside panel B (the FR-vs-NFC scatter); the
    # bottom band is C and D side by side -- the SAME q-value
    # responder-count heatmap analysis, C on the full pseudopopulation, D
    # restricted to the top-decile-sensitivity subset (see
    # compute_sensitivity) -- given the whole band width (instead of half
    # the figure height each, stacked, as in an earlier layout) so each
    # comes out closer to square. Variable names still reflect what each
    # axis plots, not its current letter (e.g. ax_scatter is panel B here,
    # ax_heatmap is panel C, ax_heatmap_top is panel D).
    #
    # Separate per-band splits specifically so the two bands' column
    # boundaries can differ: A needs more width than B (three sub-panels
    # against one scatter), while C and D want an EVEN split. A single
    # shared 6-column grid couldn't express that -- A and C spanned the same
    # columns, so widening A necessarily widened C too. The cost is that the
    # panel letters no longer land on one rectangle (B's and D's sit at
    # different x); that's accepted deliberately.
    #
    # height_ratios/hspace here are the values that reproduce the previous
    # 3-row grid's vertical layout: that grid's top band spanned two rows
    # PLUS the gap between them, so its band-to-band proportion was
    # 1.04:1.00, not the 1.2:1.6 its raw height_ratios suggested.
    gs = gridspec.GridSpec(2, 1, left=0, bottom=0, right=1, top=1, hspace=0.29,
                           height_ratios=[1.04, 1.0])

    # Top band. width_ratios tilt toward A; wspace is wider than the bottom
    # band's because this gap has to keep panel A's rightmost PSTH twin axis
    # (its "PSTH (Hz)" label and tick labels, which sit on the axis's RIGHT
    # side) clear of B's "NFC" -- without the extra room the two y-labels
    # end up collinear. Both are fixed point sizes that don't shrink with
    # the figure, so the room has to be taken from B's width rather than
    # scaled.
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],
                                              wspace=0.32, width_ratios=[1.43, 1.0])
    # Bottom band: even split, so C is exactly as wide as D. wspace holds
    # C's colorbar tick labels + colorbar label against D's y tick labels
    # (D's y-label itself is dropped, see below).
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.2)
    # Panel A's raster/spectra rows get their own nested gridspec so their
    # vertical gap can be tightened independently of the outer hspace (which
    # still needs to separate this whole A/B band from C/D below).
    # Raster+PSTH on TOP, spectrum BELOW: the raster is the raw observation
    # and the spectrum is what the analysis makes of it, so reading the
    # column top-to-bottom now follows that order.
    # wspace here is well below the outer grid's: panel A packs three columns
    # into half the figure width, so at FIGSIZE_FIG4's 6 in each is under an
    # inch wide and a gap proportional to that width is mostly wasted space,
    # while the outer gaps still have to clear B's and C/D's y-axis labels.
    gs_A = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_top[0], hspace=0.4, wspace=0.3)

    ax_A2 = fig.add_subplot(gs_A[0, 0])  # raster+PSTH, A=0
    ax_A4 = fig.add_subplot(gs_A[0, 1])  # raster+PSTH, A=0.5
    ax_A6 = fig.add_subplot(gs_A[0, 2])  # raster+PSTH, A=1
    ax_A1 = fig.add_subplot(gs_A[1, 0])  # spectrum,    A=0
    ax_A3 = fig.add_subplot(gs_A[1, 1])  # spectrum,    A=0.5
    ax_A5 = fig.add_subplot(gs_A[1, 2])  # spectrum,    A=1
    ax_scatter     = fig.add_subplot(gs_top[1])  # FR-vs-NFC scatter, panel B
    ax_heatmap     = fig.add_subplot(gs_bot[0])  # full pseudopopulation, panel C
    ax_heatmap_top = fig.add_subplot(gs_bot[1])  # top-decile-sensitivity subset, panel D

    spectra_axes = [ax_A1, ax_A3, ax_A5]
    raster_axes  = [ax_A2, ax_A4, ax_A6]

    # Phase-raster window, shared by all three mod conditions so their cycle
    # counts (and hence y-limits) match. Snapped OUT to whole stimulus
    # periods around the example unit's own span: the floor keeps phase
    # anchored to t=0, which is where warp_mod's phase-0 modulation is
    # anchored too (so the modulation trough lands at the phase it should),
    # while still dropping the ~150 empty leading cycles a literal (0, max)
    # window would raster. warp_mod only moves spikes WITHIN their own
    # period, so this window is identical for every A.
    period = 1 / FREQ
    raster_window = (np.floor(example_spk.min() / period) * period,
                     np.ceil(example_spk.max() / period) * period)

    spectra_max = 0
    psth_axes = []
    for mod_i, A in enumerate([0, 0.5, 1]):
        warped = statistics.warp_mod(example_spk, A, 1 / FREQ, 0)
        (C, T, spk_count, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, NFC) = \
            statistics.fourier_analysis([warped], freq=FREQ, Q=FOURIER_Q)

        # Same modulus-consistent spectrum panel Fig1 uses (see
        # statistics.plot_spectrum): off-frequency |c_n| scatter, a sgm_c
        # reference line, and a stem+marker at the stimulus frequency for
        # |c_s|. Replaces an older real-component-only scatter, which plotted
        # a quantity that isn't what NFC actually compares against.
        # Deliberately achromatic: this figure is a simulation, and the
        # manuscript's blue/orange are reserved for magnetic vs. visual
        # stimulation elsewhere -- coloring a simulated spectrum with either
        # would read as a stimulus contingency it doesn't have.
        statistics.plot_spectrum(spectra_axes[mod_i], fou_alt.flatten(), ff_alt, FREQ, fou0,
                                 legend=False, dot_color=SPECTRUM_DOT_COLOR,
                                 stem_color="black", sigma_color=SIGMA_COLOR)
        spectra_axes[mod_i].set_xlabel("Hz")
        spectra_max = max(spectra_max, spectra_axes[mod_i].get_ylim()[1])
        if mod_i == 0:
            spectra_axes[mod_i].set_ylabel("Magnitude")
        else:
            spectra_axes[mod_i].set_yticks([])

        # Phase raster + smoothed-PSTH overlay, the same pair of helpers
        # Fig1's panels C/D use -- replaces a plain `warped % period`
        # histogram, which threw away the cycle-by-cycle structure the
        # modulation actually imposes. Both get the SAME (spks, window,
        # freq), so the curve summarizes exactly the ticks drawn under it.
        statistics.plot_phase_raster(raster_axes[mod_i], np.squeeze(warped), raster_window, FREQ,
                                     color=RASTER_COLOR, markersize=2, linewidth=0.5,
                                     pad_x=RASTER_PAD_X, pad_y=RASTER_PAD_Y)
        raster_axes[mod_i].set_title(f"A={A}", fontsize=FP.FS_TITLE)
        psth_axes.append(statistics.plot_smoothed_phase_psth(
            raster_axes[mod_i], np.squeeze(warped), raster_window, FREQ, color=PSTH_COLOR,
            n_bins=PSTH_N_BINS, smooth_bins=PSTH_SMOOTH_BINS, alpha=1.0))
        # After the PSTH overlay, not before: plot_smoothed_phase_psth
        # re-applies statistics._PHASE_TICKS to the same axes.
        raster_axes[mod_i].set_xticks(RASTER_PHASE_TICKS)
        raster_axes[mod_i].set_xticklabels(RASTER_PHASE_TICKLABELS)
        if mod_i != 0:
            raster_axes[mod_i].set_ylabel("")
            raster_axes[mod_i].set_yticks([])

    # Shared magnitude scale across the three mod conditions -- required for
    # the yticks-stripped panels 2/3 to be readable off panel 1's axis, and
    # it's the growth of |c_s| relative to a roughly fixed |c_n| cloud that
    # the row is there to show.
    [ax.set_ylim((0, spectra_max * 1.05)) for ax in spectra_axes]

    # Same treatment for the PSTH twin axes, mirrored: one shared rate scale,
    # but labelled on the RIGHTMOST panel (where a twinx' axis naturally
    # lives) rather than the leftmost, so the raster's "Cycle" axis and the
    # PSTH's rate axis bracket the row instead of stacking three twin axes
    # into the row's already-tight wspace.
    psth_max = max(ax.get_ylim()[1] for ax in psth_axes)
    for mod_i, psth_ax in enumerate(psth_axes):
        # Headroom above (as for the spectra) plus a matching gap below, so a
        # trough that reaches 0 Hz doesn't get drawn on top of the raster's
        # bottom spine.
        psth_ax.set_ylim(-psth_max * PSTH_PAD_Y, psth_max * 1.05)
        if mod_i != len(psth_axes) - 1:
            psth_ax.set_ylabel("")
            psth_ax.set_yticks([])
            psth_ax.spines["right"].set_visible(False)
        else:
            psth_ax.set_ylabel("PSTH (Hz)", color=PSTH_COLOR, fontsize=FP.FS_LEGEND, labelpad=1)
            psth_ax.tick_params(axis="y", labelsize=FP.FS_LEGEND, pad=1)
            # The negative bottom limit is padding, not data -- drop any tick
            # the locator puts below 0, which would read as a negative rate.
            psth_ax.set_yticks([t for t in psth_ax.get_yticks() if 0 <= t <= psth_max * 1.05])

    # Panel C (q-value/FDR responder-count heatmap, ported from
    # fig4_pilot.py's plot_fig4_pilot), on this file's own full
    # pseudopopulation.
    #
    # Repeat averaging (explicit on purpose): compute_responder_df returns
    # ALL N_REPEATS raw random-pool-draw rows per (participation, amplitude)
    # cell (see its docstring), tagged by `repeat`, NOT pre-averaged.
    # Averaging happens HERE, as its own step, so this load -> average ->
    # plot pipeline stays visible from the plotting code alone, without
    # digging into compute_responder_df. Only "responders" (the plotted
    # value) is averaged; `pi0`/`repeat` are dropped by the groupby.
    resp_df_mean = (
        resp_df.groupby(["participation", "amplitude"])["responders"].mean().reset_index()
    )
    pivot = resp_df_mean.pivot(index="participation", columns="amplitude", values="responders")
    im = ax_heatmap.imshow(
        pivot.values, aspect="auto", origin="lower",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
    )
    # No label on C's colorbar: it is the same quantity as D's, and D's
    # colorbar -- the rightmost thing in the row -- carries the label for
    # both. (The two colorbars' SCALES differ, C's running to the full
    # population and D's to the top decile, which is why both keep their own
    # tick labels.) Mirrors the same call made on D's y-axis, which C's
    # y-label covers.
    fig.colorbar(im, ax=ax_heatmap)
    contour_level = 10 # max(1, len(spks) // 10)
    ax_heatmap.contour(pivot.columns, pivot.index, pivot.values, levels=[contour_level],
                        colors="white", linestyles="dashed")
    ax_heatmap.set_xlabel("5 Hz modulation amplitude (A)")
    ax_heatmap.set_ylabel("Fraction of population modulated")
    # Titles name each panel's population, since C and D are the SAME
    # analysis and differ only in which units it runs over. D's percentage
    # is derived from SENSITIVITY_PERCENTILE rather than written out, so the
    # title can't drift away from the threshold actually applied.
    ax_heatmap.set_title("All pigeon HP units", fontsize=FP.FS_TITLE)

    # Panel D: the SAME analysis as panel C, restricted to the top-decile
    # (>=90th percentile) most-sensitive units (see compute_sensitivity) --
    # the pseudopopulation subsampling fig4_pilot.py did unconditionally,
    # now shown here as an explicit comparison against panel C's full
    # population rather than a replacement for it.
    #
    # Same repeat-averaging as panel C above.
    resp_df_top_mean = (
        resp_df_top.groupby(["participation", "amplitude"])["responders"].mean().reset_index()
    )
    pivot_top = resp_df_top_mean.pivot(index="participation", columns="amplitude", values="responders")
    im_top = ax_heatmap_top.imshow(
        pivot_top.values, aspect="auto", origin="lower",
        extent=[pivot_top.columns.min(), pivot_top.columns.max(), pivot_top.index.min(), pivot_top.index.max()],
    )
    fig.colorbar(im_top, ax=ax_heatmap_top, label="Number of responders (q < 0.05)")
    contour_level_top = 10 # max(1, n_top // 10)
    ax_heatmap_top.contour(pivot_top.columns, pivot_top.index, pivot_top.values, levels=[contour_level_top],
                           colors="white", linestyles="dashed")
    ax_heatmap_top.set_xlabel("5 Hz modulation amplitude (A)")
    ax_heatmap_top.set_title(
        f"{100 - SENSITIVITY_PERCENTILE:.0f}% most sensitive pigeon HP units",
        fontsize=FP.FS_TITLE)
    # No y label: D's y axis is the same quantity as C's, on the same scale,
    # immediately to its left -- C's label reads for both. Dropping it also
    # clears the collision it had with C's own colorbar label.
    ax_heatmap_top.set_ylabel("")

    # Panel B: FR vs NFC scatter. Top-decile-sensitivity units (same
    # definition as panel D's subset) get a black outline, overlaid on the
    # same points already drawn -- same hue/palette so the outlined points
    # keep their mod-condition fill color.
    sns.scatterplot(data=NFC_modulation_FR_df, x="FR", y="NFC", hue="mod",
                    palette=PALETTE_FIG4B, s=5, linewidth=0, alpha=FP.ALPHA_SCATTER, ax=ax_scatter)
    top_ids = set(np.where(top_decile_mask)[0].tolist())
    top_df = NFC_modulation_FR_df[NFC_modulation_FR_df["id"].isin(top_ids)]
    sns.scatterplot(data=top_df, x="FR", y="NFC", hue="mod", palette=PALETTE_FIG4B,
                    s=5, linewidth=0.6, edgecolor="black", alpha=FP.ALPHA_SCATTER,
                    ax=ax_scatter, legend=False)
    handles, labels = ax_scatter.get_legend_handles_labels()
    mod_vals = [0, 0.3, 0.6]
    pairs = [(h, l) for h, l in zip(handles, labels)
             if any(abs(float(l) - v) < 1e-9 for v in mod_vals)]
    pairs = pairs[::-1]  # high-to-low top-to-bottom, instead of seaborn's ascending hue order
    ax_scatter.legend([h for h, _ in pairs],
                [f"A={l}" for _, l in pairs],
                title="modulation (5 Hz)", ncol=1, markerscale=3,
                fontsize=FP.FS_LEGEND, title_fontsize=FP.FS_LEGEND)
    # eps-corrected to match NFC_modulation_FR_df, which is computed by
    # compute_fr_df() at this same Q_frac-derived FOURIER_Q.
    ax_scatter.hlines(statistics.inverse_Rayleigh_CDF(0.99, eps=statistics.get_epsilon(FOURIER_Q)),
                *ax_scatter.get_xlim(), color="grey")
    ax_scatter.set_ylabel("NFC")
    ax_scatter.set_xscale("log")
    ax_scatter.set_xlabel("Firing rate (Hz)")

    # Panel letters: placed in FIGURE coordinates, offset by the same fixed
    # pad from each panel's own top-left corner, so all four sit on a
    # consistent rectangle (the two top-band letters share a y, the two
    # bottom-row letters share a y, and the two left-column letters share an
    # x). Axes-fraction offsets can't do this -- the same fraction means a
    # different physical distance on each panel, which is why "A" (on one
    # short spectra axes, offset far enough up to clear its "A=0" title)
    # used to sit well outside the others. gs spans the full figure
    # (left=0/bottom=0/right=1/top=1) and no auto-layout engine is active,
    # so ax.get_position() is already the final figure-coordinate box.
    PANEL_PAD_X = 0.035   # figure fraction, left of the panel's left edge
    PANEL_PAD_Y = 0.02    # figure fraction, above the panel's top edge

    def panel_letter(ax, letter):
        bbox = ax.get_position()
        fig.text(bbox.x0 - PANEL_PAD_X, bbox.y1 + PANEL_PAD_Y, letter,
                 fontfamily="arial", fontsize=12,
                 ha="left", va="bottom")

    panel_letter(ax_A2, "A")
    panel_letter(ax_scatter, "B")
    panel_letter(ax_heatmap, "C")
    panel_letter(ax_heatmap_top, "D")

    # Endpoint ticks plus one at FREQ itself, so each spectrum states the
    # frequency being driven rather than leaving it to be read off the
    # stem's position. The "Hz" label is deliberately NOT nestled back up
    # between the ticks the way it was before the FREQ tick existed -- a
    # nestled label sits at the axis center, which is exactly where the
    # 5 Hz tick label now lands (the window is symmetric about FREQ).
    for _ax in spectra_axes:
        statistics.boundary_ticks(_ax, y=False)
        statistics.stimulus_frequency_tick(_ax, FREQ)
    # The raster row's "Phase (rad)" is deliberately NOT nestled. A nestled
    # label is placed at a fixed axes FRACTION below the axis, which pinned
    # it much closer to its tick labels than the spectra row's "Hz" sits to
    # its own -- matplotlib places an un-nestled label by a fixed point
    # offset from the tick labels instead. Leaving both rows un-nestled is
    # what makes the two label gaps equal.

    # Open (top/right-despined) boxes throughout, matching Fig1's style. Done
    # in one pass at the end so it also catches the heatmaps, whose top/right
    # spines just re-outlined the imshow extent. The PSTH twin axes are
    # excluded -- their data axis IS the right spine, and only the rightmost
    # of them still shows it (see the loop above); fig.axes' colorbar axes
    # are excluded for the same reason (a despined colorbar reads as broken).
    _cbar_axes = {im.colorbar.ax for im in (im, im_top)}
    for _ax in fig.axes:
        if _ax in psth_axes or _ax in _cbar_axes:
            continue
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

    out_path = out_dir / "Fig4.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=FP.DPI)
    print(f"Saved {out_path}")
    if not in_notebook:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 4 (modulation sensitivity simulation)")
    parser.add_argument("--out-dir", default=FP.OUT_DIR, help="Output directory for PDFs")
    parser.add_argument("--data-dir", default=FP.DATA_DIR, help="Directory containing pipeline output .nwb files")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Explicit experiment names to pool (default: auto-discover "
                             "every experiment YAML with species=='Pigeon' and area=='HP')")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for the panel B/C/D sweeps. Worth setting: it "
                             "fans out the per-amplitude NFC lookup table, which dominates "
                             "runtime (~50 min at --workers 1, ~2.5 min at --workers 24). Each "
                             "worker holds its own copy of the pooled spike trains")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute simulation even if cached pickles exist")
    args = parser.parse_args([] if in_notebook else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(FR_DF_CACHE).parent.mkdir(parents=True, exist_ok=True)

    experiments = args.experiments or discover_pigeon_hp_experiments(_EXPERIMENTS_DIR)
    print(f"Pooling pigeon-HP pseudopopulation from: {experiments}")
    spks, Q_frac = load_pseudopopulation_spks(args.data_dir, experiments)

    FOURIER_Q = fourier_Q_from_frac(spks, FREQ, Q_frac, context=f"[fig4 sim @ {FREQ}Hz] ")
    print(f"  Q_frac={Q_frac} (shared mag_Q_frac across pooled experiments) -> FOURIER_Q={FOURIER_Q} bins at {FREQ} Hz")

    if not args.recompute and Path(FR_DF_CACHE).exists():
        print(f"Loading cached FR df from {FR_DF_CACHE}")
        NFC_modulation_FR_df = pd.read_pickle(FR_DF_CACHE)
    else:
        print("Computing FR vs NFC...")
        NFC_modulation_FR_df = compute_fr_df(spks, FOURIER_Q, workers=args.workers)
        NFC_modulation_FR_df.to_pickle(FR_DF_CACHE)
        print(f"Cached -> {FR_DF_CACHE}")

    if not args.recompute and Path(RESP_DF_CACHE).exists():
        print(f"Loading cached resp_df from {RESP_DF_CACHE}")
        resp_df = pd.read_pickle(RESP_DF_CACHE)
    else:
        print("Computing modulation amplitude x participation -> responder count (slow)...")
        resp_df = compute_responder_df(spks, FOURIER_Q, workers=args.workers)
        resp_df.to_pickle(RESP_DF_CACHE)
        print(f"Cached -> {RESP_DF_CACHE}")

    # Top-decile-sensitivity subset for panel C/D -- see compute_sensitivity.
    # Cheap (one un-swept Fourier pass over the full population), so always
    # recomputed regardless of --recompute; only the responder sweep on the
    # resulting subset is cached.
    sens = compute_sensitivity(spks, FOURIER_Q, freq=FREQ)
    sens_threshold = np.quantile(sens, SENSITIVITY_PERCENTILE / 100)
    top_decile_mask = sens >= sens_threshold
    top_decile_spks = [s for s, m in zip(spks, top_decile_mask) if m]
    print(f"  {len(top_decile_spks)} / {len(spks)} units at/above the "
          f"{SENSITIVITY_PERCENTILE:.0f}th percentile synthetic-baseline "
          f"sensitivity (threshold={sens_threshold:.3f})")

    if not args.recompute and Path(RESP_DF_TOP_DECILE_CACHE).exists():
        print(f"Loading cached top-decile resp_df from {RESP_DF_TOP_DECILE_CACHE}")
        resp_df_top = pd.read_pickle(RESP_DF_TOP_DECILE_CACHE)
    else:
        print("Computing modulation amplitude x participation -> responder count "
              "(top-decile-sensitivity subset, slow)...")
        resp_df_top = compute_responder_df(top_decile_spks, FOURIER_Q, workers=args.workers)
        resp_df_top.to_pickle(RESP_DF_TOP_DECILE_CACHE)
        print(f"Cached -> {RESP_DF_TOP_DECILE_CACHE}")

    plot_fig4(NFC_modulation_FR_df, resp_df, resp_df_top, top_decile_mask, spks, FOURIER_Q, out_dir)


if __name__ == "__main__":
    if in_notebook:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    main()
