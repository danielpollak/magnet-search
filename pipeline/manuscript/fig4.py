"""Fig 4 — modulation strength vs firing rate and q-value responder counts.

Pools a pigeon-HP "pseudopopulation" -- real ("mag" contingency) spike
trains from every experiment YAML with species=="Pigeon" and area=="HP" --
then synthetically modulates the pooled units to show detection thresholds.
Pooling this way (instead of a single recording) keeps Fig4's population
consistent with the "Pigeon HP" population Fig2/Fig3 already report on.
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

# _pigeon_hp_pseudopop suffix: deliberately distinct from the old
# single-recording cache filenames (modulation_strength_vs_excess_count.pkl /
# modulation_strength_vs_FR.pkl) so a stale single-session cache can never be
# silently reused now that the population is pooled across experiments. The
# old files, if present, are simply orphaned -- harmless, untracked. (The
# excess-count cache itself -- CI_DF_CACHE -- was dropped along with the
# panel it fed; qvalue_responder_df_pigeon_hp_pseudopop.pkl, if a stale copy
# lingers, is unaffected.) FR_DF_CACHE additionally carries a "_v3" suffix:
# compute_fr_df's FR values have changed meaning twice now -- v2 normalized
# by each unit's own experiment's mag-trial window T; v3 (current) instead
# bounds T by each unit's own first-to-last-spike span, a strictly tighter
# per-unit bound -- so any older _pigeon_hp_pseudopop.pkl / "_v2".pkl left
# on disk is stale and must not be silently reloaded.
FR_DF_CACHE   = _REPO_ROOT / "data" / "manuscript" / "modulation_strength_vs_FR_pigeon_hp_pseudopop_v3.pkl"
# "_lineargrid" suffix: AMPLITUDES_RESP/PARTICIPATION_RESP went through a
# log-spaced detour (several different ranges) before settling back on a
# plain linear arange -- any older qvalue_responder_df*.pkl (any "_log*"
# suffix) holds responder counts indexed by one of those OLD grids and must
# not be silently reloaded as if it matched the current one.
RESP_DF_CACHE = _REPO_ROOT / "data" / "manuscript" / "qvalue_responder_df_pigeon_hp_pseudopop_lineargrid.pkl"
RESP_DF_TOP_DECILE_CACHE = _REPO_ROOT / "data" / "manuscript" / "qvalue_responder_df_pigeon_hp_pseudopop_top_decile_lineargrid.pkl"

SENSITIVITY_PERCENTILE = 90.0  # panel C / panel D outline: top decile by compute_sensitivity

FREQ        = 5

# Panel A palette. Everything in this panel is achromatic on purpose: Fig4 is
# a SIMULATION, and the manuscript reserves blue/orange (FP.COLOR_MAG /
# FP.COLOR_VIS) for real magnetic vs. visual stimulation -- using either here
# would imply a stimulus contingency these synthetic spike trains don't have.
# Within the panel, darkness encodes what's primary: the on-frequency stem
# (the quantity the panel is about) is black, the off-frequency |c_n| cloud
# and the raster ticks are mid-grey, and the PSTH curve -- a visual aid that
# plays no part in the Fourier/NFC computation -- is lighter still.
SPECTRUM_DOT_COLOR = "0.45"
SIGMA_COLOR        = "0.6"
RASTER_COLOR       = "0.45"
PSTH_COLOR         = "0.7"

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
AMPLITUDES_RESP    = np.arange(0, 1, 0.01)
PARTICIPATION_RESP = np.arange(0, 1, 0.01)
QVALUE_FDR         = 0.05


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


def load_unit_spks_for_experiment(data_dir: str, experiment: str):
    """Load one pigeon-HP experiment's "mag"-contingency spike trains, one
    pooled spike array per real unit (pooled across that experiment's own
    mag trial recs -- they share a single Kilosort sort, same as
    fit_fourier_sig's own per-unit grouping).

    Returns (unit_spks, Q_frac) where unit_spks maps
    f"{experiment}:{cluster_id}" -> spike-time array. The namespaced key
    only exists to keep this experiment's units distinct when pooled with
    other experiments' units downstream -- cluster_id alone is not globally
    unique (see fig2.py/fig3.py's species/date/id dedup key).
    """
    # Reads the pipeline's own {experiment}.nwb (Phase 7 cutover -- no
    # paradigm writes the legacy {experiment}_processing.pickle anymore),
    # same reproducible-from-NWB principle applied to fig1.py.
    cfg = load_experiment(_EXPERIMENTS_DIR / f"{experiment}.yml")
    nwb_path = Path(data_dir) / f"{experiment}.nwb"
    io_r, nwbfile = nwb_io.read_nwbfile(str(nwb_path))
    modulation_df = nwb_io.build_modulation_frame(nwbfile, good_only=cfg.good)
    io_r.close()

    # Same plain substring test analysis_stages/multistim.py uses to build
    # its mag_mask -- "mag" contingency is the real magnetic-trial recs,
    # excluding auxiliary_stimuli (visual/WN/oddball) recs, which are always
    # "positive control".
    mag_substr = cfg.analysis.mag_rec_substring
    mag_df = modulation_df.loc[[mag_substr in rec for rec in modulation_df.rec]]
    if mag_df.empty:
        available = modulation_df.rec.unique().tolist()
        raise ValueError(
            f"No recs matching mag_rec_substring={mag_substr!r} in "
            f"{nwb_path.name}. Available recs: {available}"
        )

    unit_spks = {
        f"{experiment}:{uid}": g.spk.values
        for uid, g in mag_df.groupby("id") if len(g) > 5
    }
    print(f"  {experiment}: {len(unit_spks)} mag units with >5 spikes")

    Q_frac = cfg.analysis.mag_Q_frac if cfg.analysis.mag_Q_frac > 0 else cfg.analysis.Q_frac
    if Q_frac <= 0:
        raise ValueError(
            f"{experiment}.yml has no analysis.mag_Q_frac/Q_frac set -- "
            f"Fig4's simulation derives its off-frequency bin count from "
            f"this experiment's own Q_frac, same as the real per-experiment "
            f"analysis."
        )
    return unit_spks, Q_frac


def load_pseudopopulation_spks(data_dir: str, experiments):
    """Pool load_unit_spks_for_experiment() across every given experiment
    into one pigeon-HP pseudopopulation, sorted by spike count ascending
    (plot_fig4 picks its example unit via spks[len(spks) // 2], i.e. the
    median-by-spike-count unit -- same convention the old single-recording
    load_spks used).
    """
    all_units = {}
    q_fracs = {}
    for experiment in experiments:
        unit_spks, Q_frac = load_unit_spks_for_experiment(data_dir, experiment)
        all_units.update(unit_spks)
        q_fracs[experiment] = Q_frac

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
    print(f"  {len(spks)} pigeon-HP pseudopopulation units pooled from {len(experiments)} experiments")
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


def _init_worker(spks, freq, Q):
    _worker_state["spks"] = spks
    _worker_state["freq"] = freq
    _worker_state["Q"] = Q
    _worker_state["eps"] = statistics.get_epsilon(Q)


def _init_worker_bars(spks, freq, Q, worker_fn, lock):
    """Same as _init_worker, plus stashing worker_fn itself (so _run_chunk
    below can look it up per-process) and re-installing the shared tqdm
    lock -- tqdm's own recipe for rendering multiple bars from separate
    processes without their line-redraws stomping on each other.
    """
    _init_worker(spks, freq, Q)
    _worker_state["worker_fn"] = worker_fn
    tqdm.tqdm.set_lock(lock)


def _run_chunk(indexed_chunk):
    idx, chunk = indexed_chunk
    worker_fn = _worker_state["worker_fn"]
    bar = tqdm.tqdm(chunk, desc=f"worker {idx}", position=idx, leave=True)
    return idx, [worker_fn(t) for t in bar]


def _run_parallel(tasks, worker_fn, spks, freq, Q, workers, desc=""):
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
    """
    if workers <= 1:
        _init_worker(spks, freq, Q)
        return [worker_fn(t) for t in tqdm.tqdm(tasks, desc=desc)]
    workers = min(workers, len(tasks)) or 1
    chunks = [tasks[i::workers] for i in range(workers)]
    lock = RLock()
    with Pool(workers, initializer=_init_worker_bars,
              initargs=(spks, freq, Q, worker_fn, lock)) as pool:
        chunk_results = pool.map(_run_chunk, list(enumerate(chunks)))
    chunk_results.sort(key=lambda ic: ic[0])
    return [r for _, chunk in chunk_results for r in chunk]


def _fr_cell(task):
    (A,) = task
    spks = _worker_state["spks"]
    freq = _worker_state["freq"]
    Q = _worker_state["Q"]
    modulated = [statistics.modulate(spkt, freq, A) for spkt in spks]
    (C, _, spk_count, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, NFCs) = \
        statistics.fourier_analysis(modulated, freq, Q=Q)
    return A, NFCs


def compute_fr_df(spks, FOURIER_Q, workers=1):
    """Each unit's firing rate is its own spike count divided by its own
    first-to-last-spike span (`spkt.max() - spkt.min()`) -- bounded per
    unit, not by an experiment-wide or pooled-population-wide window. A
    unit's real observation window (a mag-trial epoch, or several pooled
    together) is generally longer than the interval between its own first
    and last spike, so this is a strictly tighter (smaller) T than either
    of those alternatives -- it can only push a unit's displayed FR up,
    never down, relative to a window-based T, which is the direction that
    shrinks the low-FR pileup this was tuned to address.
    """
    tasks = [(A,) for A in [0, 0.3, 0.6]]
    results = _run_parallel(tasks, _fr_cell, spks, FREQ, FOURIER_Q, workers,
                             desc=f"FR vs NFC ({len(tasks)} cells)")
    rows = []
    for A, NFCs in results:
        rows.append(pd.DataFrame({
            "mod": A,
            "FR": [len(spkt) / (spkt.max() - spkt.min()) for spkt in spks],
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
    participation, pool_tuple, amplitude = task
    spks = _worker_state["spks"]
    freq = _worker_state["freq"]
    Q = _worker_state["Q"]
    pool_set = set(pool_tuple)
    modulated = [
        statistics.warp_mod(spkt, amplitude, 1 / freq, 0) if i in pool_set else spkt
        for i, spkt in enumerate(spks)
    ]
    (_, _, _, _, _, _, _, _, _, NFC) = statistics.fourier_analysis(modulated, freq, Q=Q)
    pvals = statistics.corrected_pvalues(NFC, Q)
    qvals, pi0 = statistics.storey_qvalues(pvals, lambda_=0.5)
    return {
        "amplitude": amplitude,
        "participation": participation,
        "responders": int(np.sum(qvals < QVALUE_FDR)),
        "pi0": pi0,
    }


def compute_responder_df(spks, FOURIER_Q, workers=1, freq=FREQ):
    """Ports fig4_pilot.py's compute_responder_df onto this file's own
    (full, un-subsampled) pseudopopulation -- see module docstring. Baseline
    (amplitude=0.0) is identical for every participation level (warp_mod at
    amplitude 0 is a no-op), so it's computed once outside the swept grid,
    same shortcut the pilot used.
    """
    (C, T, spk_count, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, NFC0) = \
        statistics.fourier_analysis(spks, freq, Q=FOURIER_Q)
    sigma = statistics.get_sgm(fou_alt_c)
    sens = spk_count / T / 2 / sigma
    order = np.argsort(sens)  # ascending, same as the notebook
    n = len(spks)

    pvals0 = statistics.corrected_pvalues(NFC0, FOURIER_Q)
    qvals0, pi00 = statistics.storey_qvalues(pvals0, lambda_=0.5)
    baseline_responders = int(np.sum(qvals0 < QVALUE_FDR))

    tasks = []
    baseline_rows = []
    for participation in PARTICIPATION_RESP:
        pool_size = int(participation * n)  # truncate, matching the notebook
        if pool_size == 0:
            pool = np.array([], dtype=int)
        else:
            pool = order[np.round(np.linspace(0, n - 1, pool_size)).astype(int)]
        baseline_rows.append({
            "amplitude": 0.0, "participation": participation,
            "responders": baseline_responders, "pi0": pi00,
        })
        pool_tuple = tuple(pool.tolist())
        for amplitude in AMPLITUDES_RESP:
            if amplitude == 0.0:
                # Already covered by baseline_rows above (warp_mod at
                # amplitude 0 is a no-op regardless of which pool is
                # selected) -- AMPLITUDES_RESP may or may not include a
                # literal 0.0 entry itself, so guard against building a
                # second, redundant row for the same (participation, 0.0)
                # pair, which would make the pivot() below raise on a
                # duplicate index.
                continue
            tasks.append((participation, pool_tuple, amplitude))

    rows = _run_parallel(tasks, _responder_cell, spks, freq, FOURIER_Q, workers,
                          desc=f"responder sweep ({len(tasks)} cells)")
    return pd.DataFrame(baseline_rows + rows)


def plot_fig4(NFC_modulation_FR_df, resp_df, resp_df_top, top_decile_mask, spks, FOURIER_Q, out_dir: Path):
    example_spk = spks[len(spks) // 2]
    n_top = int(np.sum(top_decile_mask))

    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY_XL}
    matplotlib.rc("font", **font)

    fig = plt.figure(figsize=FP.FIGSIZE_FIG4)
    # 3 rows x 6 cols. Top row-band (grid rows 0-1, squished vertically via
    # height_ratios below) is split left/right down the middle: the left
    # half is panel A (spectra on grid row 0, PSTHs on grid row 1 -- one
    # column per mod condition, squished into 3 of the 6 columns instead of
    # spanning the full width); the right half is B (the FR-vs-NFC scatter,
    # spanning both rows next to A). The bottom row holds C and D
    # side-by-side -- the SAME q-value responder-count heatmap analysis, C
    # on the full pseudopopulation, D restricted to the top-decile-
    # sensitivity subset (see compute_sensitivity) -- given the whole bottom
    # row (instead of half the figure height each, stacked, as in the
    # previous layout) so each comes out closer to square. Variable names
    # still reflect what each axis plots, not its current letter (e.g.
    # ax_scatter is panel B here, ax_heatmap is panel C, ax_heatmap_top is
    # panel D).
    gs = gridspec.GridSpec(3, 6, left=0, bottom=0, right=1, top=1, wspace=0.5, hspace=0.5,
                            height_ratios=[0.6, 0.6, 1.6])
    # Panel A's raster/spectra rows get their own nested gridspec so their
    # vertical gap can be tightened independently of the outer hspace (which
    # still needs to separate this whole A/B band from C/D below).
    # Raster+PSTH on TOP, spectrum BELOW: the raster is the raw observation
    # and the spectrum is what the analysis makes of it, so reading the
    # column top-to-bottom now follows that order.
    gs_A = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0:2, 0:3], hspace=0.25, wspace=0.5)

    ax_A2 = fig.add_subplot(gs_A[0, 0])  # raster+PSTH, A=0
    ax_A4 = fig.add_subplot(gs_A[0, 1])  # raster+PSTH, A=0.5
    ax_A6 = fig.add_subplot(gs_A[0, 2])  # raster+PSTH, A=1
    ax_A1 = fig.add_subplot(gs_A[1, 0])  # spectrum,    A=0
    ax_A3 = fig.add_subplot(gs_A[1, 1])  # spectrum,    A=0.5
    ax_A5 = fig.add_subplot(gs_A[1, 2])  # spectrum,    A=1
    ax_scatter     = fig.add_subplot(gs[0:2, 3:6])  # FR-vs-NFC scatter, panel B
    ax_heatmap     = fig.add_subplot(gs[2, 0:3])    # full pseudopopulation, panel C
    ax_heatmap_top = fig.add_subplot(gs[2, 3:6])    # top-decile-sensitivity subset, panel D

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
                                     color=RASTER_COLOR, markersize=2, linewidth=0.5)
        raster_axes[mod_i].set_title(f"A={A}", fontsize=FP.FS_TITLE)
        psth_axes.append(statistics.plot_smoothed_phase_psth(
            raster_axes[mod_i], np.squeeze(warped), raster_window, FREQ, color=PSTH_COLOR))
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
        psth_ax.set_ylim(0, psth_max * 1.05)  # headroom, as for the spectra above
        if mod_i != len(psth_axes) - 1:
            psth_ax.set_ylabel("")
            psth_ax.set_yticks([])
            psth_ax.spines["right"].set_visible(False)
        else:
            psth_ax.set_ylabel("PSTH (Hz)", color=PSTH_COLOR, fontsize=FP.FS_LEGEND, labelpad=1)
            psth_ax.tick_params(axis="y", labelsize=FP.FS_LEGEND, pad=1)

    # Panel C (q-value/FDR responder-count heatmap, ported from
    # fig4_pilot.py's plot_fig4_pilot), on this file's own full
    # pseudopopulation.
    pivot = resp_df.pivot(index="participation", columns="amplitude", values="responders")
    im = ax_heatmap.imshow(
        pivot.values, aspect="auto", origin="lower",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
    )
    fig.colorbar(im, ax=ax_heatmap, label="Number of responders (q < 0.05)")
    contour_level = 10 # max(1, len(spks) // 10)
    ax_heatmap.contour(pivot.columns, pivot.index, pivot.values, levels=[contour_level],
                        colors="white", linestyles="dashed")
    ax_heatmap.set_xlabel("5 Hz modulation amplitude (A)")
    ax_heatmap.set_ylabel("Fraction of population modulated")

    # Panel D: the SAME analysis as panel C, restricted to the top-decile
    # (>=90th percentile) most-sensitive units (see compute_sensitivity) --
    # the pseudopopulation subsampling fig4_pilot.py did unconditionally,
    # now shown here as an explicit comparison against panel C's full
    # population rather than a replacement for it.
    pivot_top = resp_df_top.pivot(index="participation", columns="amplitude", values="responders")
    im_top = ax_heatmap_top.imshow(
        pivot_top.values, aspect="auto", origin="lower",
        extent=[pivot_top.columns.min(), pivot_top.columns.max(), pivot_top.index.min(), pivot_top.index.max()],
    )
    fig.colorbar(im_top, ax=ax_heatmap_top, label="Number of responders (q < 0.05)")
    contour_level_top = 10 # max(1, n_top // 10)
    ax_heatmap_top.contour(pivot_top.columns, pivot_top.index, pivot_top.values, levels=[contour_level_top],
                           colors="white", linestyles="dashed")
    ax_heatmap_top.set_xlabel("5 Hz modulation amplitude (A)")
    ax_heatmap_top.set_ylabel("Fraction of population modulated")

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

    statistics.boundarize_and_nestle(ax_A1, y=False, x_offset=-0.06)
    statistics.boundarize_and_nestle(ax_A3, y=False, x_offset=-0.06)
    statistics.boundarize_and_nestle(ax_A5, y=False, x_offset=-0.06)
    # Raster row's x_offset is more negative than the spectra row's -- these
    # axes got shorter under the new height_ratios (row A is squished to
    # make room for panels C/D below), and unlike the spectra row's edge-only
    # ticks (3, 7), the raster's phase axis carries interior ticks
    # (pi/2, pi, 3pi/2) right where a -0.1-nestled "Phase (rad)" would land.
    statistics.nestle_labels(ax_A2, y=False, x_offset=-0.16)
    statistics.nestle_labels(ax_A4, y=False, x_offset=-0.16)
    statistics.nestle_labels(ax_A6, y=False, x_offset=-0.16)

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
                        help="Parallel workers for the panel B/C/D sweeps (each sweep's grid "
                             "cells are independent -- try e.g. --workers 8)")
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
