"""Fig 4 — modulation strength vs excess suspects, firing rate, and q-value
responder counts.

Pools a pigeon-HP "pseudopopulation" -- real ("mag" contingency) spike
trains from every experiment YAML with species=="Pigeon" and area=="HP" --
then synthetically modulates the pooled units to show detection thresholds.
Pooling this way (instead of a single recording) keeps Fig4's population
consistent with the "Pigeon HP" population Fig2/Fig3 already report on.
Panel C ports fig4_pilot.py's Storey q-value/FDR responder-count heatmap
onto this SAME full pseudopopulation (no percentile subsampling -- pilot's
own >=90th-percentile-sensitivity subsampling was pilot-only scaffolding;
kept broad here for consistency with the other panels).

Requires:
  data/{experiment}.nwb for every discovered pigeon-HP experiment
  seaborn

Cached simulation results are saved to data/manuscript/ on first run and
reused on subsequent runs (pass --recompute to force a fresh simulation).
Every sweep (panels B, C, D) is embarrassingly parallel across its grid
cells -- pass --workers N (N>1) to fan them out across a multiprocessing
Pool, same convention as pipeline/processing.py's/analysis.py's --workers.

Usage:
    python pipeline/manuscript/fig4.py
    python pipeline/manuscript/fig4.py --workers 8
    python pipeline/manuscript/fig4.py --experiments 20230413_firstsite 20230415
    python pipeline/manuscript/fig4.py --recompute
"""
import argparse
import os
from multiprocessing import Pool
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
# old files, if present, are simply orphaned -- harmless, untracked.
CI_DF_CACHE   = _REPO_ROOT / "data" / "manuscript" / "modulation_strength_vs_excess_count_pigeon_hp_pseudopop.pkl"
FR_DF_CACHE   = _REPO_ROOT / "data" / "manuscript" / "modulation_strength_vs_FR_pigeon_hp_pseudopop.pkl"
RESP_DF_CACHE = _REPO_ROOT / "data" / "manuscript" / "qvalue_responder_df_pigeon_hp_pseudopop.pkl"

FREQ        = 5
A_L         = np.linspace(0, 0.3, num=11)
PERCENTAGES = np.linspace(0, 1, num=5)

# Panel C (ported from fig4_pilot.py) sweep grids -- matches Markus Meister's
# MM_Analysis_3.ipynb notebook's own grids (finer/wider than A_L/PERCENTAGES
# above, which is panel B's own coarser sweep). Named "_RESP" (for the
# responder-count sweep computed below), not "_C" -- the panel letters this
# maps to are a display choice made once, in plot_fig4, not baked into
# identifiers here.
AMPLITUDES_RESP    = np.arange(0.1, 1.01, 0.1)   # amplitude=0.0 (baseline) handled separately
PARTICIPATION_RESP = np.arange(0.0, 1.01, 0.1)
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
# Every grid cell below (one (percent, A) pair for panel B, one A for panel
# D, one (participation, amplitude) pair for panel C) is an independent call
# to statistics.fourier_analysis over the SAME pooled `spks` -- embarrassingly
# parallel. _init_worker stashes the shared, possibly-large `spks` list once
# per worker process (via Pool's initializer) instead of re-pickling it into
# every task, and the worker functions below read it back out of that
# per-process global. With --workers 1 (the default) _run_parallel skips
# Pool entirely and calls _init_worker/the worker function directly in this
# process, so the single- and multi-process code paths share one
# implementation of each sweep's actual math.
# ---------------------------------------------------------------------------
_worker_state = {}


def _init_worker(spks, freq, Q):
    _worker_state["spks"] = spks
    _worker_state["freq"] = freq
    _worker_state["Q"] = Q
    _worker_state["eps"] = statistics.get_epsilon(Q)


def _run_parallel(tasks, worker_fn, spks, freq, Q, workers, desc=""):
    if workers <= 1:
        _init_worker(spks, freq, Q)
        return [worker_fn(t) for t in tqdm.tqdm(tasks, desc=desc)]
    with Pool(workers, initializer=_init_worker, initargs=(spks, freq, Q)) as pool:
        return list(tqdm.tqdm(pool.imap(worker_fn, tasks), total=len(tasks), desc=desc))


def _ci_cell(task):
    percent_i, percent, A = task
    spks = _worker_state["spks"]
    freq = _worker_state["freq"]
    Q = _worker_state["Q"]
    eps = _worker_state["eps"]

    modulated = [None] * len(spks)
    for spk_i, spkt in enumerate(spks):
        if spk_i % len(PERCENTAGES) < percent_i:
            modulated[spk_i] = statistics.warp_mod(spkt, A, 1 / freq, 0)
        else:
            modulated[spk_i] = spkt
    (C, T, spk_count, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, NFCs) = \
        statistics.fourier_analysis(modulated, freq, Q=Q)
    n_empirical, f_expected, l_bound, h_bound = \
        statistics.suspect_count_significance(NFCs, 0.99, conf_int_α=0.05, eps=eps)
    return {"mod": A, "ci": (l_bound, h_bound), "mid": n_empirical, "%": percent}


def compute_ci_df(spks, FOURIER_Q, workers=1):
    tasks = [
        (percent_i, percent, A)
        for percent_i, percent in enumerate(PERCENTAGES)
        for A in A_L
    ]
    rows = _run_parallel(tasks, _ci_cell, spks, FREQ, FOURIER_Q, workers,
                          desc=f"excess-count sweep ({len(tasks)} cells)")
    return pd.DataFrame(rows)


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
    T = statistics.latesttime(spks) - statistics.earliesttime(spks)
    tasks = [(A,) for A in [0, 0.3, 0.6]]
    results = _run_parallel(tasks, _fr_cell, spks, FREQ, FOURIER_Q, workers,
                             desc=f"FR vs NFC ({len(tasks)} cells)")
    rows = []
    for A, NFCs in results:
        rows.append(pd.DataFrame({
            "mod": A,
            "FR": [len(spkt) / T for spkt in spks],
            "NFC": NFCs,
            "id": np.arange(len(spks)),
        }))
    return pd.concat(rows)


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
            tasks.append((participation, pool_tuple, amplitude))

    rows = _run_parallel(tasks, _responder_cell, spks, freq, FOURIER_Q, workers,
                          desc=f"responder sweep ({len(tasks)} cells)")
    return pd.DataFrame(baseline_rows + rows)


def plot_fig4(ci_df, NFC_modulation_FR_df, resp_df, spks, FOURIER_Q, out_dir: Path):
    example_spk = spks[len(spks) // 2]

    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY_XL}
    matplotlib.rc("font", **font)

    fig = plt.figure(figsize=FP.FIGSIZE_FIG4)
    # 3 rows x 6 cols. Top row-band (grid rows 0-1) is split left/right down
    # the middle: the left half is panel A (spectra on grid row 0, PSTHs on
    # grid row 1 -- one column per mod condition, squished into 3 of the 6
    # columns instead of spanning the full width); the right half is B
    # (grid row 0) stacked directly on top of C, the q-value responder
    # heatmap (grid row 1) -- so B/C together occupy the same left-right
    # half and top-bottom split as A. D (the FR-vs-NFC scatter) is the only
    # thing below that top band, spanning the full width on its own row.
    # Panel letters C/D are swapped from their code names below
    # (ax_heatmap -> "C", ax_scatter -> "D") per explicit request --
    # variable names still reflect what each axis plots, not its letter.
    gs = gridspec.GridSpec(3, 6, left=0, bottom=0, right=1, top=1, wspace=0.5, hspace=0.5)

    ax_A1 = fig.add_subplot(gs[0, 0])  # spectrum, A=0
    ax_A3 = fig.add_subplot(gs[0, 1])  # spectrum, A=0.5
    ax_A5 = fig.add_subplot(gs[0, 2])  # spectrum, A=1
    ax_A2 = fig.add_subplot(gs[1, 0])  # PSTH,     A=0
    ax_A4 = fig.add_subplot(gs[1, 1])  # PSTH,     A=0.5
    ax_A6 = fig.add_subplot(gs[1, 2])  # PSTH,     A=1
    ax_B       = fig.add_subplot(gs[0, 3:6])
    ax_heatmap = fig.add_subplot(gs[1, 3:6])
    ax_scatter = fig.add_subplot(gs[2, :])

    spectra_axes = [ax_A1, ax_A3, ax_A5]
    psth_axes    = [ax_A2, ax_A4, ax_A6]

    psth_max = 0
    for mod_i, A in enumerate([0, 0.5, 1]):
        warped = statistics.warp_mod(example_spk, A, 1 / FREQ, 0)
        (C, T, spk_count, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, NFC) = \
            statistics.fourier_analysis([warped], freq=FREQ, Q=FOURIER_Q)

        spectra_axes[mod_i].plot(ff_alt, fou_alt.real.T, ".", color="orange")
        spectra_axes[mod_i].plot(fff[i0], fou0.real.T, ".")
        spectra_axes[mod_i].set_xlabel("Hz")
        spectra_axes[mod_i].set_title(f"A={A}")
        if mod_i == 0:
            spectra_axes[mod_i].set_ylabel("Real component")
        else:
            spectra_axes[mod_i].set_yticks([])

        counts, _, _ = psth_axes[mod_i].hist(np.squeeze(warped % (1 / FREQ)), bins=25)
        psth_max = max(psth_max, counts.max())
        psth_axes[mod_i].set_xlabel("Time (s)")
        if mod_i == 0:
            psth_axes[mod_i].set_ylabel("Spike counts")
        else:
            psth_axes[mod_i].set_yticks([])

    [ax.set_ylim((-1, 7)) for ax in spectra_axes]
    # Headroom scaled off the tallest bin actually observed across the three
    # panels (was a hardcoded (0, 30) that clipped once the pigeon-HP
    # pseudopopulation's higher-spike-count example units pushed bin counts
    # past 30).
    [ax.set_ylim((0, psth_max * 1.1)) for ax in psth_axes]

    conf_limits = ci_df.ci[0]
    ax_B.fill_between([0, ci_df["mod"].max()], conf_limits[0], conf_limits[1],
                      color="grey", alpha=0.5)
    cmap = plt.cm.viridis
    percent_color_d = {p: cmap.colors[255 // 4 * pi]
                       for pi, p in enumerate(PERCENTAGES)}

    for percent, pct_df in ci_df.groupby("%"):
        ax_B.plot(pct_df["mod"], pct_df["mid"], color=percent_color_d[percent])
        for ind, (_, row) in enumerate(pct_df.iterrows()):
            ax_B.scatter(row["mod"], row.mid, color=percent_color_d[percent],
                         label=f"{int(percent * 100)}%" if ind == 0 else None)

    ax_B.legend(title=f"% units\nmodulated\n(N={len(spks)})")
    ax_B.set_xticks(A_L)
    ax_B.set_xticklabels(A_L, rotation=0)
    ax_B.set_xlabel("5 Hz modulation amplitude")
    ax_B.set_ylabel("Excess suspects")

    # Panel C (q-value/FDR responder-count heatmap, ported from
    # fig4_pilot.py's plot_fig4_pilot), on this file's own full
    # pseudopopulation -- not the pilot's >=90th-percentile-sensitivity
    # subsample.
    pivot = resp_df.pivot(index="participation", columns="amplitude", values="responders")
    im = ax_heatmap.imshow(
        pivot.values, aspect="auto", origin="lower",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
    )
    fig.colorbar(im, ax=ax_heatmap, label="Number of responders (q < 0.05)")
    contour_level = max(1, len(spks) // 10)
    ax_heatmap.contour(pivot.columns, pivot.index, pivot.values, levels=[contour_level],
                        colors="white", linestyles="dashed")
    ax_heatmap.set_xlabel("Modulation amplitude")
    ax_heatmap.set_ylabel("Participation level")

    # Panel D: FR vs NFC scatter.
    sns.scatterplot(data=NFC_modulation_FR_df, x="FR", y="NFC", hue="mod",
                    palette="Set1", s=5, linewidth=0, alpha=FP.ALPHA_SCATTER, ax=ax_scatter)
    handles, labels = ax_scatter.get_legend_handles_labels()
    mod_vals = [0, 0.3, 0.6]
    pairs = [(h, l) for h, l in zip(handles, labels)
             if any(abs(float(l) - v) < 1e-9 for v in mod_vals)]
    ax_scatter.legend([h for h, _ in pairs],
                [f"A={l}" for _, l in pairs],
                title="modulation (5Hz)", ncol=3)
    # eps-corrected to match NFC_modulation_FR_df, which is computed by
    # compute_fr_df() at this same Q_frac-derived FOURIER_Q -- same corrected
    # null used for ci_df's confidence bounds in panel B.
    ax_scatter.hlines(statistics.inverse_Rayleigh_CDF(0.99, eps=statistics.get_epsilon(FOURIER_Q)),
                *ax_scatter.get_xlim(), color="grey")
    ax_scatter.set_ylabel(r"$\hat{c}$")
    ax_scatter.set_xscale("log")
    ax_scatter.set_xlabel("Firing rate (Hz)")

    # Panel letters: each row is now uniform height within itself, so a
    # simple per-axis axes-fraction annotate suffices (no cross-row
    # figure-coordinate alignment hack needed, unlike the earlier layout
    # where A/B/D shared one taller row-span).
    ax_A1.annotate("A", xy=(-0.12, 1.35), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    ax_B.annotate("B", xy=(-0.12, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    ax_heatmap.annotate("C", xy=(-0.05, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)
    ax_scatter.annotate("D", xy=(-0.03, 1.05), xycoords="axes fraction", fontfamily="arial", fontsize=12)

    statistics.boundarize_and_nestle(ax_A1, y=False, x_offset=-0.1)
    statistics.boundarize_and_nestle(ax_A3, y=False, x_offset=-0.1)
    statistics.boundarize_and_nestle(ax_A5, y=False, x_offset=-0.1)
    statistics.nestle_labels(ax_A2, y=False, x_offset=-0.1)
    statistics.nestle_labels(ax_A4, y=False, x_offset=-0.1)
    statistics.nestle_labels(ax_A6, y=False, x_offset=-0.1)

    xticks = ax_B.get_xticks()
    ax_B.set_xticks([xticks[0], xticks[-1]])
    yticks = ax_B.get_yticks()
    ax_B.set_yticks([yticks[1], yticks[-1]])
    statistics.nestle_labels(ax_B, x_offset=-0.05, y_offset=-0.05)

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
    Path(CI_DF_CACHE).parent.mkdir(parents=True, exist_ok=True)

    experiments = args.experiments or discover_pigeon_hp_experiments(_EXPERIMENTS_DIR)
    print(f"Pooling pigeon-HP pseudopopulation from: {experiments}")
    spks, Q_frac = load_pseudopopulation_spks(args.data_dir, experiments)

    FOURIER_Q = fourier_Q_from_frac(spks, FREQ, Q_frac, context=f"[fig4 sim @ {FREQ}Hz] ")
    print(f"  Q_frac={Q_frac} (shared mag_Q_frac across pooled experiments) -> FOURIER_Q={FOURIER_Q} bins at {FREQ} Hz")

    if not args.recompute and Path(CI_DF_CACHE).exists():
        print(f"Loading cached ci_df from {CI_DF_CACHE}")
        ci_df = pd.read_pickle(CI_DF_CACHE)
    else:
        print("Computing modulation strength vs excess count (slow)...")
        ci_df = compute_ci_df(spks, FOURIER_Q, workers=args.workers)
        ci_df.to_pickle(CI_DF_CACHE)
        print(f"Cached -> {CI_DF_CACHE}")

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

    plot_fig4(ci_df, NFC_modulation_FR_df, resp_df, spks, FOURIER_Q, out_dir)


if __name__ == "__main__":
    if in_notebook:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    main()
