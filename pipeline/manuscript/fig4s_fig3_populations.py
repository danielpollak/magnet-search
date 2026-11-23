"""Fig 4 supplement -- Fig 4C's detectability heatmap, run on Fig 3's own
populations: (left) Fig 3A's spiking neurons across all species, (right)
Fig 3B's pigeon neurons.

Each neuron contributes the ONE recording Fig 3 tests it in
(fig3.one_recording_per_neuron: its magnetic recording closest to 3 Hz), at
that recording's own length, its own off-frequency window (the experiment's
Q_frac, as fit_fourier_sig applies it) and fit_fourier_sig's own T
(max(period) / freq). Synthetic time-warp modulation is injected at fig4.FREQ
(4 Hz) in every recording and tested there. Everything else is Fig 4C's
sweep: for each (amplitude A, fraction f) cell, a random fraction f of the
population is modulated at A, every neuron gets a p-value from the
eps-corrected null at its own Q, Storey q-values are taken over the whole
population, and the cell holds the number of neurons with q < 0.05, averaged
over fig4.N_REPEATS random draws of the modulated pool. Dashed contour:
fig4.DETECTION_RESPONDERS responders.

GCaMP imaging (zebrafish, medaka) is left out: the simulation warps spike
times. Fig 3 has no owl neurons. The mouse's 49 neurons come from its sorted
spike trains in data/precomputed/mouse/ (the 2 Hz recording, which is the one
Fig 3 uses, i.e. the one closest to 3 Hz), between its first and last magnet
trigger, with the 100 off-frequency bins its own analysis used.

Caveat (accepted, 2026-11-10): 335 of the 402 quail recordings and 12 zebra
finch recordings Fig 3 uses were themselves 4 Hz magnetic stimulation.

--quadrants (2026-11-11): Fig4S_fig3_populations_60s.pdf, the same sweep on
all four of Fig 3's populations, laid out in Fig 3's quadrants: A all
spiking neurons, B all pigeon neurons, C pigeon hippocampus neurons, D the
10% most sensitive pigeon hippocampus neurons (Fig 3D's own selection: the
parquet's `sens` above the 90th percentile of Fig 3's pigeon HP neurons,
then kept if their recording survives the cap). Same REC_CAP_S cap. Its
cache (QUAD_CACHE) holds the per-neuron p-value table, so the four sweeps
come from one NFC computation.

Cached results: data/manuscript/fig3pop_detectability_*.pkl (pass --recompute).

Usage:
    python pipeline/manuscript/fig4s_fig3_populations.py --workers 24
    python pipeline/manuscript/fig4s_fig3_populations.py --workers 24 --quadrants
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import tqdm.auto as tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from magpyneto2 import statistics
from pipeline import nwb_io
from pipeline.schema import load_experiment

import format_parameters as FP
import fig3
import fig4

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS_DIR = _REPO_ROOT / "experiments"
_MOUSE_DIR = _REPO_ROOT / "data" / "precomputed" / "mouse"
IMAGING_SPECIES = ("zebrafish", "medaka")
# Every recording is capped at its first REC_CAP_S seconds and recordings
# shorter than that are dropped (2026-11-11, user's call), so recording
# length no longer differs between neurons or species. Before the cap the
# lengths were pigeon 25-139 s (median 60), zebra finch 44-161 s (80), quail
# 240-274 s and mouse 718 s, and the long quail/mouse recordings made the
# all-species population look more detectable than its pigeon subset.
REC_CAP_S = 60.0
MOUSE_Q_BINS = 100  # the mouse analysis's own off-frequency bin count (mouse_raw_data.py, section 6)
_TAG = f"f{fig4.FREQ:g}Hz_cap{REC_CAP_S:g}s_lineargrid_halfpart_{fig4.N_REPEATS}rep"
CACHE = _REPO_ROOT / "data" / "manuscript" / f"fig3pop_detectability_{_TAG}.pkl"


def quad_cache_path(cap_s, stim_freq=False):
    """Per-neuron p-value table cache for the four-quadrant sweep; cap_s=None
    is Fig 3's own spike trains at their full analysed length; stim_freq:
    modulation at each recording's own stimulus frequency, not fig4.FREQ."""
    cap = "nocap" if cap_s is None else f"cap{cap_s:g}s"
    freq = "fstim" if stim_freq else f"f{fig4.FREQ:g}Hz"
    return (_REPO_ROOT / "data" / "manuscript" /
            f"fig3pop_detectability_quadrants_{freq}_{cap}_lineargrid_halfpart_{fig4.N_REPEATS}rep.pkl")


QUAD_CACHE = quad_cache_path(REC_CAP_S)
SENS_PERCENTILE = 90  # Fig 3D's top-decile cut (fig3.plot_fig3's conditions)
NEURON_KEY = ["species", "ID", "date", "id"]


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------
def fig3_populations():
    """Fig 3A's spiking neurons and Fig 3B's pigeon neurons, one row each:
    exactly the rows fig3.py plots (same waves, same per-neuron recording)."""
    df = pd.read_parquet(_REPO_ROOT / "data" / "manuscript" / "all_fourier_df.parquet")
    neg, _, _ = statistics.get_poscontrols_negresults(df)
    one = fig3.one_recording_per_neuron(fig3._split_into_waves(neg))
    spiking = one.loc[~one.species.isin(IMAGING_SPECIES)].reset_index(drop=True)
    return spiking


def rec_to_experiment():
    """rec name -> experiment (NWB) name, over every spiking experiment YAML."""
    out = {}
    for yml in sorted(_EXPERIMENTS_DIR.glob("*.yml")):
        cfg = load_experiment(yml)
        if cfg.paradigm in ("engert", "medaka", "manual"):
            continue
        nwb_path = _REPO_ROOT / "data" / f"{cfg.name}.nwb"
        if not nwb_path.exists():
            continue
        io_r, nwbfile = nwb_io.read_nwbfile(str(nwb_path))
        recs = nwb_io.read_fourier_results_as_full_fourier_df(nwbfile).rec.unique()
        io_r.close()
        for rec in recs:
            if rec in out and out[rec][0] != cfg.name:
                raise ValueError(f"rec {rec!r} appears in both {out[rec][0]} and {cfg.name}")
            out[rec] = (cfg.name, yml)
    return out


def _q_frac(cfg):
    return cfg.analysis.mag_Q_frac if cfg.analysis.mag_Q_frac > 0 else cfg.analysis.Q_frac


def load_groups(pop):
    """One group per (experiment, rec): the population's neurons recorded in
    it, their spike trains, T, and the Q_frac / bin count the real analysis
    used. Checks that the reconstructed T and Q reproduce the parquet's own
    off-frequency bin count and NFC at the recording's real frequency."""
    rec_map = rec_to_experiment()
    groups = []
    npix = pop.loc[pop.species != "mouse"]
    unmapped = sorted(set(npix.rec) - set(rec_map))
    if unmapped:
        raise ValueError(f"recs with no experiment: {unmapped[:5]} ...")
    npix = npix.assign(experiment=[rec_map[r][0] for r in npix.rec])
    for experiment, edf in tqdm.tqdm(npix.groupby("experiment"), desc="loading NWBs"):
        cfg = load_experiment(rec_map[edf.rec.iloc[0]][1])
        io_r, nwbfile = nwb_io.read_nwbfile(str(_REPO_ROOT / "data" / f"{experiment}.nwb"))
        mod = nwb_io.build_modulation_frame(nwbfile, good_only=cfg.good)
        epochs = nwbfile.intervals["stimulus_epochs"].to_dataframe()
        io_r.close()
        for rec, rdf in edf.groupby("rec"):
            mrec = mod.loc[mod.rec == rec]
            f_real = float(rdf.freq.iloc[0])
            mrec = mrec.loc[np.isclose(mrec.freq, f_real)]
            T = (1 / f_real) * np.max(mrec.period)
            by_id = {i: np.sort(g.spk.values) for i, g in mrec.groupby("id")}
            ids = rdf.id.to_numpy()
            missing = [i for i in ids if i not in by_id]
            if missing:
                raise ValueError(f"{experiment} {rec}: units {missing[:5]} have no spikes")
            spks = [by_id[i] for i in ids]
            Q_frac = _q_frac(cfg)
            M_real = statistics.bins_for_fraction(f_real, Q_frac, 1 / T)
            if not np.allclose(M_real, rdf.Q.to_numpy()):
                raise ValueError(f"{experiment} {rec}: Q {M_real} != parquet {rdf.Q.unique()}")
            NFC_real = statistics.fourier_analysis(spks, f_real, Q=M_real, T=T)[-1]
            if not np.allclose(NFC_real, rdf.NFC.to_numpy(), rtol=1e-6, atol=1e-9):
                raise ValueError(f"{experiment} {rec}: NFC at {f_real} Hz does not reproduce the parquet's")
            # Window start in the spikes' own (recording-local) frame, as
            # fig4.concat_mag_recs derives it.
            ep = epochs.loc[epochs.rec == rec]
            if len(ep) != 1:
                raise ValueError(f"{experiment} {rec}: {len(ep)} stimulus epochs, expected 1")
            t0 = float(ep.start_time.iloc[0] - ep.local_offset_seconds.iloc[0])
            lo = min(x.min() for x in spks if len(x))
            hi = max(x.max() for x in spks if len(x))
            tol = 1 / f_real  # one stimulus period: T counts whole periods
            if lo < t0 - tol or hi > t0 + T + tol:
                raise ValueError(f"{experiment} {rec}: spikes {lo:.2f}-{hi:.2f} s outside "
                                 f"the epoch {t0:.2f}-{t0 + T:.2f} s")
            groups.append(dict(rows=rdf.index.to_numpy(), spks=spks, T=T, t0=t0, Q_frac=Q_frac,
                               f=f_real))
    mouse = pop.loc[pop.species == "mouse"]
    if len(mouse):
        groups.append(_mouse_group(mouse))
    return groups


def _mouse_group(mouse):
    (rec,) = mouse.rec.unique()
    d = _MOUSE_DIR / f"{rec}.rhd"
    trig = sio.loadmat(d / "magstim.mat")["magstim_trigger"].ravel()
    st = sio.loadmat(d / "spiketrain.mat")
    by_id = {int(c): x.ravel() for c, x in zip(st["cellid"].ravel(), st["spiketrain"].ravel())}
    t0, t1 = trig[0], trig[-1]
    spks = []
    for i in mouse.id.to_numpy():
        s = by_id[int(i)]
        spks.append(np.sort(s[(s >= t0) & (s < t1)] - t0))
    (f,) = mouse.freq.unique()
    return dict(rows=mouse.index.to_numpy(), spks=spks, T=float(t1 - t0), t0=0.0, Q_bins=MOUSE_Q_BINS,
                f=float(f))


def cap_groups(groups, cap_s=REC_CAP_S, stim_freq=False):
    """Every recording cut to its first `cap_s` seconds (from its stimulus
    epoch start); recordings shorter than `cap_s` are dropped. Q is
    re-derived for the capped length (Q_frac at fig4.FREQ, 1/cap_s
    resolution); the mouse keeps the bin count its own analysis used.
    cap_s=None: no cut -- every recording keeps the spike train and T Fig 3
    analysed (load_groups checks they reproduce the parquet's NFC), with Q
    at fig4.FREQ for that T.
    stim_freq: the modulation frequency `f` of each group is its recording's
    own stimulus frequency (the one Fig 3 tests) instead of fig4.FREQ; with
    cap_s=None, Q is then exactly the bin count of the real analysis."""
    kept = []
    for g in groups:
        f = g["f"] if stim_freq else fig4.FREQ
        if cap_s is None:
            Q = g.get("Q_bins") or statistics.bins_for_fraction(f, g["Q_frac"], 1 / g["T"])
            kept.append(dict(rows=g["rows"], spks=g["spks"], T=float(g["T"]), Q=Q, f=f))
            continue
        if g["T"] < cap_s:
            continue
        spks = [x[x < g["t0"] + cap_s] for x in g["spks"]]
        Q = g.get("Q_bins") or statistics.bins_for_fraction(f, g["Q_frac"], 1 / cap_s)
        kept.append(dict(rows=g["rows"], spks=spks, T=float(cap_s), Q=Q, f=f))
    return kept


# ---------------------------------------------------------------------------
# Per-neuron p-value table: p[a][i] = neuron i's p-value if modulated at a.
# A neuron's p-value depends only on its own spike train, T and Q, so the
# table is exact for any pool (same argument as fig4.compute_responder_df).
# ---------------------------------------------------------------------------
_G = {}


def _init(groups):
    _G["groups"] = groups


def _nfc_row(amplitude):
    out = []
    for g in _G["groups"]:
        warped = [statistics.warp_mod(s, amplitude, 1 / g["f"], 0) for s in g["spks"]]
        out.append(statistics.fourier_analysis(warped, g["f"], Q=g["Q"], T=g["T"])[-1])
    return amplitude, out


def pvalue_table(groups, n, workers):
    amps = [float(a) for a in fig4.AMPLITUDES_RESP]
    if workers > 1:
        with Pool(workers, initializer=_init, initargs=(groups,)) as pool:
            rows = list(tqdm.tqdm(pool.imap(_nfc_row, amps), total=len(amps), desc="NFC table"))
    else:
        _init(groups)
        rows = [_nfc_row(a) for a in tqdm.tqdm(amps, desc="NFC table")]
    nfc = dict(rows)
    P = np.full((len(amps), n), np.nan)
    for gi, g in enumerate(groups):
        G = np.array([np.asarray(nfc[a][gi], float) for a in amps])  # (amps, units in group)
        upper = max(6.0, float(np.nanmax(G)) * 1.05)
        for ai in range(len(amps)):
            P[ai, g["rows"]] = statistics.corrected_pvalues(G[ai], g["Q"], upper=upper)
    return np.array(amps), P


def responder_sweep(P, amps, members):
    """Fig 4C's sweep over the neurons `members` (row indices into P)."""
    P = P[:, members]
    n = P.shape[1]
    a0 = int(np.flatnonzero(amps == 0.0)[0])
    rows = []
    rngs = [np.random.default_rng(r) for r in range(fig4.N_REPEATS)]
    for participation in fig4.PARTICIPATION_RESP:
        k = int(participation * n)
        for r in range(fig4.N_REPEATS):
            pool = rngs[r].choice(n, size=k, replace=False) if k else np.array([], int)
            for ai, a in enumerate(amps):
                p = P[a0].copy()
                p[pool] = P[ai, pool]
                q, _ = statistics.storey_qvalues(p, lambda_=0.5)
                rows.append((a, participation, r, int(np.sum(q < fig4.QVALUE_FDR))))
    return pd.DataFrame(rows, columns=["amplitude", "participation", "repeat", "responders"])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
LOG_VMAX = 100.0


def _heatmap(fig, ax, resp, title, cbar_label=None, log=False):
    m = resp.groupby(["participation", "amplitude"])["responders"].mean().reset_index()
    pv = m.pivot(index="participation", columns="amplitude", values="responders")
    # log=True: logarithmic colour axis fixed at 1-LOG_VMAX responders, shared
    # by every panel; cells below 1 (incl. 0) / above LOG_VMAX take the
    # colormap's end colours (colorbar arrows).
    norm = mcolors.LogNorm(vmin=1, vmax=LOG_VMAX) if log else None
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under(cmap(0.0))
    cmap.set_over(cmap(1.0))
    # LogNorm masks 0 (drawn blank); lift it under vmin so it takes the under colour.
    vals = np.where(pv.values <= 0, 0.5, pv.values) if log else pv.values
    im = ax.imshow(vals, aspect="auto", origin="lower", norm=norm, cmap=cmap,
                   extent=[pv.columns.min(), pv.columns.max(), pv.index.min(), pv.index.max()])
    fig.colorbar(im, ax=ax, label=cbar_label, extend="both" if log else "neither")
    ax.contour(pv.columns, pv.index, pv.values, levels=[fig4.DETECTION_RESPONDERS],
               colors="white", linestyles="dashed")
    ax.set_xlabel(f"{fig4.FREQ:g} Hz modulation amplitude (A)")
    ax.set_title(title, fontsize=FP.FS_TITLE)


def fig3_quadrant_members(pop, kept, area="HP"):
    """Row indices (into `pop`, restricted to `kept`) of Fig 3's four
    quadrant populations, in fig3.plot_fig3's order. D's top decile is taken
    over Fig 3's own pigeon `area` neurons (the parquet's `sens`, strict >
    the percentile, as fig3.plot_uniform_p applies it) BEFORE the cap, so it
    is the same set of neurons Fig 3D shows, less those the cap drops."""
    area_name = {"HP": "hippocampus", "CB": "cerebellum"}[area]
    pigeon = pop.species.to_numpy() == "Pigeon"
    in_area = pigeon & (pop.area.to_numpy() == area)
    sens = pop.sens.to_numpy()
    top = in_area & (sens > np.percentile(sens[in_area], SENS_PERCENTILE))
    in_kept = np.zeros(len(pop), bool)
    in_kept[kept] = True
    return [
        ("All spiking neurons across all species", np.flatnonzero(in_kept)),
        ("All pigeon neurons", np.flatnonzero(in_kept & pigeon)),
        (f"Pigeon {area_name} neurons", np.flatnonzero(in_kept & in_area)),
        (f"Top {100 - SENS_PERCENTILE}% most sensitive pigeon {area} neurons",
         np.flatnonzero(in_kept & top)),
    ]


def quadrant_sweeps(cap_s=REC_CAP_S, workers=1, recompute=False, stim_freq=False):
    """Fig 3's four populations' responder sweeps: [(title, members, resp)],
    members being row indices into fig3_populations(). The cache stores the
    neurons' identities and is recomputed if the parquet's Fig 3 population
    no longer matches it (a positional cache would silently map onto other
    neurons after aggregate.py)."""
    pop = fig3_populations()
    ident = pop[NEURON_KEY + ["rec"]].astype(str).reset_index(drop=True)
    cache = quad_cache_path(cap_s, stim_freq)
    cached = None
    if cache.exists() and not recompute:
        cached = pd.read_pickle(cache)
        if len(cached) != 4 or not cached[3].equals(ident):
            print(f"{cache} is stale (Fig 3 population changed); recomputing")
            cached = None
        else:
            print(f"Loading cached p-value table from {cache}")
    if cached is None:
        groups = cap_groups(load_groups(pop), cap_s=cap_s, stim_freq=stim_freq)
        kept = np.sort(np.concatenate([g["rows"] for g in groups]))
        amps, P = pvalue_table(groups, len(pop), workers)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cached = (amps, P, kept, ident)
        pd.to_pickle(cached, cache)
        print(f"Cached -> {cache}")
    amps, P, kept, _ = cached
    quads = fig3_quadrant_members(pop, kept)
    return [(title, members, responder_sweep(P, amps, members))
            for title, members in tqdm.tqdm(quads, desc="sweeps")]


def main_quadrants(args):
    """Fig4S_fig3_populations_60s.pdf: the sweep on Fig 3's four
    populations, in Fig 3's 2x2 quadrant layout."""
    sw = quadrant_sweeps(REC_CAP_S, args.workers, args.recompute)
    quads = [(t, m) for t, m, _ in sw]
    resps = [r for _, _, r in sw]

    plt.rc("font", family=FP.FONT_FAMILY, size=FP.FS_BODY)
    fig = plt.figure(figsize=FP.FIGSIZE_FIG3_MAIN)
    # Fig 3's outer quadrant grid (fig3.plot_fig3), one heatmap per quadrant.
    outer = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.55,
                             left=0.1, right=0.98, top=0.88, bottom=0.07)
    for i, ((title, members), resp) in enumerate(zip(quads, resps)):
        orow, ocol = divmod(i, 2)
        ax = fig.add_subplot(outer[orow, ocol])
        _heatmap(fig, ax, resp, f"N = {len(members)}, first {REC_CAP_S:g} s",
                 cbar_label=f"Responders (q < {fig4.QVALUE_FDR:g})" if ocol == 1 else None,
                 log=args.log)
        ax.title.set_fontsize(FP.FS_BODY)
        if ocol == 0:
            ax.set_ylabel("Fraction of population modulated")
        # Quadrant title + letter as fig3.plot_fig3 places them.
        quad_pos = outer[orow, ocol].get_position(fig)
        title_y = quad_pos.y1 + 0.35 / fig.get_figheight()
        fig.text((quad_pos.x0 + quad_pos.x1) / 2, title_y, title,
                 ha="center", va="bottom", fontsize=FP.FS_BODY + 1)
        fig.text(quad_pos.x0 - 0.07, title_y, "ABCD"[i],
                 ha="left", va="bottom", fontfamily="arial", fontsize=12)
    out = Path(FP.SUPPLEMENT_DIR) / (f"Fig4S_fig3_populations_{REC_CAP_S:g}s"
                                     f"{'_log' if args.log else ''}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    FP.save_main_figure(fig, out)
    print(f"Saved {out}")
    for (title, members), resp in zip(quads, resps):
        print(f"{title} (N={len(members)}): fraction modulated for {fig4.DETECTION_RESPONDERS} "
              f"responders at " + ", ".join(f"A={a:g}: {fig4.detection_participation(resp, a):.2f}"
                                            for a in fig4.QUOTED_AMPLITUDES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--quadrants", action="store_true",
                    help="write Fig4S_fig3_populations_60s.pdf (Fig 3's four populations) instead")
    ap.add_argument("--log", action="store_true",
                    help="with --quadrants: logarithmic colour axes (writes ..._60s_log.pdf)")
    args = ap.parse_args()

    if args.quadrants:
        return main_quadrants(args)

    pop = fig3_populations()
    if CACHE.exists() and not args.recompute:
        print(f"Loading cached sweeps from {CACHE}")
        resp_all, resp_pigeon, kept = pd.read_pickle(CACHE)
    else:
        groups = cap_groups(load_groups(pop))
        kept = np.sort(np.concatenate([g["rows"] for g in groups]))
        amps, P = pvalue_table(groups, len(pop), args.workers)
        pigeon_kept = kept[pop.species.to_numpy()[kept] == "Pigeon"]
        resp_all = responder_sweep(P, amps, kept)
        resp_pigeon = responder_sweep(P, amps, pigeon_kept)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle((resp_all, resp_pigeon, kept), CACHE)
        print(f"Cached -> {CACHE}")
    kept_pop = pop.loc[kept]
    print(f"Fig 3 spiking neurons kept after the {REC_CAP_S:g} s cap:")
    print(pd.DataFrame({"Fig 3": pop.groupby("species").size(),
                        "kept": kept_pop.groupby("species").size()}).fillna(0).astype(int).to_string())
    pigeon = np.flatnonzero(kept_pop.species.to_numpy() == "Pigeon")
    pop = kept_pop

    plt.rc("font", family=FP.FONT_FAMILY, size=FP.FS_BODY)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FP.MAIN_FIG_WIDTH, 3.3), tight_layout=True)
    _heatmap(fig, ax_a, resp_all, f"Fig 3A: all spiking neurons\n(N={len(pop)}, first {REC_CAP_S:g} s)")
    _heatmap(fig, ax_b, resp_pigeon, f"Fig 3B: all pigeon neurons\n(N={len(pigeon)}, first {REC_CAP_S:g} s)",
             cbar_label=f"Number of responders (q < {fig4.QVALUE_FDR:g})")
    ax_a.set_ylabel("Fraction of population modulated")
    for ax, letter in ((ax_a, "A"), (ax_b, "B")):
        ax.annotate(letter, xy=(-0.2, 1.08), xycoords="axes fraction",
                    fontfamily="arial", fontsize=12)
    out = Path(FP.SUPPLEMENT_DIR) / "Fig4S_fig3_populations.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    FP.save_main_figure(fig, out)
    print(f"Saved {out}")
    for name, resp, n in (("all spiking", resp_all, len(pop)), ("pigeon", resp_pigeon, len(pigeon))):
        print(f"{name} (N={n}): fraction modulated for {fig4.DETECTION_RESPONDERS} responders at "
              + ", ".join(f"A={a:g}: {fig4.detection_participation(resp, a):.2f}"
                          for a in fig4.QUOTED_AMPLITUDES))


if __name__ == "__main__":
    main()
