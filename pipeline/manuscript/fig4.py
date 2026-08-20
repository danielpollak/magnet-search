"""Fig 4 — modulation strength vs excess suspects and firing rate.

Loads spike trains from the pipeline's own {experiment}.nwb, then
synthetically modulates them to show detection thresholds.

Requires:
  data/{experiment}.nwb  (default: 20220228_firstsite)
  seaborn

Cached simulation results are saved to data/manuscript/ on first run and
reused on subsequent runs (pass --recompute to force a fresh simulation).

Usage:
    python pipeline/manuscript/fig4.py
    python pipeline/manuscript/fig4.py --experiment 20230413_firstsite --rec "2023-04-13_15-08-42_W25R_Mag2"
    python pipeline/manuscript/fig4.py --recompute
"""
import argparse
import glob
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
CI_DF_CACHE = _REPO_ROOT / "data" / "manuscript" / "modulation_strength_vs_excess_count.pkl"
FR_DF_CACHE = _REPO_ROOT / "data" / "manuscript" / "modulation_strength_vs_FR.pkl"

DEFAULT_EXPERIMENT = "20220228_firstsite"
DEFAULT_REC        = "10 Hz_2022-02-28_16-21-40"
FREQ               = 5
A_L                = np.linspace(0, 0.3, num=11)
PERCENTAGES        = np.linspace(0, 1, num=5)


def load_spks(data_dir: str, experiment: str, rec_name: str):
    # Reads the pipeline's own {experiment}.nwb (Phase 7 cutover -- no
    # paradigm writes the legacy {experiment}_processing.pickle anymore),
    # same reproducible-from-NWB principle applied to fig1.py.
    experiments_dir = Path(__file__).parent.parent.parent / "experiments"
    cfg = load_experiment(experiments_dir / f"{experiment}.yml")
    nwb_path = Path(data_dir) / f"{experiment}.nwb"
    io_r, nwbfile = nwb_io.read_nwbfile(str(nwb_path))
    modulation_df = nwb_io.build_modulation_frame(nwbfile, good_only=cfg.good)
    io_r.close()

    rec_df = modulation_df.loc[modulation_df.rec == rec_name]
    if rec_df.empty:
        available = modulation_df.rec.unique().tolist()
        raise ValueError(
            f"rec {rec_name!r} not found in {nwb_path.name}. "
            f"Available: {available}"
        )

    spks = [g.spk.values for _, g in rec_df.groupby("id") if len(g) > 5]
    spks = [spks[i] for i in np.argsort([len(s) for s in spks]).astype(int)]
    print(f"  {len(spks)} units with >5 spikes from {rec_name!r}")

    if cfg.analysis.Q_frac <= 0:
        raise ValueError(
            f"{experiment}.yml has no analysis.Q_frac set (got "
            f"{cfg.analysis.Q_frac!r}) -- Fig4's simulation derives its "
            f"off-frequency bin count from this experiment's own Q_frac, "
            f"same as the real per-experiment analysis."
        )
    return spks, cfg.analysis.Q_frac


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


def compute_ci_df(spks, FOURIER_Q):
    # Same corrected null the per-unit p-values elsewhere use for a real Q_frac
    # analysis (see fit_fourier_sig) -- fixed for the whole loop since every
    # call below analyzes at the same Q.
    eps = statistics.get_epsilon(FOURIER_Q)

    data_d = {"mod": [], "ci": [], "mid": [], "%": []}
    for percent_i, percent in enumerate(PERCENTAGES):
        for A in tqdm.tqdm(A_L, desc=f"{int(percent*100)}% modulated"):
            modulated = [None] * len(spks)
            for spk_i, spkt in enumerate(spks):
                if spk_i % len(PERCENTAGES) < percent_i:
                    modulated[spk_i] = statistics.warp_mod(spkt, A, 1 / FREQ, 0)
                else:
                    modulated[spk_i] = spkt
            (C, T, spk_count, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, NFCs) = \
                statistics.fourier_analysis(modulated, FREQ, Q=FOURIER_Q)
            n_empirical, f_expected, l_bound, h_bound = \
                statistics.suspect_count_significance(NFCs, 0.99, conf_int_α=0.05, eps=eps)
            data_d["mod"].append(A)
            data_d["ci"].append((l_bound, h_bound))
            data_d["mid"].append(n_empirical)
            data_d["%"].append(percent)
    return pd.DataFrame(data_d)


def compute_fr_df(spks, FOURIER_Q):
    T = statistics.latesttime(spks) - statistics.earliesttime(spks)
    rows = []
    for A in tqdm.tqdm([0, 0.3, 0.6], desc="FR vs NFC"):
        modulated = [statistics.modulate(spkt, FREQ, A) for spkt in spks]
        (C, _, spk_count, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, NFCs) = \
            statistics.fourier_analysis(modulated, FREQ, Q=FOURIER_Q)
        rows.append(pd.DataFrame({
            "mod": A,
            "FR": [len(spkt) / T for spkt in spks],
            "NFC": NFCs,
            "id": np.arange(len(modulated)),
        }))
    return pd.concat(rows)


def plot_fig4(ci_df, NFC_modulation_FR_df, spks, FOURIER_Q, out_dir: Path):
    example_spk = spks[len(spks) // 2]

    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY_XL}
    matplotlib.rc("font", **font)

    fig = plt.figure(figsize=FP.FIGSIZE_FIG4)
    gs = gridspec.GridSpec(3, 5, left=0, bottom=0, right=1, top=1, wspace=0.4, hspace=0.4)

    ax_A1 = fig.add_subplot(gs[0, 0])
    ax_A2 = fig.add_subplot(gs[1, 0])
    ax_A3 = fig.add_subplot(gs[0, 1])
    ax_A4 = fig.add_subplot(gs[1, 1])
    ax_A5 = fig.add_subplot(gs[0, 2])
    ax_A6 = fig.add_subplot(gs[1, 2])
    ax_B  = fig.add_subplot(gs[:2, 3:5])
    ax_C  = fig.add_subplot(gs[2, :])

    spectra_axes = [ax_A1, ax_A3, ax_A5]
    psth_axes    = [ax_A2, ax_A4, ax_A6]

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

        psth_axes[mod_i].hist(np.squeeze(warped % (1 / FREQ)), bins=25)
        psth_axes[mod_i].set_xlabel("Time (s)")
        if mod_i == 0:
            psth_axes[mod_i].set_ylabel("Spike counts")
        else:
            psth_axes[mod_i].set_yticks([])

    [ax.set_ylim((-1, 7)) for ax in spectra_axes]
    [ax.set_ylim((0, 30))  for ax in psth_axes]

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

    sns.scatterplot(data=NFC_modulation_FR_df, x="FR", y="NFC", hue="mod",
                    palette="Set1", size=0.5, alpha=FP.ALPHA_SCATTER, ax=ax_C)
    handles, labels = ax_C.get_legend_handles_labels()
    mod_vals = [0, 0.3, 0.6]
    pairs = [(h, l) for h, l in zip(handles, labels)
             if any(abs(float(l) - v) < 1e-9 for v in mod_vals)]
    ax_C.legend([h for h, _ in pairs],
                [f"A={l}" for _, l in pairs],
                title="modulation (5Hz)", ncol=3)
    # eps-corrected to match NFC_modulation_FR_df, which is computed by
    # compute_fr_df() at this same Q_frac-derived FOURIER_Q -- same corrected
    # null used for ci_df's confidence bounds in panel B.
    ax_C.hlines(statistics.inverse_Rayleigh_CDF(0.99, eps=statistics.get_epsilon(FOURIER_Q)),
                *ax_C.get_xlim(), color="grey")
    ax_C.set_ylabel(r"$\hat{c}$")
    ax_C.set_xscale("log")
    ax_C.set_xlabel("Firing rate (Hz)")

    fig.canvas.draw()

    # A and B: align on the same horizontal line using figure coordinates.
    # ax_A1 is one row tall; ax_B spans two rows, so axes-fraction y offsets
    # produce different figure-level positions without this correction.
    _label_y = max(ax_A1.get_position().y1, ax_B.get_position().y1) + 0.03
    for _ax, _lbl, _xoff in [(ax_A1, "A", -0.3), (ax_B, "B", -0.1)]:
        _pos = _ax.get_position()
        fig.text(_pos.x0 + _xoff * _pos.width, _label_y, _lbl,
                 fontfamily="arial", fontsize=12)

    ax_C.annotate("C", xy=(-0.05, 1), xycoords="axes fraction", fontfamily="arial", fontsize=12)

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
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT,
                        help=f"Experiment name to load spike trains from (default: {DEFAULT_EXPERIMENT})")
    parser.add_argument("--rec", default=DEFAULT_REC,
                        help=f"Recording name (rec column value) to use as spike source (default: {DEFAULT_REC!r})")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute simulation even if cached pickles exist")
    args = parser.parse_args([] if in_notebook else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(CI_DF_CACHE).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading spikes from {args.experiment} / {args.rec!r} ...")
    spks, Q_frac = load_spks(args.data_dir, args.experiment, args.rec)

    FOURIER_Q = fourier_Q_from_frac(spks, FREQ, Q_frac, context=f"[fig4 sim @ {FREQ}Hz] ")
    print(f"  Q_frac={Q_frac} ({args.experiment}.yml) -> FOURIER_Q={FOURIER_Q} bins at {FREQ} Hz")

    if not args.recompute and Path(CI_DF_CACHE).exists():
        print(f"Loading cached ci_df from {CI_DF_CACHE}")
        ci_df = pd.read_pickle(CI_DF_CACHE)
    else:
        print("Computing modulation strength vs excess count (slow)...")
        ci_df = compute_ci_df(spks, FOURIER_Q)
        ci_df.to_pickle(CI_DF_CACHE)
        print(f"Cached -> {CI_DF_CACHE}")

    if not args.recompute and Path(FR_DF_CACHE).exists():
        print(f"Loading cached FR df from {FR_DF_CACHE}")
        NFC_modulation_FR_df = pd.read_pickle(FR_DF_CACHE)
    else:
        print("Computing FR vs NFC...")
        NFC_modulation_FR_df = compute_fr_df(spks, FOURIER_Q)
        NFC_modulation_FR_df.to_pickle(FR_DF_CACHE)
        print(f"Cached -> {FR_DF_CACHE}")

    plot_fig4(ci_df, NFC_modulation_FR_df, spks, FOURIER_Q, out_dir)


if __name__ == "__main__":
    if in_notebook:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    main()
