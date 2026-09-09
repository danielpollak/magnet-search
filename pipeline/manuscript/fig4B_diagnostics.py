"""Fig 4B diagnostics -- the same NFC scatter, re-plotted against four
different per-unit x-axes, to investigate the visible banding in Fig4B (two
apparent populations of units differing in how strongly the same synthetic
modulation shows up in their NFC).

Only the A = 0.6 modulation condition is plotted -- it shows the banding
most distinctly, and restricting to it means only that one slice of fig4's
FR sweep has to be (re)computed.

Panels, in order:
  A  total recording time (per unit: first-to-last-spike span, seconds)
  B  total spike count
  C  Fano factor (count variance / count mean over fixed-width bins,
     --fano-bin seconds, default 1/FREQ = one stimulus period)
  D  firing rate (Hz) -- the original Fig4B x-axis, for reference

Two single-page companions are written alongside it: OUT_NAME_SPLIT
(NFC vs spike count alone, split into three stacked axes by recording
duration, independent y scales -- all data kept) and OUT_NAME_EQUAL (the
by-recording-time page recomputed with every unit truncated to one common
window -- data thrown away, bands gone).

HISTORICAL NOTE (2026-09-07): the banding this file was written to explain
was an artifact, and not the one it concluded. fig4 pooled each site's 6-10
magnetic recordings by cluster id while nwb_io.build_modulation_frame hands
back RECORDING-LOCAL spike times, so the recs were OVERLAID on one ~30-90 s
axis rather than concatenated. The "four different recording durations"
(62/64/96/236 s) this file's panels chase were therefore not durations at
all, just the union of each site's recs' local epoch windows. fig4 now
concatenates the recs onto a real non-overlapping timeline
(fig4.concat_mag_recs) and truncates every unit to a common window anchored
at t=0 of it (fig4.truncate_pseudopop_to_window), which removes the banding
at its source. The panels below still run, and are kept as the record of the
investigation, but read their "recording time" axis as "extent of the
overlaid recs" for any output generated before that date.

The main PDF has three pages, the same four panels each time: page 1 in a flat
hue, page 2 colored by total recording time, page 3 colored by Fano factor
-- for checking whether the structure left on a given x axis is still the
four sessions' discrete recording durations. On pages 2/3 the one panel
whose own x IS that page's color variable stays flat-hued (panel A on page
2, panel C on page 3), since coloring it would only restate its x axis.

Everything else about each panel matches fig4.py's panel B: same pooled
pigeon-HP pseudopopulation, same Dark2 hue that panel gives A=0.6, same
eps-corrected NFC 99th-percentile line, same black outlines on the
top-decile-sensitivity units. Only the x variable changes across panels, so
any band that persists or collapses is attributable to that x variable and
nothing else.

The per-unit NFC values come from fig4.py's own cached FR dataframe
(FR_DF_CACHE) -- this script never re-runs the simulation unless
--recompute is passed, and even then it computes only the A=0.6 slice and
does NOT overwrite fig4's cache (which holds all three conditions).

Unit identity: `id` in that cache is the index into
fig4.load_pseudopopulation_spks's spike-count-sorted unit list, which this
script reconstructs (keeping the "experiment:cluster_id" keys fig4
discards) and cross-checks against the cached FR column before joining, so
a stale cache can't silently mis-align the new x variables against the
NFCs.

Usage:
    python pipeline/manuscript/fig4B_diagnostics.py
    python pipeline/manuscript/fig4B_diagnostics.py --fano-bin 1.0
    python pipeline/manuscript/fig4B_diagnostics.py --recompute --workers 8
"""
import argparse
from pathlib import Path

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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

from magpyneto2 import statistics

import fig4
import format_parameters as FP

OUT_NAME = "fig4B_diagnostics.pdf"
# Two single-page companions to OUT_NAME's page 2 (the by-recording-time
# coloring), each showing a different way of dealing with the duration
# bands that page exposes.
OUT_NAME_EQUAL = "fig4B_diagnostics_time_equal.pdf"   # truncate every unit to one window
OUT_NAME_SPLIT = "fig4B_diagnostics_split.pdf"        # keep all data, separate panel B by duration

# Per-unit active-span bin edges for OUT_NAME_SPLIT's three stacked panel-B
# axes (shortest on top, longest on the bottom).
#
# Re-chosen 2026-09-07 for the concatenated timeline. The old (72, 150)
# edges were placed at empty stretches of a per-unit T histogram that no
# longer exists -- those "clusters" at ~60-65/~92-100/~130-140/~236 s were
# the union of each site's OVERLAID rec windows, not durations (see the
# module docstring's historical note). On the real timeline every unit is
# observed for the full fig4.EQUAL_WINDOW_S, so T here is now just how much
# of that window a unit was ACTIVE across, and it is heavily
# left-skewed: 10th pct 261 s, 25th 323 s, median 344 s, and 37% of units
# span >99% of the window. (72, 150) therefore binned 15/73/2350 -- one
# axis holding 96% of the population. These edges sit near the 10th and
# 50th percentiles instead, so the three axes are populated enough to
# compare.
SPLIT_EDGES = (260.0, 340.0)

# The one modulation condition plotted here: fig4 panel B's strongest.
# Derived from fig4.AMPLITUDES_FIG4B rather than restated, so lowering
# panel B's amplitudes (as [0, 0.3, 0.6] -> [0, 0.1, 0.2] did) carries here
# automatically -- otherwise this script silently plots a condition
# fig4.FR_DF_CACHE no longer contains, which its own mod-filter below would
# only catch after a full reload. Taking the max, not palette index 2,
# keeps "the strongest condition" true if the list ever changes length;
# MOD_ALL.index below still resolves the matching seaborn Dark2 hue, so a
# panel here is visually the same population, same color, just a different x.
MOD_ALL = list(fig4.AMPLITUDES_FIG4B)
MOD_LEVEL = max(MOD_ALL)

# (column, axis label, log-x?) -- panel order is the order requested:
# time, spike count, Fano factor, then the original firing-rate version.
PANELS = [
    ("T",     "Total recording time (s)", True),
    ("n_spk", "Total spike count",        True),
    ("fano",  "Fano factor",              True),
    ("FR",    "Firing rate (Hz)",         True),
]


def mod_color():
    return sns.color_palette(fig4.PALETTE_FIG4B)[MOD_ALL.index(MOD_LEVEL)]


def load_keyed_pseudopopulation(data_dir: str, experiments):
    """fig4.load_pseudopopulation_spks, but keeping each unit's
    "{experiment}:{cluster_id}" key. Same pooling order, same
    EQUAL_WINDOW_S common-window truncation and same spike-count-ascending
    sort, so the returned list index equals the `id` column
    fig4.compute_fr_df writes (np.arange(len(spks)) over exactly this list).
    """
    all_units = {}
    q_fracs = {}
    for experiment in experiments:
        # Third return value (that experiment's total concatenated mag
        # duration) is checked against EQUAL_WINDOW_S by
        # fig4.load_pseudopopulation_spks; this script inherits the same
        # window, so re-checking here would only duplicate that guard.
        unit_spks, Q_frac, _total_T = fig4.load_unit_spks_for_experiment(
            data_dir, experiment)
        all_units.update(unit_spks)
        q_fracs[experiment] = Q_frac
    # Same common-window truncation fig4.load_pseudopopulation_spks applies
    # -- not optional here: this script joins fig4's own FR cache onto these
    # units by position, so pooling a different unit set than fig4 did would
    # trip the cache-staleness guard in main() (or, worse, silently misalign).
    all_units = fig4.truncate_pseudopop_to_window(all_units)
    unique_q_fracs = set(q_fracs.values())
    if len(unique_q_fracs) > 1:
        raise ValueError(f"mag_Q_frac/Q_frac differs across pooled experiments: {q_fracs}")
    keys_sorted = sorted(all_units, key=lambda k: len(all_units[k]))
    spks = [all_units[k] for k in keys_sorted]
    print(f"  {len(spks)} pigeon-HP pseudopopulation units pooled from {len(experiments)} experiments")
    return keys_sorted, spks, unique_q_fracs.pop()


def compute_fr_df_one_mod(spks, FOURIER_Q, workers=1, mod=MOD_LEVEL,
                          window_s=fig4.EQUAL_WINDOW_S):
    """fig4.compute_fr_df restricted to a single modulation amplitude --
    same worker (fig4._fr_cell), same FR definition, one grid cell instead
    of three. Deliberately not written to fig4.FR_DF_CACHE: that cache is
    the three-condition frame fig4.py itself reloads.
    """
    results = fig4._run_parallel([(mod,)], fig4._fr_cell, spks, fig4.FREQ, FOURIER_Q,
                                 workers, desc=f"FR vs NFC (A={mod})")
    A, NFCs = results[0]
    return pd.DataFrame({
        "mod": A,
        "FR": fig4.unit_firing_rates(spks, window_s),
        "NFC": NFCs,
        "id": np.arange(len(spks)),
    })


def fano_factor(spkt, bin_s):
    """Count variance / count mean over consecutive fixed-width bins
    spanning the unit's own first-to-last-spike interval. The trailing
    partial bin is dropped (a short final bin holds systematically fewer
    spikes, which would inflate the variance for no physiological reason).
    Returns NaN if the span holds fewer than two whole bins, or if the mean
    count is 0.
    """
    span = spkt.max() - spkt.min()
    n_bins = int(span // bin_s)
    if n_bins < 2:
        return np.nan
    edges = spkt.min() + np.arange(n_bins + 1) * bin_s
    counts, _ = np.histogram(spkt, bins=edges)
    if counts.mean() == 0:
        return np.nan
    return counts.var(ddof=1) / counts.mean()


def build_unit_metrics(keys, spks, bin_s, window_s=fig4.EQUAL_WINDOW_S):
    """One row per unit: the per-unit x variables the panels plot against.
    `id` matches fig4.compute_fr_df's own `id`.

    `T` stays each unit's actual first-to-last-spike span -- panel A plots
    it, and post-equalization it is exactly the diagnostic of how much of
    its common window a unit actually fills. `FR` must instead use
    `window_s`, matching fig4.unit_firing_rates, or main()'s cache
    cross-check against fig4's own FR column would fail.
    """
    rows = []
    frs = fig4.unit_firing_rates(spks, window_s)
    for i, (key, spkt, fr) in enumerate(zip(keys, spks, frs)):
        rows.append({
            "id": i,
            "unit": key,
            "experiment": key.split(":")[0],
            "T": spkt.max() - spkt.min(),
            "n_spk": len(spkt),
            "FR": fr,
            "fano": fano_factor(spkt, bin_s),
        })
    return pd.DataFrame(rows)


def print_numeric_diagnostics(df, metrics):
    """Rank correlation of NFC against each candidate x variable, plus a
    per-experiment summary -- the banding is only interpretable if you can
    see WHICH variable it tracks, and whether the two apparent populations
    are just two source experiments.
    """
    print(f"\nSpearman rank correlation of NFC vs each x variable (A={MOD_LEVEL}):")
    for m in metrics:
        sub = df[[m, "NFC"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) > 2:
            res = sp_stats.spearmanr(sub[m], sub["NFC"])
            print(f"  {m:<8} rho={res.statistic:+.3f}  p={res.pvalue:.2e}  n={len(sub)}")
        else:
            print(f"  {m:<8} (too few finite values: n={len(sub)})")

    print("\nPer-experiment unit counts and medians:")
    summary = df.groupby("experiment").agg(
        n_units=("id", "size"),
        median_T=("T", "median"),
        median_n_spk=("n_spk", "median"),
        median_FR=("FR", "median"),
        median_fano=("fano", "median"),
        median_NFC=("NFC", "median"),
    )
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))


def color_norm(values, log_color):
    """Shared color scale for one page's coloring variable, computed once
    over the whole population so every panel (and, in the split figure,
    every stacked sub-axis) maps a value to the same color.
    """
    finite = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if log_color:
        return LogNorm(vmin=finite[finite > 0].min(), vmax=finite.max())
    return Normalize(vmin=finite.min(), vmax=finite.max())


def _cmap():
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("lightgrey")  # units with no defined Fano factor stay visible
    return cmap


def _draw_panel(fig, ax, df, top_df, col, xlabel, logx, nfc_99, color_by,
                color_label, norm, add_colorbar=True):
    """One NFC-vs-x panel.

    `color_by=None` draws every point in the flat A=0.6 hue -- page 1, and
    also the one panel per later page whose own x variable IS that page's
    coloring variable (a color ramp there would just restate the x axis).
    Otherwise points carry a continuous colormap over `color_by` on the
    caller-supplied shared `norm`; the top-decile-sensitivity overlay uses
    the same scale, so an outlined point never changes fill between pages.
    `add_colorbar=False` suppresses this panel's own colorbar (the split
    figure's three stacked axes share one).
    """
    if color_by is None:
        color = mod_color()
        sns.scatterplot(data=df, x=col, y="NFC", color=color, s=5, linewidth=0,
                        alpha=FP.ALPHA_SCATTER, ax=ax, legend=False)
        sns.scatterplot(data=top_df, x=col, y="NFC", color=color, s=5, linewidth=0.6,
                        edgecolor="black", alpha=FP.ALPHA_SCATTER, ax=ax, legend=False)
    else:
        cmap = _cmap()
        ax.scatter(df[col], df["NFC"], c=df[color_by], cmap=cmap, norm=norm,
                   s=5, linewidth=0, alpha=FP.ALPHA_SCATTER)
        ax.scatter(top_df[col], top_df["NFC"], c=top_df[color_by], cmap=cmap, norm=norm,
                   s=5, linewidth=0.6, edgecolor="black", alpha=FP.ALPHA_SCATTER)
        if add_colorbar:
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=color_label)
    if logx:
        ax.set_xscale("log")
    # axhline, not hlines(*get_xlim()): on the split figure's shared-x axes
    # the limits aren't final at draw time, which would leave a short stub.
    ax.axhline(nfc_99, color="grey")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("NFC")


def plot_page(df, top_df, nfc_99, bin_s, color_by, color_label, log_color, subtitle):
    """One 2x2 page of the diagnostics PDF: the same four panels (time,
    spike count, Fano factor, firing rate), optionally re-colored by
    `color_by`.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    titles = {"C": f"Fano factor, {bin_s:.3g} s bins",
              "D": "original Fig4B x-axis"}

    norm = color_norm(df[color_by], log_color) if color_by is not None else None

    for ax, (col, xlabel, logx), letter in zip(axes.ravel(), PANELS, "ABCD"):
        # Coloring a panel by its own x variable carries no information, so
        # that panel falls back to the flat hue on its page.
        panel_color_by = None if col == color_by else color_by
        _draw_panel(fig, ax, df, top_df, col, xlabel, logx, nfc_99,
                    panel_color_by, color_label, norm)
        title = titles.get(letter, "")
        if color_by is not None and panel_color_by is None:
            title = (title + " -- " if title else "") + "x is this page's color variable"
        ax.set_title(title, fontsize=FP.FS_TITLE)
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontfamily="arial",
                fontsize=FP.FS_PANEL, ha="left", va="bottom")

    fig.suptitle(f"Fig4B banding diagnostics, A={MOD_LEVEL} modulation only -- {subtitle}\n"
                 f"({len(top_df)} top-decile-sensitivity units outlined in black; "
                 f"grey line = 99th percentile of the null NFC)",
                 fontsize=FP.FS_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def duration_groups(df, edges=SPLIT_EDGES):
    """Split units into duration bins, shortest first. Returns a list of
    (label, sub-dataframe); empty bins are dropped.
    """
    bounds = [-np.inf, *edges, np.inf]
    groups = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sub = df[(df["T"] >= lo) & (df["T"] < hi)]
        if sub.empty:
            continue
        if lo == -np.inf:
            label = f"T < {hi:.0f} s"
        elif hi == np.inf:
            label = f"T > {lo:.0f} s"
        else:
            label = f"{lo:.0f} s < T < {hi:.0f} s"
        groups.append((f"{label}  (n={len(sub)}, median {sub['T'].median():.0f} s)", sub))
    return groups


def plot_split_page(df, top_df, nfc_99):
    """NFC vs spike count only, split into three stacked axes by recording
    duration -- shortest on top, longest on the bottom -- so the
    duration-dependent offset at matched spike count can be read off
    directly instead of being inferred from overlapping colors.

    The time/Fano/firing-rate panels are deliberately absent: this figure
    exists to give the three duration axes the full page height. x
    (spike count) is shared so the three are horizontally comparable, but y
    is NOT -- each axis autoscales to its own NFC range, which is the
    quantity being compared.
    """
    norm = color_norm(df["T"], log_color=False)
    color_label = "Total recording time (s)"
    groups = duration_groups(df)  # shortest first -> top

    fig = plt.figure(figsize=(7.5, 9))
    gs = gridspec.GridSpec(len(groups), 1, figure=fig, hspace=0.18)
    col, xlabel, logx = PANELS[1]

    axes = []
    for i, (label, sub) in enumerate(groups):
        # sharex only -- see docstring; each axis keeps its own y range.
        share = dict(sharex=axes[0]) if axes else {}
        ax = fig.add_subplot(gs[i], **share)
        sub_top = top_df[top_df["id"].isin(sub["id"])]
        _draw_panel(fig, ax, sub, sub_top, col, xlabel, logx, nfc_99,
                    "T", color_label, norm, add_colorbar=False)
        ax.set_title(label, fontsize=FP.FS_TITLE)
        # Only the bottom axis keeps the x label/ticks -- the three are one
        # panel, not three panels. Every axis keeps its own y label, since
        # each now carries its own y scale.
        if i < len(groups) - 1:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        axes.append(ax)

    # x limits from the full population so the three axes stay comparable
    # to the un-split panel B on the other pages.
    axes[0].set_xlim(df[col].min() * 0.7, df[col].max() * 1.4)
    fig.colorbar(ScalarMappable(norm=norm, cmap=_cmap()), ax=axes, label=color_label)

    fig.suptitle(f"NFC vs spike count by recording duration, A={MOD_LEVEL} modulation "
                 f"(shortest top, longest bottom; independent y scales)\n"
                 f"({len(top_df)} top-decile-sensitivity units outlined in black; "
                 f"grey line = 99th percentile of the null NFC)",
                 fontsize=FP.FS_TITLE)
    return fig


def truncate_to_common_window(keys, spks, T_common, min_spikes=fig4.MIN_SPIKES):
    """Keep only each unit's first `T_common` seconds, measured from its own
    first spike, so every unit is analyzed over an equal-length window.
    Units left with fewer than `min_spikes` spikes are dropped (same floor
    fig4.load_unit_spks_for_experiment applies at load time).
    """
    kept_keys, kept_spks, dropped = [], [], 0
    for key, spkt in zip(keys, spks):
        trunc = spkt[spkt - spkt.min() < T_common]
        if len(trunc) < min_spikes:
            dropped += 1
            continue
        kept_keys.append(key)
        kept_spks.append(trunc)
    return kept_keys, kept_spks, dropped


def build_time_equalized_df(keys, spks, Q_frac, bin_s, workers, window=None,
                            recompute=False, cache_dir: Path = None):
    """Truncate the pseudopopulation to one common window, recompute the
    A=0.6 NFCs on the truncated trains, and return
    (df, top_decile_mask, FOURIER_Q, T_common).

    The NFCs CANNOT come from fig4's cache -- they are a different
    (shorter) observation of every unit -- so this runs one fresh Fourier
    cell, cached under its own window-stamped filename.
    """
    spans = np.array([s.max() - s.min() for s in spks])
    # Default is half fig4's own common window, NOT spans.min() as it was
    # before the concatenation. On the overlaid timeline spans.min() was
    # ~39 s and stood for "the longest window every unit can supply"; on the
    # real timeline a unit's span is only how long it was ACTIVE, so
    # spans.min() is now 0.0 s (a unit whose handful of spikes land close
    # together) and truncating there would empty the population. Half the
    # window keeps this page asking a well-posed question -- does halving
    # every unit's observation change the picture -- rather than a broken
    # one.
    T_common = float(window) if window else float(fig4.EQUAL_WINDOW_S) / 2
    keys_t, spks_t, dropped = truncate_to_common_window(keys, spks, T_common)
    source = "--equal-window" if window else "half fig4.EQUAL_WINDOW_S"
    if spans.min() == 0.0:
        print(f"  (shortest unit span in the pool is {spans.min():.1f} s -- "
              f"not usable as a window, see build_time_equalized_df)")
    print(f"\nTime-equalized population: window = {T_common:.1f} s ({source})")
    print(f"  {len(spks_t)} units kept, {dropped} dropped for having "
          f"<{fig4.MIN_SPIKES} spikes "
          f"left after truncation")

    FOURIER_Q = fig4.fourier_Q_from_frac(spks_t, fig4.FREQ, Q_frac,
                                         context=f"[fig4B time-equalized @ {fig4.FREQ}Hz] ")
    print(f"  FOURIER_Q={FOURIER_Q} bins at {fig4.FREQ} Hz (recomputed for the shorter window)")

    cache = Path(cache_dir) / f"fig4B_time_equalized_A{MOD_LEVEL}_{T_common:.0f}s.pkl"
    if not recompute and cache.exists():
        print(f"  Loading cached time-equalized NFCs from {cache}")
        nfc_df = pd.read_pickle(cache)
        if list(nfc_df["unit"]) != keys_t:
            raise ValueError(
                f"Cached time-equalized NFCs at {cache} cover a different unit set -- "
                f"re-run with --recompute."
            )
    else:
        print(f"  Computing time-equalized NFCs at A={MOD_LEVEL}...")
        fr = compute_fr_df_one_mod(spks_t, FOURIER_Q, workers=workers,
                                   window_s=T_common)
        nfc_df = pd.DataFrame({"unit": keys_t, "NFC": fr["NFC"].values})
        cache.parent.mkdir(parents=True, exist_ok=True)
        nfc_df.to_pickle(cache)
        print(f"  Cached -> {cache}")

    # T_common, not EQUAL_WINDOW_S: this companion re-truncates to its own
    # (shorter) window, so its FR denominator has to follow it.
    metrics = build_unit_metrics(keys_t, spks_t, bin_s, window_s=T_common)
    df = metrics.merge(nfc_df, on="unit")
    df["mod"] = MOD_LEVEL

    sens = fig4.compute_sensitivity(spks_t, FOURIER_Q, freq=fig4.FREQ)
    top_decile_mask = sens >= np.quantile(sens, fig4.SENSITIVITY_PERCENTILE / 100)
    return df, top_decile_mask, FOURIER_Q, T_common


def save_single_page(fig, out_path: Path):
    fig.savefig(out_path, dpi=FP.DPI_FIG4, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    if in_notebook:
        plt.show()
    else:
        plt.close(fig)


def plot_fig4B_diagnostics(df, top_decile_mask, FOURIER_Q, bin_s, out_dir: Path):
    """Three-page PDF, same four panels each time:
      1. flat A=0.6 hue (the original diagnostic)
      2. colored by total recording time -- tests whether the bands that
         survive on the spike-count/Fano/FR axes are still just the four
         sessions' discrete recording durations
      3. colored by Fano factor
    """
    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY_XL}
    matplotlib.rc("font", **font)

    top_ids = set(np.where(top_decile_mask)[0].tolist())
    top_df = df[df["id"].isin(top_ids)]
    # eps-corrected exactly as in fig4.plot_fig4 -- the NFCs plotted here
    # are the ones fig4._fr_cell produced at this same FOURIER_Q.
    nfc_99 = statistics.inverse_Rayleigh_CDF(0.99, eps=statistics.get_epsilon(FOURIER_Q))

    pages = [
        # (color_by, color_label, log_color, subtitle)
        (None, None, False, "single hue"),
        ("T", "Total recording time (s)", False, "colored by total recording time"),
        ("fano", f"Fano factor ({bin_s:.3g} s bins)", True, "colored by Fano factor"),
    ]

    out_path = out_dir / OUT_NAME
    with PdfPages(out_path) as pdf:
        for color_by, color_label, log_color, subtitle in pages:
            fig = plot_page(df, top_df, nfc_99, bin_s, color_by, color_label,
                            log_color, subtitle)
            pdf.savefig(fig, dpi=FP.DPI_FIG4, bbox_inches="tight")
            if in_notebook:
                plt.show()
            else:
                plt.close(fig)
    print(f"\nSaved {len(pages)} pages -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fig4B banding diagnostics (four x-axis variants)")
    parser.add_argument("--out-dir", default=FP.OUT_DIR, help="Output directory for the PDF")
    parser.add_argument("--data-dir", default=FP.DATA_DIR, help="Directory containing pipeline output .nwb files")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Explicit experiment names to pool (default: fig4's own auto-discovery)")
    parser.add_argument("--fano-bin", type=float, default=1 / fig4.FREQ,
                        help="Fano-factor bin width in seconds (default: 1/FREQ, one stimulus period)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers, used only when --recompute re-runs the A=0.6 FR cell")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute the A=0.6 NFCs instead of reading fig4's FR cache")
    parser.add_argument("--equal-window", type=float, default=None,
                        help=f"Common window in seconds for {OUT_NAME_EQUAL} "
                             f"(default: half fig4.EQUAL_WINDOW_S -- see "
                             f"build_time_equalized_df)")
    args = parser.parse_args([] if in_notebook else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = args.experiments or fig4.discover_pigeon_hp_experiments(fig4._EXPERIMENTS_DIR)
    print(f"Pooling pigeon-HP pseudopopulation from: {experiments}")
    keys, spks, Q_frac = load_keyed_pseudopopulation(args.data_dir, experiments)

    FOURIER_Q = fig4.fourier_Q_from_frac(spks, fig4.FREQ, Q_frac,
                                         context=f"[fig4B diagnostics @ {fig4.FREQ}Hz] ")
    print(f"  Q_frac={Q_frac} -> FOURIER_Q={FOURIER_Q} bins at {fig4.FREQ} Hz")

    if not args.recompute and Path(fig4.FR_DF_CACHE).exists():
        print(f"Loading cached FR df from {fig4.FR_DF_CACHE}")
        fr_df = pd.read_pickle(fig4.FR_DF_CACHE)
        fr_df = fr_df[np.isclose(fr_df["mod"], MOD_LEVEL)]
        if fr_df.empty:
            raise ValueError(
                f"Cached FR df has no mod=={MOD_LEVEL} rows -- re-run with --recompute."
            )
    else:
        print(f"Computing NFCs at A={MOD_LEVEL}...")
        fr_df = compute_fr_df_one_mod(spks, FOURIER_Q, workers=args.workers)

    metrics_df = build_unit_metrics(keys, spks, args.fano_bin)

    # Guard against a stale cache built from a different population/order:
    # the cache's own FR column must reproduce from the units loaded here,
    # unit-for-unit, or the join below would attach the wrong x values to
    # each NFC.
    if len(fr_df) != len(metrics_df):
        raise ValueError(
            f"FR df has {len(fr_df)} units but {len(metrics_df)} were loaded -- "
            f"stale cache; re-run with --recompute."
        )
    check = fr_df.merge(metrics_df[["id", "FR"]], on="id", suffixes=("_cached", "_here"))
    if not np.allclose(check["FR_cached"], check["FR_here"], rtol=1e-9, atol=0):
        raise ValueError(
            "Cached FR values don't match the units loaded here -- the cache is "
            "stale or its unit ordering differs; re-run with --recompute."
        )

    df = fr_df.drop(columns=["FR"]).merge(metrics_df, on="id")

    n_nan_fano = int(df["fano"].isna().sum())
    if n_nan_fano:
        print(f"  {n_nan_fano} units have <2 whole {args.fano_bin:.3g}s bins -- Fano factor NaN, "
              f"dropped from panel C only")

    # Top-decile-sensitivity units, same definition/threshold as fig4's
    # panels C/D (and its own panel-B outlines).
    sens = fig4.compute_sensitivity(spks, FOURIER_Q, freq=fig4.FREQ)
    sens_threshold = np.quantile(sens, fig4.SENSITIVITY_PERCENTILE / 100)
    top_decile_mask = sens >= sens_threshold
    print(f"  {int(np.sum(top_decile_mask))} / {len(spks)} units at/above the "
          f"{fig4.SENSITIVITY_PERCENTILE:.0f}th percentile sensitivity "
          f"(threshold={sens_threshold:.3f})")

    print_numeric_diagnostics(df, [c for c, _, _ in PANELS])
    plot_fig4B_diagnostics(df, top_decile_mask, FOURIER_Q, args.fano_bin, out_dir)

    # Companion 1 -- keeps every spike, separates panel B by duration.
    top_df = df[df["id"].isin(set(np.where(top_decile_mask)[0].tolist()))]
    nfc_99 = statistics.inverse_Rayleigh_CDF(0.99, eps=statistics.get_epsilon(FOURIER_Q))
    save_single_page(plot_split_page(df, top_df, nfc_99),
                     out_dir / OUT_NAME_SPLIT)

    # Companion 2 -- the version that DOES throw data away, for reference.
    eq_df, eq_mask, eq_Q, T_common = build_time_equalized_df(
        keys, spks, Q_frac, args.fano_bin, args.workers, window=args.equal_window,
        recompute=args.recompute, cache_dir=Path(fig4.FR_DF_CACHE).parent)
    print(f"\n[time-equalized, {T_common:.1f} s window]")
    print_numeric_diagnostics(eq_df, [c for c, _, _ in PANELS])
    eq_top_df = eq_df[eq_df["id"].isin(set(np.where(eq_mask)[0].tolist()))]
    eq_nfc_99 = statistics.inverse_Rayleigh_CDF(0.99, eps=statistics.get_epsilon(eq_Q))
    fig_eq = plot_page(eq_df, eq_top_df, eq_nfc_99, args.fano_bin,
                       color_by="T", color_label="Total recording time (s)",
                       log_color=False,
                       subtitle=f"every unit truncated to a common {T_common:.1f} s window")
    save_single_page(fig_eq, out_dir / OUT_NAME_EQUAL)


if __name__ == "__main__":
    if in_notebook:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    main()
