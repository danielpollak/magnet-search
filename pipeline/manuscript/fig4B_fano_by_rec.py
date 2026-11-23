"""Fig 4B diagnostic -- Fano-factor ECDFs, one curve per source recording,
plus one concatenated-timeline raster per recording site.

Asks a single question: do the four pigeon-HP sites Fig4 pools into one
pseudopopulation have comparable spike-count variability, or does one of them
sit apart? Pooling units across sites is only defensible if they do, and the
Fano factor (count variance / count mean over fixed-width bins) is the
per-unit summary of that variability at the simulation's own timescale.

Terminology, because it matters here: one "recording site" (one experiment
YAML, one bird, one penetration) contains 6-10 separate magnetic RECORDINGS.
Each site is one raster page and one ECDF curve.

Page 1 -- two panels, the same four ECDFs on each, differing only in how much
of each unit is included:

  A  Fig4's ACTUAL population -- every unit truncated to the first
     EQUAL_WINDOW_S (350 s) of its site's concatenated timeline by
     fig4.truncate_pseudopop_to_window.
  B  Every "mag"-contingency unit over its site's FULL concatenated timeline
     (352.5-784.8 s depending on the site), with no truncation.

So panel B's curves are not panel A's plus extra units -- they are the same
units observed for longer, and for unequally longer depending on the site,
which is exactly why A exists.

Pages 2 onward -- one raster per site, every mag unit on its own horizontal
line, on that site's concatenated timeline.

Why "concatenated" is load-bearing (2026-09-07): `spk` from
nwb_io.build_modulation_frame is RECORDING-LOCAL time (`all_sts -
beginning_time`, see the local_offset comment there), which is exactly right
for fit_fourier_sig -- it groups by ("rec", "freq"), so a group is always one
rec. Pooling a unit's recs, as Fig4 does, is the one thing that convention
cannot survive: every rec's magnetic epoch starts ~30-60 s into that rec, so
a naive groupby("id") OVERLAYS all 6-10 recs on a single ~30-90 s axis. An
earlier version of this diagnostic plotted exactly that, and its apparent
139-178 s "silent gap" in 20230414_firstsite was not a gap in anything: it
was the space between nine recs' local epoch windows (~30-170 s) and
`mag8_inclined_lightson`'s (209-270 s). fig4.concat_mag_recs now shifts each
rec onto a real non-overlapping timeline, and these pages plot that.

On a raster page, alternating background bands are the individual recs, in
`start_time` (catalog) order, with their boundaries marked. Units are sorted
by spike count, sparsest at the bottom. The dashed line is EQUAL_WINDOW_S:
everything to its right is what panel A's truncation discards. A unit that
fires in only some of a site's recs shows up as a gap aligned to those band
boundaries -- genuine drop-in/drop-out, now that there is a real timeline for
it to be visible against.

The Fano bin width defaults to one stimulus period (1/fig4.FREQ = 0.2 s),
matching fig4B_diagnostics.py's own default so the two diagnostics report the
same numbers; --fano-bin overrides it. Note that a Fano factor measured
across the whole concatenated timeline spans recs with different magnetic
stimuli (mag2/3/5/8, inclined, lights-on), so any rate step at a rec boundary
contributes to it; at 0.2 s bins that is a handful of bins out of ~1750 and
not a material effect, but at a much larger --fano-bin it would be.

No Fourier computation happens here -- this reads spike trains from each
experiment's NWB and counts them, so it needs no fig4 cache and takes
seconds.

Usage:
    python pipeline/manuscript/fig4B_fano_by_rec.py
    python pipeline/manuscript/fig4B_fano_by_rec.py --fano-bin 1.0
    python pipeline/manuscript/fig4B_fano_by_rec.py --no-rasters
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

from pipeline import nwb_io
from pipeline.schema import load_experiment

import fig4
import format_parameters as FP
# Reused rather than reimplemented so this diagnostic's Fano factor is
# bit-for-bit the one fig4B_diagnostics.py's panel C already reports.
from fig4B_diagnostics import fano_factor

OUT_NAME = "Fig4B_fano_by_rec.pdf"

# One hue per source recording site. Dark2 (as in fig4.PALETTE_FIG4B) rather
# than seaborn's Set1 default, whose first two entries are close enough to the
# manuscript's magnetic/visual blue and orange to be misread as them -- these
# curves are sites, not stimulus conditions.
PALETTE = "Dark2"

# Poisson reference. A renewal process with Fano == 1 is the null a count
# variance is usually read against, so the line is what makes "these units
# are over- or under-dispersed" readable off the x axis.
POISSON_REF = 1.0

# Raster page colors: whether a unit survives truncation to EQUAL_WINDOW_S is
# the ONLY thing encoded, so the two colors are a neutral dark grey (kept)
# against one fixed accent (dropped), the same on every page. Deliberately not
# the per-site Dark2 hues page 1 uses -- a row's color here means
# kept/dropped, and reusing the site palette would suggest it still means
# "which site", which the page title already says.
KEPT_COLOR    = "0.25"
DROPPED_COLOR = "#d62728"
# Lighter tints of the same two colors for the per-unit active-span underlay,
# so a unit's extent stays visible where its ticks are too sparse to see (see
# plot_raster_page) without the underlay competing with the ticks on top.
KEPT_SPAN_COLOR    = "0.8"
DROPPED_SPAN_COLOR = "#f6b8b8"

# Alternating rec-band shading. Two near-white greys rather than one shaded
# band per rec: the bands exist to make the concatenation legible, not to draw
# attention, and anything darker competes with the spike ticks drawn over it.
REC_BAND_COLORS = ("white", "0.94")

# Raster tick/span sizes. A site contributes up to ~620 units x several
# thousand spikes, so these pages are ~1-3M ticks each: drawn as one scatter
# call per kept/dropped group (not one per unit) with rasterized=True, so the
# PDF stays small and the vector text on top stays sharp. A "|" tick marker
# rather than ".": at a few hundred rows per page a round marker of legible
# area bleeds vertically into neighboring units, while a vertical bar grows
# only along the time axis.
RASTER_MARKER_S = 1.5
RASTER_SPAN_LW  = 0.5
RASTER_FIGSIZE  = (12, 7)


def load_raw_units(data_dir: str, experiments):
    """Every experiment's mag-contingency units, keyed
    "{experiment}:{cluster_id}", on that experiment's own concatenated
    timeline but with NO truncation to a common window -- the dict
    fig4.load_pseudopopulation_spks builds immediately before it calls
    truncate_pseudopop_to_window. Panel B plots this; panel A plots the
    truncated version of the same dict.

    Also returns each experiment's total concatenated duration, which is
    what the ECDF page needs to say how much longer panel B's observation is
    than panel A's, per site.
    """
    all_units, total_Ts = {}, {}
    for experiment in experiments:
        unit_spks, _Q_frac, total_T = fig4.load_unit_spks_for_experiment(
            data_dir, experiment)
        all_units.update(unit_spks)
        total_Ts[experiment] = total_T
    print(f"  {len(all_units)} mag units loaded from {len(experiments)} experiments "
          f"(concatenated, before any common-window truncation)")
    return all_units, total_Ts


def rec_layout(data_dir: str, experiment: str):
    """This experiment's mag recs as (rec, offset, duration) in concatenated
    order, plus the total duration -- the block structure a raster page draws
    as background bands.

    Delegates to fig4.concat_mag_recs rather than recomputing the offsets, so
    a page's bands are guaranteed to be the same tiling the spike times on it
    were shifted onto.
    """
    cfg = load_experiment(fig4._EXPERIMENTS_DIR / f"{experiment}.yml")
    io_r, nwbfile = nwb_io.read_nwbfile(str(Path(data_dir) / f"{experiment}.nwb"))
    rec_map, total_T = fig4.concat_mag_recs(nwbfile, cfg.analysis.mag_rec_substring)
    io_r.close()
    blocks = sorted((offset, dur, rec) for rec, (_ls, dur, offset) in rec_map.items())
    return [(rec, offset, dur) for offset, dur, rec in blocks], total_T


def fano_frame(unit_spks, bin_s, population):
    """One row per unit: its Fano factor, its active span, and which site it
    came from. `population` labels which panel the row belongs to.

    Units whose Fano factor is undefined (fewer than two whole bins in their
    span, or a zero mean count -- see fano_factor) come back as NaN and are
    dropped here, with a count printed, rather than silently vanishing into
    seaborn's own NaN handling.
    """
    rows = [{
        "unit": key,
        "experiment": key.split(":")[0],
        "T": spkt.max() - spkt.min(),
        "n_spk": len(spkt),
        "fano": fano_factor(spkt, bin_s),
        "population": population,
    } for key, spkt in unit_spks.items()]
    df = pd.DataFrame(rows)
    n_nan = int(df["fano"].isna().sum())
    if n_nan:
        print(f"  [{population}] {n_nan} of {len(df)} units have no defined Fano factor "
              f"at {bin_s:.3g} s bins -- dropped")
    return df.dropna(subset=["fano"]).reset_index(drop=True)


def print_summary(df, label, bin_s):
    """Per-site Fano summary plus a Kruskal-Wallis test of whether the four
    distributions differ at all -- the ECDFs show the shape, this says
    whether the separation is more than sampling noise.
    """
    print(f"\n[{label}] per-site Fano factor ({bin_s:.3g} s bins):")
    summary = df.groupby("experiment").agg(
        n_units=("fano", "size"),
        # mean_T alongside median_T specifically because mean_T is what the
        # legend labels print -- printing only the median would leave no
        # printed number matching the figure.
        mean_T=("T", "mean"),
        median_T=("T", "median"),
        median_n_spk=("n_spk", "median"),
        fano_p25=("fano", lambda s: s.quantile(0.25)),
        median_fano=("fano", "median"),
        fano_p75=("fano", lambda s: s.quantile(0.75)),
    )
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    groups = [g["fano"].values for _, g in df.groupby("experiment")]
    if len(groups) > 1 and all(len(g) > 1 for g in groups):
        res = sp_stats.kruskal(*groups)
        print(f"  Kruskal-Wallis across {len(groups)} sites: "
              f"H={res.statistic:.1f}, p={res.pvalue:.2e}")


def plot_panel(ax, df, bin_s, title):
    """One panel: an ECDF of Fano factor per site, on a shared log x.

    Legend labels carry each site's unit count and the MEAN per-unit active
    span -- the four sites contribute very unequal n, and span is the
    confound the whole comparison turns on, since a Fano factor measured over
    785 s is not the same observation as one measured over 350 s. Showing it
    per curve is what makes panel A's tighter spread readable as "the
    observation windows are now equal" rather than as an unexplained
    difference between the panels.

    The span averaged is each unit's own first-to-last-spike interval, which
    is what fano_factor actually bins -- not the site's full concatenated
    duration. In panel A the two coincide for all but the sparsest units; in
    panel B the span is close to the site's whole timeline.
    """
    order = sorted(df["experiment"].unique())
    counts = df.groupby("experiment").size()
    mean_T = df.groupby("experiment")["T"].mean()
    sns.ecdfplot(data=df, x="fano", hue="experiment", hue_order=order,
                 palette=PALETTE, linewidth=1.4, ax=ax)
    ax.axvline(POISSON_REF, color="grey", linestyle="dashed", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(f"Fano factor ({bin_s:.3g} s bins)")
    ax.set_ylabel("Cumulative fraction of units")
    ax.set_title(title, fontsize=FP.FS_TITLE)
    # sns.ecdfplot puts the site names in its own legend; relabel with the
    # per-site n and mean span rather than adding a second legend. The title
    # states what the two numbers are, so each row can stay compact enough
    # not to crowd the axes.
    legend = ax.get_legend()
    if legend is not None:
        handles = legend.legend_handles
        ax.legend(handles,
                  [f"{e}  n={counts[e]}, {mean_T[e]:.0f} s" for e in order],
                  title="site (units, mean span)",
                  fontsize=FP.FS_LEGEND,
                  title_fontsize=FP.FS_LEGEND, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def units_by_experiment(unit_spks):
    """Split the pooled "{experiment}:{cluster_id}" dict back into one dict
    per source site, sites in name order.
    """
    by_exp = {}
    for key, spkt in unit_spks.items():
        by_exp.setdefault(key.split(":")[0], {})[key] = spkt
    return {e: by_exp[e] for e in sorted(by_exp)}


def plot_raster_page(experiment, unit_spks, window_s, blocks, total_T,
                     min_spikes=fig4.MIN_SPIKES):
    """One raster page for one site: every mag unit on its own horizontal
    line, on that site's concatenated timeline.

    Alternating background bands are the site's individual mag recs, in the
    same concatenated order the spike times were shifted onto (`blocks`, from
    rec_layout). Without them the page reads as one continuous recording,
    which is precisely the misreading that motivated this whole diagnostic.

    Each unit also gets a light ACTIVE-SPAN segment (first spike to last)
    under its ticks. A few hundred units on one page leaves each row well
    under a point tall, so a sparse unit's ticks alone are close to
    invisible -- and it is the sparse units that truncation drops, so a
    spikes-only raster hides the population the page is about.

    Rows are sorted by spike count, sparsest at the bottom, which groups the
    units truncation will drop and makes a unit that fires in only some recs
    read as a gap aligned to the band boundaries.
    """
    keys = sorted(unit_spks, key=lambda k: len(unit_spks[k]))
    # Same predicate fig4.truncate_pseudopop_to_window applies: what survives
    # is decided by spike count inside the window, not by span.
    kept = np.array([int((unit_spks[k] < window_s).sum()) >= min_spikes
                     for k in keys])
    starts = np.array([unit_spks[k].min() for k in keys])
    stops = np.array([unit_spks[k].max() for k in keys])

    fig, ax = plt.subplots(figsize=RASTER_FIGSIZE)

    # Rec bands first, so every tick and segment is drawn over them.
    for band_i, (_rec, offset, dur) in enumerate(blocks):
        ax.axvspan(offset, offset + dur, ymin=0, ymax=1, zorder=0,
                   facecolor=REC_BAND_COLORS[band_i % len(REC_BAND_COLORS)],
                   edgecolor="none")
        if band_i:
            ax.axvline(offset, color="0.75", linewidth=0.6, zorder=1)

    groups = (
        (kept,  KEPT_COLOR,    KEPT_SPAN_COLOR,    f"kept (n={int(kept.sum())})"),
        (~kept, DROPPED_COLOR, DROPPED_SPAN_COLOR, f"dropped (n={int((~kept).sum())})"),
    )
    for mask, color, span_color, label in groups:
        idx = np.where(mask)[0]
        if not len(idx):
            continue
        ax.hlines(idx, starts[idx], stops[idx], color=span_color,
                  linewidth=RASTER_SPAN_LW, rasterized=True, zorder=2)
        # Accumulated into one scatter call per group rather than one per
        # unit -- a site contributes up to a few million ticks.
        xs = np.concatenate([unit_spks[keys[i]] for i in idx])
        ys = np.concatenate([np.full(len(unit_spks[keys[i]]), i) for i in idx])
        ax.scatter(xs, ys, s=RASTER_MARKER_S, marker="|", linewidths=0.2,
                   color=color, label=label, rasterized=True, zorder=3)

    ax.axvline(window_s, color="black", linestyle="dashed", linewidth=1.2, zorder=4)
    ax.set_xlim(0, total_T)
    ax.set_ylim(-0.01 * len(keys), 1.01 * len(keys))
    ax.set_xlabel(f"Time on {experiment}'s concatenated timeline (s) "
                  f"-- alternating bands are its {len(blocks)} mag recordings")
    ax.set_ylabel("Unit (sorted by spike count, sparsest at bottom)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # markerscale is large because RASTER_MARKER_S is sized for a
    # multi-million-tick raster, not for a legend key.
    ax.legend(loc="lower right", markerscale=12, fontsize=FP.FS_LEGEND,
              title=f"truncation to {window_s:g} s", title_fontsize=FP.FS_LEGEND)

    rec_durs = [dur for _rec, _o, dur in blocks]
    fig.suptitle(
        f"{experiment}: {len(blocks)} mag recordings concatenated "
        f"({min(rec_durs):.0f}-{max(rec_durs):.0f} s each, {total_T:.0f} s total), "
        f"{len(keys)} mag units\n"
        f"dashed line = the {window_s:g} s common window; "
        f"{int((~kept).sum())} units fall below {min_spikes} spikes inside it",
        fontsize=FP.FS_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def ecdf_figure(df_eq, df_raw, bin_s, window_s, total_Ts):
    """Page 1: the two Fano ECDF panels."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_panel(axes[0], df_eq, bin_s,
               f"Fig4's population: first {window_s:g} s of each site's "
               f"concatenated timeline")
    span = (f"{min(total_Ts.values()):.0f}-{max(total_Ts.values()):.0f} s"
            if total_Ts else "full")
    plot_panel(axes[1], df_raw, bin_s,
               f"Full concatenated timeline per site ({span}), no truncation")
    for ax, letter in zip(axes, "AB"):
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontfamily="arial",
                fontsize=FP.FS_PANEL, ha="left", va="bottom")

    fig.suptitle("Fano-factor ECDFs by recording site, pigeon-HP pseudopopulation\n"
                 "(all four sites in both panels; dashed line = Poisson, Fano = 1)",
                 fontsize=FP.FS_TITLE)
    # rect top at 0.96, not the 0.92 a two-line suptitle would suggest: at this
    # figure's 4.5 in height 8% is over a third of an inch, which left a
    # conspicuous band of white between the title and the axes.
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def write_pdf(df_eq, df_raw, raw_units, bin_s, window_s, out_path: Path,
              total_Ts, rasters=True, data_dir=None):
    """Page 1 = the ECDF comparison, then one raster page per site.

    bbox_inches="tight" is deliberately NOT passed to pdf.savefig here, as it
    was back when this was a single-page figure: with it every page is
    cropped to its own content and so comes out a different size, which reads
    badly when flipping between pages.
    """
    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY_XL}
    matplotlib.rc("font", **font)

    figs = [ecdf_figure(df_eq, df_raw, bin_s, window_s, total_Ts)]
    if rasters:
        for experiment, units in units_by_experiment(raw_units).items():
            blocks, total_T = rec_layout(data_dir, experiment)
            print(f"  raster page: {experiment} ({len(units)} units, "
                  f"{len(blocks)} mag recs, {total_T:.1f} s timeline)")
            figs.append(plot_raster_page(experiment, units, window_s,
                                         blocks, total_T))

    with PdfPages(out_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=FP.DPI_FIG4)
            if in_notebook:
                plt.show()
            else:
                plt.close(fig)
    print(f"\nSaved {len(figs)} pages -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fano-factor ECDFs and concatenated-timeline rasters, "
                    "one per pigeon-HP recording site")
    parser.add_argument("--out-dir", default=FP.DIAGNOSTIC_DIR, help="Output directory for the PDF")
    parser.add_argument("--data-dir", default=FP.DATA_DIR,
                        help="Directory containing pipeline output .nwb files")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Explicit experiment names to pool (default: fig4's own "
                             "auto-discovery of species=='Pigeon' & area=='HP')")
    parser.add_argument("--fano-bin", type=float, default=1 / fig4.FREQ,
                        help="Fano-factor bin width in seconds (default: 1/FREQ, one "
                             "stimulus period -- same default as fig4B_diagnostics.py)")
    parser.add_argument("--no-rasters", action="store_true",
                        help="Write only the ECDF page, skipping the per-site "
                             "raster pages (the slow part)")
    args = parser.parse_args([] if in_notebook else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = args.experiments or fig4.discover_pigeon_hp_experiments(fig4._EXPERIMENTS_DIR)
    print(f"Pooling pigeon-HP pseudopopulation from: {experiments}")
    raw_units, total_Ts = load_raw_units(args.data_dir, experiments)
    # Panel A's units are exactly panel B's, put through the same truncation
    # fig4 applies at load time -- called on the same dict rather than
    # reloading, so the two panels can never describe different underlying
    # data.
    eq_units = fig4.truncate_pseudopop_to_window(raw_units)

    df_raw = fano_frame(raw_units, args.fano_bin, "full timeline")
    df_eq = fano_frame(eq_units, args.fano_bin, f"{fig4.EQUAL_WINDOW_S:g} s window")

    print_summary(df_eq, f"panel A -- common {fig4.EQUAL_WINDOW_S:g} s window", args.fano_bin)
    print_summary(df_raw, "panel B -- full concatenated timeline", args.fano_bin)

    write_pdf(df_eq, df_raw, raw_units, args.fano_bin, fig4.EQUAL_WINDOW_S,
              out_dir / OUT_NAME, total_Ts, rasters=not args.no_rasters,
              data_dir=args.data_dir)


if __name__ == "__main__":
    if in_notebook:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    main()
