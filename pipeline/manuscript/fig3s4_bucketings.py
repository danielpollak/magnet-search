"""Fig 3 variants -- one multipage PDF of alternative bucketings of the
p-/q-value uniformity figure.

"Canonical" below means the occurrence-wave version of Fig 3 (page 1 of
Fig3S2_all_recordings.pdf): buckets = occurrence waves; quadrants = all neurons /
pigeon / pigeon HP / top 10% most sensitive pigeon HP. It used to be page 1
here too, and page 2 was the same buckets as per-bucket ECDF-deviation stacks
(both dropped 2026-10-19, the pages below renumbered); the PDF's page numbers
are printed at the bottom of each page.

Page 1 Quadrants = species; buckets = stimulus FREQUENCY. A neuron that
        appears more than once at the same frequency contributes its first row
        to that frequency's bucket and every later one to a separate
        "<f> Hz repeats" bucket for that frequency (see `bucket_by_frequency`)
        -- so no first-occurrence bucket ever counts the same neuron twice,
        and the repeat rows are still shown rather than dropped.
Page 2  Quadrants = species; buckets = BRAIN REGION. A unit repeated within
        the same region is kept only the first time (unlike page 1, repeats
        are dropped, not re-bucketed) -- region is a property of the
        electrode, not of the recording, so a unit's 2nd row in the same
        region carries no new region information.
Page 3  Canonical quadrants again, but buckets are EQUAL-SIZED: rows are dealt
        round-robin so every bucket ends up within one row of N/B, with no
        unit twice in a bucket (see `bucket_round_robin`). This is the same
        population as the canonical figure with the same number of buckets,
        rebalanced -- its occurrence waves shrink from thousands of units to a
        few hundred, which makes the
        later buckets' apparent deviation mostly small-n noise.
Page 4  Page 3's equal-sized buckets, the sorted-p-value panel replaced by a
        stack of per-bucket ECDF-deviation mini-axes in the style of Fig 1's
        inset: ECDF(p) - p over the null's 95% binomial band, uniform null at
        0. Equal sizes give every row the same band. The sorted-q-value
        panel below is unchanged.
Page 5  The full cross-tabulation: one panel per (species, brain area) pair
        with magnetic units, on a 3-wide grid, x frequency buckets. Owl,
        which has units but no Fourier p-value at all, gets "no p-values"
        (see `_blank_reason`).

Pages 6-9 are two readings of one question -- does brain area matter -- kept
side by side because which one answers it depends on whether you want each
region judged on its own or the regions judged against each other:

Page 6  Pigeon, one panel per brain area, that area's units pooled into a
        single bucket. Regions compared ACROSS panels, each on its own axes.
Page 7  The same for every (species, area) pair -- i.e. page 5's panels with
        the frequency split collapsed.
Page 8  Pigeon, ONE panel, brain areas as buckets overplotted on shared axes.
        Regions directly comparable.
Page 9  The same for all species, bucketed by (species, area) rather than by
        bare area -- see `bucket_by_species_area` for why the pair matters.
Page 10 One panel, one bucket per species, each unit counted exactly once.
Page 11 Canonical quadrants, ONE bucket holding every observation,
        drawn with an occurrence bootstrap: each draw keeps every
        neuron once but picks WHICH of its recordings at random
        (`statistics.iter_occurrence_draws`), so repeat recordings are used
        rather than dropped or double-counted. Top row: ECDF(p) - p, median
        and 95% band over draws (what Fig 2 C/D drew until 2026-10-19). Bottom row: sorted
        q-values from the same draws, median and 95% band per rank.

Every page shows only the magnetic (negative-result) population. Until
2026-11-12 each quadrant also had a right-hand half for the visual/auditory
positive controls, and a page matching the magnetic bucket size to the A/V
one; both were dropped as irrelevant to the question these splits ask
(could a subset of the magnetic data hide responders?). Bucket colour is a
shade of Fig 3's magnetic colormap.

Requires:
  data/manuscript/all_fourier_df.parquet  (run python pipeline/aggregate.py first)

Usage:
    python pipeline/manuscript/fig3s4_bucketings.py
    python pipeline/manuscript/fig3s4_bucketings.py --pages 1 2
    python pipeline/manuscript/fig3s4_bucketings.py --out-dir figs/paper
"""
import argparse
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from magpyneto2 import statistics
import fig3

import format_parameters as FP


# Identity key for "the same neuron" -- same as fig3._split_into_waves'
# default, and for the same reason (see get_poscontrols_negresults: `id`
# restarts per subject, so `date`+`id` alone collides two animals recorded on
# one calendar date).
NEURON_KEY = list(statistics.UNIQUE_NEURON_KEY)

# Fig 3's 2x2 quadrant grid positions, in fig3_conditions() order.
QUADRANT_SLOTS = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Page 1's four species panels (the four with both magnetic and A/V data,
# from when the A/V half was still drawn).
SPECIES_PANELS = ["Pigeon", "zebra finch", "zebrafish", "medaka"]

# Frequencies are grouped on a 2-decimal key rather than their raw float
# value. This is what merges the two oddball frequencies (0.977875 and
# 0.978892 Hz -- the same stimulus, whose nominal rate differs in the 4th
# decimal between sessions) into one bucket, and likewise the two spellings of
# the 1/60 Hz visual frequency (0.016666666... written by one analysis stage,
# 0.016667 by another). Bucket LABELS use the bucket's own median raw
# frequency at 3 significant figures, so the 1/60 Hz bucket still reads
# "0.0167 Hz" rather than the rounding key's "0.02 Hz".
FREQ_KEY_DECIMALS = 2

N_BOOT = 1000

# Panels with at most this many buckets take their shades from this part of
# fig3.cmap_to_black's ramp (0 = light tint, 0.5 = saturated hue, 1 = black)
# instead of all of it -- see `_shades`.
NARROW_SHADES_MAX_N = 5
NARROW_SHADES_RANGE = (0.3, 0.65)

# Colormap position used for EVERY bucket in an ECDF-deviation stack (page 4),
# instead of the shade ramp the sorted-p/q panels use. Those panels
# overplot all buckets on one axes, where varying shade is the only thing
# separating them; a stack gives each bucket its own axes, so there the ramp
# conveys nothing and only costs legibility -- the light end of Blues/Oranges
# is thin and washed out against white at a 1pt linewidth. 0.9 is the dark end
# of the same ramp, i.e. what the darkest bucket already was.
ECDF_SHADE = 0.9

# Seed for the row/unit shuffle that `bucket_round_robin` deals from. Fixed so
# the page is reproducible, but non-trivial so the bucketing is not an
# artifact of parquet row order (same reasoning as fig3s3_reproducibility's RANDOM_SEED).
RR_SEED = 0

# Columns for page 5's species x area grid. 3 gives a roughly portrait page for
# the current 13 (species, area) pairs; the row count follows from the pair
# count, so this is the only knob that needs touching if the annotations grow.
SPECIES_AREA_NCOLS = 3


# -- Bucketings ---------------------------------------------------------------

def bucket_by_frequency(df, split_repeats=False):
    """Buckets = distinct stimulus frequencies, ascending, plus a trailing
    "repeats" bucket (or, with `split_repeats`, one trailing "<f> Hz repeats"
    bucket per frequency, ascending -- page 1).

    A neuron recorded twice at the same frequency (e.g. two 3 Hz recordings in
    one session) would otherwise appear twice inside one bucket, which breaks
    the assumption the whole figure rests on -- that a bucket's p-values are
    one p-value per neuron, so their ECDF can be read against a uniform null
    and Storey's estimator applied to them. Later occurrences are therefore
    moved out into their own bucket rather than silently double-counted. They
    are kept (not dropped) because that pooled repeat bucket is itself
    informative: it is exactly the population whose neurons got a second
    independent look at the same stimulus.
    """
    df = df.copy()
    df["_fkey"] = np.round(df["freq"].values, FREQ_KEY_DECIMALS)
    df["_occ"] = df.groupby(NEURON_KEY + ["_fkey"], dropna=False).cumcount()

    buckets = []
    firsts = df.loc[df["_occ"] == 0]
    for _, g in firsts.groupby("_fkey", sort=True):
        buckets.append((f"{np.median(g['freq'].values):.3g} Hz", g))

    repeats = df.loc[df["_occ"] > 0]
    if split_repeats:
        for _, g in repeats.groupby("_fkey", sort=True):
            buckets.append((f"{np.median(g['freq'].values):.3g} Hz repeats", g))
    elif len(repeats):
        buckets.append(("repeats", repeats))
    return buckets


def bucket_by_area(df):
    """Buckets = brain regions, largest first. Repeat rows for the same neuron
    in the same region are dropped after the first (see module docstring).
    """
    df = df.copy()
    df["_occ"] = df.groupby(NEURON_KEY + ["area"], dropna=False).cumcount()
    firsts = df.loc[df["_occ"] == 0]
    groups = [(str(area), g) for area, g in firsts.groupby("area", sort=False)]
    groups.sort(key=lambda kv: -len(kv[1]))
    return groups


def bucket_round_robin(df, n_buckets=None, seed=RR_SEED):
    """Buckets of (near-)equal size, dealt round-robin, never placing the same
    unit in a bucket twice.

    Why this is exact rather than approximate. Deal the rows out *grouped by
    unit*: lay every row in a single list with each unit's rows contiguous,
    then send position i to bucket `i % B`. A unit owning m rows therefore
    occupies B consecutive positions' worth of a cyclic walk -- m distinct
    buckets, as long as m <= B -- so the "skip a bucket this unit is already
    in" case never arises at all, and no bucket can hold a unit twice. And
    because the deal is one unbroken cyclic sweep of length N, every bucket
    receives either floor(N / B) or ceil(N / B) rows. Both properties hold
    simultaneously and by construction, not on average: verified on the real
    data, all four quadrants come out with max(size) - min(size) <= 1 and at
    most one row per unit per bucket.

    `n_buckets` defaults to the busiest unit's row count, which is the smallest
    B for which the above can hold -- and, for the canonical populations, is
    also exactly the number of occurrence waves the canonical figure uses, so
    page 3 is its buckets rebalanced rather than a different
    number of them. A smaller B is accepted but cannot be honoured in full:
    rows past a unit's B-th have nowhere to go without duplicating it, so they
    are dropped and the count is printed.

    Bucket INDEX carries no meaning here -- unlike an occurrence wave ("each
    unit's k-th recording") or a frequency, a round-robin bucket is an
    arbitrary mixed sample of units, each contributing one arbitrary
    recording. That is the point: it makes the buckets statistically
    exchangeable, so their spread really is a null spread, but it also means
    the shade ramp is decorative and the buckets should not be read in order.
    """
    if len(df) == 0:
        return []

    rng = np.random.default_rng(seed)
    df = df.iloc[rng.permutation(len(df))]
    # Shuffling FIRST and factorizing after randomizes both which unit is
    # dealt when (factorize codes units by order of first appearance) and
    # which of a unit's own recordings lands in which of its buckets.
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(df[NEURON_KEY]))
    order = np.argsort(codes, kind="stable")
    df, codes = df.iloc[order], codes[order]

    mult = np.bincount(codes)
    max_mult = int(mult.max())
    n_buckets = max_mult if n_buckets is None else max(1, int(n_buckets))

    def _positions(codes, mult):
        starts = np.concatenate([[0], np.cumsum(mult)[:-1]])
        return starts[codes] + (np.arange(len(codes)) - starts[codes]), starts

    pos, starts = _positions(codes, mult)
    if n_buckets < max_mult:
        within = pos - starts[codes]
        keep = within < n_buckets
        print(f"    round-robin: dropping {int((~keep).sum())} of {len(df)} rows "
              f"-- busiest unit has {max_mult} rows but only {n_buckets} buckets")
        df, codes = df.iloc[np.flatnonzero(keep)], codes[keep]
        # Re-derive positions on the surviving rows: the deal has to be one
        # unbroken sweep to stay balanced, and dropping rows leaves gaps in
        # the original position numbering.
        mult = np.bincount(codes, minlength=mult.size)
        pos, _ = _positions(codes, mult)

    df = df.assign(_rr=pos % n_buckets)
    return [(f"#{b + 1}", g) for b, g in df.groupby("_rr", sort=True)]


def _dedup_units(df, extra_key=()):
    """One row per unit (per `extra_key` group, if given), first occurrence
    kept. Later rows are DROPPED, not re-bucketed -- see `bucket_by_frequency`
    for the case where they are worth keeping instead.
    """
    return df.loc[df.groupby(NEURON_KEY + list(extra_key), dropna=False)
                  .cumcount() == 0]


def bucket_by_species(df):
    """A single bucket per species, each holding that species' units once.

    Deduplication is unconditional here, and across the whole species rather
    than within a recording: a species bucket is meant to be "this species'
    units", so a unit recorded five times would otherwise weight that species
    five times over in its own ECDF and its own Storey correction.
    """
    d = _dedup_units(df)
    return [(str(sp), g) for sp, g in d.groupby("species", sort=False)]


def bucket_by_species_area(df):
    """One bucket per (species, brain area) pair, labelled with both.

    Distinct from `bucket_by_area` specifically because "WB" is not one region:
    it is zebrafish whole-brain in one species and medaka whole-brain in
    another. Bucketing on the bare area name silently pools two different
    animals' entire brains into a single bucket; this keeps them apart, which
    matters on any page that puts more than one species' areas on shared axes.
    """
    d = _dedup_units(df, extra_key=["species", "area"])
    groups = [(f"{sp}, {ar}", g) for (sp, ar), g in
              d.groupby(["species", "area"], sort=False)]
    groups.sort(key=lambda kv: -len(kv[1]))
    return groups


def _magnetic_frame(all_fourier_df):
    """The magnetic (negative-result) population -- the only one these pages
    draw."""
    neg, _, _ = statistics.get_poscontrols_negresults(all_fourier_df)
    return neg


def _shades(cmap_name, n):
    # With few buckets the full tint-to-black ramp puts them at its extremes
    # (a washed-out tint next to black); a narrower stretch around the
    # saturated hue reads better. A lone bucket takes the middle of that
    # stretch rather than fig3's black.
    if 0 < n <= NARROW_SHADES_MAX_N:
        ramp = fig3.cmap_to_black(cmap_name)
        pos = np.linspace(*NARROW_SHADES_RANGE, n) if n > 1 else [np.mean(NARROW_SHADES_RANGE)]
        return list(ramp(pos))
    return fig3.shades_to_black(cmap_name, n)


def _drop_empty(buckets):
    return [(lab, g) for lab, g in buckets if len(g) > 0]


# -- Sorted p/q panel (pages 1 and 2) -----------------------------------------

def _bucket_legend(ax, labels, colors, loc="upper left"):
    """Bucket identity is the entire point of pages 1 and 2, so unlike the
    canonical figure (whose buckets are an ordered wave index that the shade
    ramp already conveys) these need a real key. `loc="upper left"` is safe on
    a sorted-p-value axes specifically: a sorted curve runs from the bottom
    left to the top right, so the upper-left corner is empty by construction.
    """
    if not labels:
        return
    handles = [Line2D([], [], marker="o", linestyle="none", color=c, markersize=FP.MS_SMALL)
               for c in colors]
    ncol = 2 if len(labels) > 6 else 1
    # A translucent white patch behind the key rather than frameon=False: the
    # upper-left corner is empty only for the WIDEST bucket's curve, and the
    # steeper (smaller-n) buckets do run through where a 2-column key has to
    # go.
    ax.legend(handles, labels, loc=loc, ncol=ncol, fontsize=FP.FS_LEGEND * 0.7,
              frameon=True, facecolor="white", framealpha=0.75, edgecolor="none",
              handlelength=1.0, handletextpad=0.4,
              labelspacing=0.25, columnspacing=0.8, borderpad=0.15)


def _quadrant_header(fig, cell, title, letter, dy=0.04):
    pos = cell.get_position(fig)
    fig.text((pos.x0 + pos.x1) / 2, pos.y1 + dy, title, ha="center", va="bottom",
             fontsize=FP.FS_BODY + 1, fontweight="bold")
    # Letter out over the y-axis label, 0.45in left of the panel: a single
    # magnetic column is narrower than a long title, which reached the
    # panel's own left edge.
    letter_x = pos.x0 - 0.45 / fig.get_size_inches()[0]
    fig.text(letter_x, pos.y1 + dy, letter, ha="left", va="bottom",
             fontfamily="arial", fontsize=12)


def _bucket_note(buckets):
    """"10 buckets, 818-819 units" -- the summary that replaces the per-bucket
    key when bucket index is meaningless (see `bucket_round_robin`). Naming all
    18 interchangeable buckets in a legend would be noise; their common size is
    the one number a reader actually wants.
    """
    sizes = [len(g) for _, g in buckets]
    if not sizes:
        return ""
    if len(sizes) == 1:
        # A one-bucket panel has no bucket structure to summarize -- the only
        # number that means anything is its n.
        return f"{sizes[0]} units"
    span = f"{min(sizes)}" if min(sizes) == max(sizes) else f"{min(sizes)}-{max(sizes)}"
    return f"{len(sizes)} buckets, {span} units"


def _condition_subset(pop, cond_filter, percentile, sens_col):
    """`pop` restricted to one Fig 3 quadrant: its condition filter, then (for
    the top-10% quadrant) ONE global sensitivity threshold over what's left."""
    sub = pop if cond_filter is None else pop.loc[cond_filter(pop)]
    if percentile is not None and len(sub):
        sub = sub.loc[sub[sens_col] > np.percentile(sub[sens_col], percentile)]
    return sub


def _blank_reason(buckets, pval_col):
    """Why a panel has nothing to draw, or None if it does: "no units", or
    "no p-values" for a population that exists but has no Fourier p-value at
    all -- Owl, whose 972 rows come from the precomputed Bayesian-coefficient
    pickle and were never run through `fit_fourier_sig` (see CLAUDE.md on
    aggregate.py's species coverage).
    """
    if not buckets:
        return "no units"
    if not any(np.isfinite(g[pval_col].values).any() for _, g in buckets):
        n = sum(len(g) for _, g in buckets)
        return f"no p-values\n({n} units)"
    return None


def _plot_bucket_quadrant(fig, cell, buckets, title, letter,
                          pval_col="p_value", sens_col="sens", legend=True,
                          header_dy=0.04):
    """Draw one panel (sorted p-values above, sorted q-values below) of the
    magnetic population for an arbitrary bucketing, reusing
    `fig3.plot_uniform_p` unchanged so the per-bucket Storey correction, axis
    scaling and FDR threshold line are identical to the canonical figure's.

    `legend=False` swaps the per-bucket key for a one-line size summary in the
    title -- for bucketings whose buckets are interchangeable and whose labels
    therefore identify nothing.
    """
    inner = cell.subgridspec(2, 1, hspace=0.12)
    ax_p = fig.add_subplot(inner[0])
    ax_q = fig.add_subplot(inner[1], sharex=ax_p)
    ax_p.tick_params(labelbottom=False)

    buckets = _drop_empty(buckets)
    blank = _blank_reason(buckets, pval_col)
    note = ""
    if blank is not None:
        for ax in (ax_p, ax_q):
            ax.text(0.5, 0.5, blank, transform=ax.transAxes, ha="center",
                    va="center", fontsize=FP.FS_LEGEND, color=FP.COLOR_NULL)
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        # A bucket with rows but no finite p-value would send an all-NaN array
        # into storey_qvalues; nothing to draw from it either way.
        buckets = [(lab, g) for lab, g in buckets
                   if np.isfinite(g[pval_col].values).any()]
        labels = [f"{lab} ({len(g)})" for lab, g in buckets]
        colors = _shades("Blues", len(buckets))
        fig3.plot_uniform_p([g for _, g in buckets], [ax_p, ax_q], colors=colors,
                            pval_col=pval_col, sens_col=sens_col)
        if legend:
            _bucket_legend(ax_p, labels, colors)
        else:
            note = _bucket_note(buckets)

    if note:
        ax_p.set_title(note, fontsize=FP.FS_TITLE)
    ax_p.set_ylabel("Sorted p-values")
    ax_q.set_ylabel("Sorted q-values")
    ax_q.set_xlabel("Unit")

    _quadrant_header(fig, cell, title, letter, dy=header_dy)


# Per-quadrant geometry in INCHES, plus the page margins around the grid.
# Absolute rather than figure-fraction because these pages range from a 2x2
# (pages 1/2) to a 3x5 (page 5): a fraction that leaves the right gap on an
# 11x7in page leaves a 2.5in canyon on a 17x17in one. Height and gaps
# reproduce page 2's original proportions, which were tuned by eye on the
# 2x2; the width is one column of the old magnetic | A/V pair.
QUAD_W_IN, QUAD_H_IN = 2.6, 2.48      # one panel's p-over-q block
QUAD_GAP_W_IN, QUAD_GAP_H_IN = 1.1, 0.87   # gaps between panels
# `top` has to clear BOTH the suptitle and the top row's own quadrant headers,
# which sit 0.29in above their quadrant -- at 0.85in the two collided (the
# suptitle is centred, and on an odd-column grid so is the middle quadrant's
# title).
PAGE_MARGIN_IN = dict(left=0.9, right=0.25, top=1.1, bottom=0.5)


def _grid_figure(n_panels, ncols, scale=1.0):
    """Figure + outer GridSpec sized from QUAD_*_IN, so a quadrant is the same
    physical size whatever the panel count. Returns (fig, outer, slots).

    `scale` enlarges the quadrant itself -- a single-panel page at 1.0 would be
    a 5x4in sheet, which is a thumbnail rather than a page.
    """
    nrows = int(np.ceil(n_panels / ncols))
    m = PAGE_MARGIN_IN
    quad_w, quad_h = QUAD_W_IN * scale, QUAD_H_IN * scale
    figw = m["left"] + ncols * quad_w + (ncols - 1) * QUAD_GAP_W_IN + m["right"]
    figh = m["top"] + nrows * quad_h + (nrows - 1) * QUAD_GAP_H_IN + m["bottom"]

    fig = plt.figure(figsize=(figw, figh))
    outer = fig.add_gridspec(
        nrows, ncols,
        wspace=QUAD_GAP_W_IN / quad_w, hspace=QUAD_GAP_H_IN / quad_h,
        left=m["left"] / figw, right=1 - m["right"] / figw,
        top=1 - m["top"] / figh, bottom=m["bottom"] / figh)
    slots = [(i // ncols, i % ncols) for i in range(n_panels)]
    return fig, outer, slots


def _panel_letters(n):
    """A..Z then AA, AB, ... -- page 5 needs ~13, and a future cut could need
    more than 26.
    """
    from string import ascii_uppercase as AZ
    out = []
    for i in range(n):
        out.append(AZ[i] if i < 26 else AZ[i // 26 - 1] + AZ[i % 26])
    return out


def build_grid_bucket_page(all_fourier_df, panels, bucket_fn, suptitle,
                           ncols=2, pval_col="p_value", sens_col="sens",
                           legend=True, scale=1.0):
    """A page of arbitrarily many panels laid out on a `ncols`-wide grid.

    `panels` is a list of `(title, row_filter)`; `row_filter` is applied to the
    magnetic population (or None for "everything"), and `bucket_fn` then
    buckets it.
    """
    neg = _magnetic_frame(all_fourier_df)
    fig, outer, slots = _grid_figure(len(panels), ncols, scale=scale)
    figh = fig.get_size_inches()[1]

    for (title, row_filter), (orow, ocol), letter in zip(
            panels, slots, _panel_letters(len(panels))):
        neg_s = neg if row_filter is None else neg.loc[row_filter(neg)]
        _plot_bucket_quadrant(fig, outer[orow, ocol], bucket_fn(neg_s),
                              title, letter,
                              pval_col=pval_col, sens_col=sens_col,
                              legend=legend,
                              # Header offset in inches for the same reason the
                              # grid geometry is: above the panel (and its
                              # bucket-size note, if any), whatever the page
                              # height.
                              header_dy=(0.29 if legend else 0.42) / figh)

    fig.suptitle(suptitle, fontsize=FP.FS_BODY + 2, fontweight="bold",
                 y=1 - 0.28 / figh)
    return fig


def build_bucket_page(all_fourier_df, bucket_fn, suptitle, pval_col="p_value",
                      sens_col="sens"):
    """A page whose four panels are the four species of SPECIES_PANELS and
    whose buckets come from `bucket_fn`, applied to each species' magnetic
    population.
    """
    panels = [(sp, lambda d, sp=sp: d.species == sp) for sp in SPECIES_PANELS]
    return build_grid_bucket_page(all_fourier_df, panels, bucket_fn, suptitle,
                                  ncols=2, pval_col=pval_col, sens_col=sens_col)


def species_area_panels(all_fourier_df):
    """Every (species, brain area) pair in the magnetic population, species
    ordered by size and areas likewise within a species, so the densest
    panels come first.
    """
    neg = _magnetic_frame(all_fourier_df)
    pairs = set(map(tuple, neg[["species", "area"]].dropna().values))

    def _n(species, area=None):
        m = neg.species == species
        if area is not None:
            m &= neg.area == area
        return int(m.sum())

    ordered = sorted(pairs, key=lambda sa: (-_n(sa[0]), sa[0], -_n(*sa), sa[1]))
    return [(f"{sp}, {ar}", lambda d, sp=sp, ar=ar: (d.species == sp) & (d.area == ar))
            for sp, ar in ordered]


def area_panels(all_fourier_df, species):
    """One panel per brain area of a single species, largest first."""
    return [(title, f) for title, f in species_area_panels(all_fourier_df)
            if title.startswith(f"{species}, ")]


def build_condition_bucket_page(all_fourier_df, bucket_fn, suptitle,
                                pval_col="p_value", sens_col="sens", legend=True):
    """Like `build_bucket_page`, but the four panels are the CANONICAL
    conditions (`fig3_conditions`) rather than four species -- so the page is
    directly comparable to the canonical figure, differing only in how the same population
    is cut into buckets.

    Note the sensitivity filter for the top-10% quadrant runs over that
    quadrant's whole population, before bucketing, where the canonical figure
    takes the top 10% of each wave separately. It has to: the buckets don't
    exist yet at that point. The two come to the same total count, but this
    version uses one global sensitivity threshold instead of B different ones,
    which is also the more defensible of the two.
    """
    neg = _magnetic_frame(all_fourier_df)

    fig = plt.figure(figsize=(FP.FIGSIZE_FIG3[0] * 1.0, FP.FIGSIZE_FIG3[1] * 1.8))
    # legend=False puts the bucket-size note above each panel, so the panel
    # header has to sit higher to clear it -- and the grid lower, so the
    # raised top-row headers still clear the suptitle.
    header_dy = 0.02 if legend else 0.055
    outer = fig.add_gridspec(2, 2, wspace=0.45, hspace=0.42,
                             left=0.12, right=0.98, top=0.85 if legend else 0.82,
                             bottom=0.07)

    for (title, cond_filter, percentile), (orow, ocol), letter in zip(
            fig3_conditions(), QUADRANT_SLOTS, "ABCD"):
        sub = _condition_subset(neg, cond_filter, percentile, sens_col)
        _plot_bucket_quadrant(fig, outer[orow, ocol], bucket_fn(sub),
                              title, letter, pval_col=pval_col, sens_col=sens_col,
                              legend=legend, header_dy=header_dy)

    fig.suptitle(suptitle, fontsize=FP.FS_BODY + 2, fontweight="bold", y=0.975)
    return fig


# -- ECDF-deviation stack (page 4) --------------------------------------------

def _ecdf_deviation(pvals):
    """ECDF(p) - p at each sorted p-value, plus the number of finite p-values.
    Same construction as Fig 1's ECDF-deviation inset.
    """
    pvals = np.asarray(pvals, dtype=float)
    pvals = pvals[np.isfinite(pvals)]
    if len(pvals) < 2:
        return None
    x = np.sort(pvals)
    e = np.arange(1, len(x) + 1) / len(x)
    return x, e - x, len(x)


def _nice_step(span, frac=1 / 3):
    """A round number roughly `frac` of `span`, for a scale bar's length.

    The stacks' y spans are whatever the data happen to be (0.06 for one
    quadrant, 0.6 for another), so the bar length has to be derived rather
    than hardcoded -- but it also has to be a number a reader can do
    arithmetic with, hence the 1/2/2.5/5 mantissa ladder rather than just
    span/3.
    """
    raw = span * frac
    if not np.isfinite(raw) or raw <= 0:
        return None
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _y_scale_bar(ax, step, fontsize):
    """A vertical scale bar in place of a y axis, drawn once per stack.

    Every mini-axes in a stack shares one y scale, so one bar describes all of
    them -- the same argument that previously put y tick labels on a single
    row, taken to its conclusion: with a bar there is no need for a y axis
    line, ticks or labels on ANY row.

    The bar's foot sits exactly at y=0, so it doubles as the label for the
    dashed zero line running through every row -- which, with the frame gone,
    is the only y reference left. x is in axes fractions and y in data units
    (`get_yaxis_transform`), so the bar hugs the left edge of the axes while
    its length stays honest in data units.
    """
    trans = ax.get_yaxis_transform()
    x = -0.015
    ax.plot([x, x], [0, step], transform=trans, color="black", linewidth=1.2,
            clip_on=False, solid_capstyle="butt", zorder=5)
    for y, text in ((step, f"{step:g}"), (0, "0")):
        ax.annotate(text, xy=(x, y), xycoords=trans, xytext=(-2, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=fontsize, annotation_clip=False)


def _add_stack_scale_bar(axs):
    """The stack's single y scale bar, on its middle row (see `_y_scale_bar`).
    Separate from `_plot_ecdf_dev_stack` so a caller that re-scales the stack
    afterwards can draw the bar once the y range is final."""
    if not axs:
        return
    step = _nice_step(float(np.diff(axs[0].get_ylim())[0]))
    if step is not None:
        _y_scale_bar(axs[len(axs) // 2], step, FP.FS_LEGEND * 0.7)


def _plot_ecdf_dev_stack(fig, cell, buckets, cmap, xlabel=True, ylabel=None,
                         pval_col="p_value"):
    """One mini-axes per bucket, stacked and sharing x (p-value) and y
    (deviation). Sharing y is what makes the stack readable as a single
    comparison -- the buckets differ by orders of magnitude in n, so letting
    each autoscale would make a tiny, noisy bucket look as deviant as a large,
    genuinely skewed one.

    Frameless by design: N stacked axes means N copies of every spine, tick and
    label, and none of that repetition carries information the stack doesn't
    already establish once. What replaces it is a dashed zero line on every row
    (the baseline each curve is read against) and a single y scale bar standing
    on that zero (see `_y_scale_bar`). x keeps real tick labels on the bottom
    row only, because p-value is a bounded absolute coordinate where position
    -- not just scale -- is meaningful; its spine goes too.
    """
    buckets = _drop_empty(buckets)
    if not buckets:
        return []
    # One flat, fully dark colour for every bucket -- see ECDF_SHADE.
    colors = [matplotlib.colormaps[cmap](ECDF_SHADE)] * len(buckets)
    inner = cell.subgridspec(len(buckets), 1, hspace=0.0)
    bar_row = len(buckets) // 2
    fs = FP.FS_LEGEND * 0.7

    axs = []
    for i, ((label, g), color) in enumerate(zip(buckets, colors)):
        ax = fig.add_subplot(inner[i], sharex=axs[0] if axs else None,
                             sharey=axs[0] if axs else None)
        axs.append(ax)
        dev = _ecdf_deviation(g[pval_col].values)
        ax.axhline(0, color=FP.COLOR_NULL, linestyle="--",
                   linewidth=FP.LW_REFERENCE, alpha=0.7)
        if dev is not None:
            x, d, n = dev
            # The null's 95% binomial band, as in Figs 1 and 2 (was a
            # bootstrap band around the curve until 2026-10-19).
            statistics.plot_binomial_ecdf_band(ax, n)
            ax.plot(x, d, color=color, linewidth=FP.LW_TRACE)
        # Bucket label sits OUTSIDE the axes, on the right. Inside, it landed
        # on top of the curve for any bucket with a real deviation -- exactly
        # the buckets worth reading.
        ax.text(1.02, 0.5, f"{label} ({len(g)})", transform=ax.transAxes,
                ha="left", va="center", fontsize=fs, color=color)
        ax.set_xlim(0, 1)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(left=False, labelleft=False, labelsize=fs, pad=1)
        if i < len(buckets) - 1:
            ax.tick_params(bottom=False, labelbottom=False)

    # Three x ticks, not matplotlib's six: floating tick marks with no spine
    # under them read as clutter past about three.
    axs[-1].set_xticks([0, 0.5, 1])
    if xlabel:
        axs[-1].set_xlabel("p-value", fontsize=FP.FS_LEGEND, labelpad=1)

    # No scale bar here: the caller draws it (`_add_stack_scale_bar`) once
    # the y range is final.
    if ylabel:
        # Sits outboard of the scale bar, and is the only y annotation on the
        # stack besides the bar's own two numbers. Anchored to one mini-axes
        # but free to overflow it: rotated text costs VERTICAL space, of which
        # a 10-to-18-row stack has plenty even though any single row is ~0.3in,
        # so the label can spell the quantity out in full rather than
        # abbreviate to fit one row's height.
        axs[bar_row].set_ylabel(ylabel, fontsize=FP.FS_LEGEND, labelpad=24)
    return axs


def _apply_percentile(buckets, percentile, sens_col):
    """Keep only each bucket's most sensitive `100 - percentile`% of neurons.

    Deliberately per-bucket, matching `fig3.plot_uniform_p`'s own handling of
    its `percentile` argument, so the ECDF stack and the sorted-q-value panel
    below it show the same neurons. The top-10% quadrant originally
    passed `percentile` to `plot_uniform_p` only, which left its ECDF stack
    showing the whole unfiltered population -- a quadrant whose two halves
    disagreed about which neurons it was about.
    """
    if percentile is None:
        return buckets
    out = []
    for label, g in buckets:
        if len(g) == 0:
            out.append((label, g))
            continue
        cut = np.percentile(g[sens_col], percentile)
        out.append((label, g.loc[g[sens_col] > cut]))
    return out


def _plot_ecdf_quadrant(fig, cell, buckets, title, letter,
                        pval_col="p_value", sens_col="sens"):
    """Per-bucket ECDF-deviation stack on top, the sorted-q-value panel below.
    Height ratio favours the top row because it holds up to ~10 stacked
    mini-axes against the q-row's single axes.

    Buckets arrive final: every filter, including the sensitivity percentile,
    has already been applied by `build_ecdf_page` (where the ordering of
    filtering vs. bucketing depends on the bucketing). The stack and the
    q-value panel below it therefore always show the same units.
    """
    inner = cell.subgridspec(2, 1, hspace=0.12, height_ratios=[3.6, 1])

    axs = _plot_ecdf_dev_stack(fig, inner[0], buckets, "Blues",
                               ylabel="ECDF deviation from uniform null",
                               pval_col=pval_col)
    _add_stack_scale_bar(axs)

    ax_q = fig.add_subplot(inner[1])
    groups = [g for _, g in _drop_empty(buckets)]
    fig3.plot_uniform_p(groups, [None, ax_q],
                        colors=_shades("Blues", len(groups)),
                        pval_col=pval_col, sens_col=sens_col)
    ax_q.set_ylabel("Sorted q-values")
    ax_q.set_xlabel("Unit")

    _quadrant_header(fig, cell, title, letter, dy=0.012)


def fig3_conditions():
    """The canonical figure's four quadrant definitions, kept in one place so
    the ECDF page cannot drift from the canonical figure.
    """
    return [
        ("All neurons across all species", None, None),
        ("All pigeon neurons", lambda df: df.species == "Pigeon", None),
        ("Pigeon hippocampus neurons", lambda df: (df.area == "HP") & (df.species == "Pigeon"), None),
        ("Top 10% most sensitive neurons in pigeon HP",
         lambda df: (df.area == "HP") & (df.species == "Pigeon"), 90),
    ]


def build_ecdf_page(all_fourier_df, bucket_fn=None, suptitle=None,
                    pval_col="p_value", sens_col="sens"):
    """The canonical figure's quadrants, drawn as per-bucket ECDF-deviation
    mini-axes. The page is taller than the others because each quadrant
    stacks one mini-axes per bucket.

    `bucket_fn=None` buckets by occurrence wave (no current page). Any other bucketing
    is passed as a callable (page 4 passes `bucket_round_robin`).

    The two differ in where the sensitivity percentile has to be applied,
    which is why this isn't one code path:

    * Occurrence waves are computed once over the whole population and then
      narrowed per quadrant, so the percentile runs per bucket AFTER
      bucketing -- matching what the canonical figure does.
    * Any bucket_fn is handed an already-narrowed population, so the
      percentile has to run BEFORE bucketing. It must, for the round-robin
      case: bucketing first and filtering second would cut each equal-sized
      bucket down by a different amount and throw away the equal sizes that
      are the entire point of that page.

    Both orderings keep the same total unit count; the former applies B
    separate sensitivity thresholds, the latter one global threshold.
    """
    if bucket_fn is None:
        waves, _, _ = fig3.split_into_occurrence_waves(
            all_fourier_df, pval_col=pval_col)
        base = [(f"#{i + 1}", w) for i, w in enumerate(waves)]
    else:
        base = _magnetic_frame(all_fourier_df)

    fig = plt.figure(figsize=(FP.FIGSIZE_FIG3[0] * 1.15, FP.FIGSIZE_FIG3[1] * 2.6))
    # `right` stops short of the page edge to leave room for the outside
    # bucket labels on each stack's right-hand side.
    outer = fig.add_gridspec(2, 2, wspace=0.5, hspace=0.16,
                             left=0.1, right=0.88, top=0.92, bottom=0.05)

    def _quadrant_buckets(base, cond_filter, percentile):
        if bucket_fn is None:
            buckets = base if cond_filter is None else \
                [(lab, g.loc[cond_filter(g)]) for lab, g in base]
            return _apply_percentile(buckets, percentile, sens_col)
        return bucket_fn(_condition_subset(base, cond_filter, percentile, sens_col))

    for (title, cond_filter, percentile), (orow, ocol), letter in zip(
            fig3_conditions(), QUADRANT_SLOTS, "ABCD"):
        _plot_ecdf_quadrant(fig, outer[orow, ocol],
                            _quadrant_buckets(base, cond_filter, percentile),
                            title, letter, pval_col=pval_col, sens_col=sens_col)

    if suptitle is None:
        suptitle = ("Canonical buckets, drawn as per-bucket ECDF deviation "
                    "(gray: null 95% binomial band; dashed line = uniform null)")
    fig.suptitle(suptitle, fontsize=FP.FS_BODY + 2, fontweight="bold", y=0.995)
    return fig


# -- Occurrence bootstrap, single bucket (page 11) ---------------------------

def _occurrence_bootstrap_curves(df, pval_col="p_value", n_boot=N_BOOT, seed=0,
                                 alpha=0.05):
    """The occurrence bootstrap (`statistics.iter_occurrence_draws`,
    one randomly chosen recording per neuron per draw), summarised two ways
    from the SAME draws:

    * ECDF(p) - p on a fixed p grid -- identical to
      `statistics.bootstrap_occurrence_ecdf_band`, as Fig 2 C/D used until 2026-10-19;
    * Storey q-values (lambda 0.5, as in `fig3.plot_uniform_p`), sorted, per
      rank -- every draw has exactly one value per neuron, so rank k is
      comparable across draws.

    Returns None if there is nothing to draw, else a dict of the grid, the
    median/lower/upper ECDF deviation, the median/lower/upper sorted q-values,
    the neuron count and the recording count (rows with a finite p-value,
    which is all the bootstrap draws from). The seed matches Fig 2's (0).
    """
    n_rec = int(np.isfinite(pd.to_numeric(df[pval_col], errors="coerce")).sum())
    if n_rec == 0:
        return None
    x = statistics.ECDF_P_GRID
    devs, qs = np.empty((n_boot, len(x))), None
    for i, picked in enumerate(statistics.iter_occurrence_draws(
            df, value_col=pval_col, group_cols=NEURON_KEY, n_boot=n_boot, seed=seed)):
        picked = np.sort(picked)
        if qs is None:
            qs = np.empty((n_boot, len(picked)))
        devs[i] = statistics.ecdf_minus_uniform(picked, x)
        # `picked` is sorted, so its q-values already come out in rank order.
        qs[i] = statistics.storey_qvalues(picked, lambda_=0.5)[0]
    pct = [100 * alpha / 2, 50, 100 * (1 - alpha / 2)]
    dev_lo, dev_med, dev_hi = np.percentile(devs, pct, axis=0)
    q_lo, q_med, q_hi = np.percentile(qs, pct, axis=0)
    return dict(x=x, dev_med=dev_med, dev_lo=dev_lo, dev_hi=dev_hi,
                q_med=q_med, q_lo=q_lo, q_hi=q_hi, n_neurons=len(q_med),
                n_recordings=n_rec)


def _plot_occurrence_quadrant(fig, cell, df, title, letter,
                              pval_col="p_value", n_boot=N_BOOT):
    """ECDF deviation on top, sorted q-values below, for one all-observations
    bucket, drawn from `_occurrence_bootstrap_curves`."""
    inner = cell.subgridspec(2, 1, hspace=0.75)
    ax_e = fig.add_subplot(inner[0])
    ax_q = fig.add_subplot(inner[1])
    color = FP.COLOR_MAG

    curves = _occurrence_bootstrap_curves(df, pval_col=pval_col, n_boot=n_boot)
    if curves is None:
        for ax in (ax_e, ax_q):
            ax.text(0.5, 0.5, "no units", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FP.FS_LEGEND,
                    color=FP.COLOR_NULL)
    else:
        x = curves["x"]
        ax_e.fill_between(x, curves["dev_lo"], curves["dev_hi"], color=color,
                          alpha=FP.ALPHA_CONFIDENCE, linewidth=0)
        ax_e.plot(x, curves["dev_med"], color=color, linewidth=FP.LW_TRACE)
        ax_e.axhline(0, color=FP.COLOR_NULL, linestyle="--",
                     linewidth=FP.LW_REFERENCE, alpha=0.6)
        ax_e.set_xlim(0, 1)
        ax_e.set_xticks([0, 0.5, 1])
        ax_e.set_xlabel("p-value")

        n = curves["n_neurons"]
        rank = np.arange(n)
        ax_q.fill_between(rank, curves["q_lo"], curves["q_hi"], color=color,
                          alpha=FP.ALPHA_CONFIDENCE, linewidth=0, rasterized=True)
        ax_q.plot(rank, curves["q_med"], color=color, linewidth=FP.LW_TRACE)
        ax_q.set_yscale("log")
        ax_q.set_ylim(*fig3.Q_YLIM)
        ax_q.set_xticks([0, n])
        ax_q.set_xticklabels([0, n])
        ax_q.set_xlabel("Unit")
        # Responder count of the median curve, as in every other Fig 3
        # q-value panel.
        fig3.draw_fdr_threshold(ax_q, curves["q_med"], color)

        ax_e.set_title(f"{n} neurons, {curves['n_recordings']} recordings",
                       fontsize=FP.FS_TITLE)

    ax_e.set_ylabel("ECDF - uniform")
    ax_q.set_ylabel("Sorted q-values")
    _quadrant_header(fig, cell, title, letter, dy=0.05)


def build_occurrence_bootstrap_page(all_fourier_df, pval_col="p_value",
                                    sens_col="sens", n_boot=N_BOOT):
    """Page 11: canonical quadrants, one bucket each, an occurrence bootstrap
    (see `_occurrence_bootstrap_curves`).

    Every observation stays in (no dedup) -- the bootstrap is what keeps each
    neuron counted once per draw. The top-10% quadrant therefore takes one
    global sensitivity threshold over all its rows, as page 3 does.
    """
    neg = _magnetic_frame(all_fourier_df)

    fig = plt.figure(figsize=(FP.FIGSIZE_FIG3[0] * 1.0, FP.FIGSIZE_FIG3[1] * 1.9))
    outer = fig.add_gridspec(2, 2, wspace=0.45, hspace=0.55,
                             left=0.12, right=0.98, top=0.82, bottom=0.07)
    for (title, cond_filter, percentile), (orow, ocol), letter in zip(
            fig3_conditions(), QUADRANT_SLOTS, "ABCD"):
        sub = _condition_subset(neg, cond_filter, percentile, sens_col)
        _plot_occurrence_quadrant(fig, outer[orow, ocol], sub,
                                  title, letter, pval_col=pval_col, n_boot=n_boot)

    fig.suptitle("Canonical quadrants, one bucket of every recording,\n"
                 "occurrence bootstrap (one random recording per neuron per draw; "
                 "median and 95% band)",
                 fontsize=FP.FS_BODY + 2, fontweight="bold", y=0.975)
    return fig


# -- Driver -------------------------------------------------------------------

def build_pages(all_fourier_df, pages, out_path: Path, n_boot=N_BOOT):
    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY}
    matplotlib.rc("font", **font)

    with PdfPages(out_path) as pdf:
        for page in pages:
            print(f"  page {page} ...")
            if page == 1:
                fig = build_bucket_page(
                    all_fourier_df,
                    lambda df: bucket_by_frequency(df, split_repeats=True),
                    "Species x stimulus frequency "
                    "(repeat unit x frequency rows split into per-frequency 'repeats')")
            elif page == 2:
                fig = build_bucket_page(
                    all_fourier_df, bucket_by_area,
                    "Species x brain region "
                    "(repeat unit x region rows dropped after the first)")
            elif page == 3:
                fig = build_condition_bucket_page(
                    all_fourier_df, bucket_round_robin,
                    "Canonical quadrants, equal-sized buckets "
                    "(units dealt round-robin, one recording per unit per bucket)",
                    legend=False)
            elif page == 4:
                fig = build_ecdf_page(
                    all_fourier_df, bucket_fn=bucket_round_robin,
                    suptitle="Equal-sized buckets, drawn as per-bucket ECDF "
                             "deviation (gray: null 95% binomial band; dashed "
                             "line = uniform null)")
            elif page == 5:
                fig = build_grid_bucket_page(
                    all_fourier_df, species_area_panels(all_fourier_df),
                    bucket_by_frequency,
                    "Species x brain area x frequency -- every "
                    "(species, area) pair, bucketed by frequency",
                    ncols=SPECIES_AREA_NCOLS)
            elif page == 6:
                fig = build_grid_bucket_page(
                    all_fourier_df, area_panels(all_fourier_df, "Pigeon"),
                    bucket_by_area,
                    "Pigeon, one panel per brain area "
                    "(each area pooled into a single bucket)",
                    # 2-wide, not 3: pigeon has exactly 3 areas, and one row of
                    # 3 makes a 16x4in strip rather than a page.
                    ncols=2, legend=False)
            elif page == 7:
                fig = build_grid_bucket_page(
                    all_fourier_df, species_area_panels(all_fourier_df),
                    bucket_by_area,
                    "All species, one panel per brain area "
                    "(each area pooled into a single bucket)",
                    ncols=SPECIES_AREA_NCOLS, legend=False)
            elif page == 8:
                fig = build_grid_bucket_page(
                    all_fourier_df,
                    [("Pigeon", lambda d: d.species == "Pigeon")],
                    bucket_by_area,
                    "Pigeon, brain areas as buckets on shared axes",
                    ncols=1, scale=2.2)
            elif page == 9:
                fig = build_grid_bucket_page(
                    all_fourier_df, [("All species", None)],
                    bucket_by_species_area,
                    "All species, (species, brain area) pairs as buckets "
                    "on shared axes",
                    ncols=1, scale=2.2)
            elif page == 10:
                fig = build_grid_bucket_page(
                    all_fourier_df, [("All species", None)],
                    bucket_by_species,
                    "Species as buckets on shared axes "
                    "(units appearing more than once dropped after the first)",
                    ncols=1, scale=2.2)
            elif page == 11:
                fig = build_occurrence_bootstrap_page(all_fourier_df, n_boot=n_boot)
            else:
                raise ValueError(f"unknown page {page}")
            # No top/right frame on any axes, insets included.
            for ax in fig.axes:
                for a in [ax, *ax.child_axes]:
                    a.spines[["top", "right"]].set_visible(False)
            # Page id at the bottom (equal to the PDF page with the default --pages).
            fig.text(0.5, -0.02, str(page), ha="center", va="top",
                     fontsize=FP.FS_TITLE)
            pdf.savefig(fig, bbox_inches="tight", dpi=FP.DPI)
            if not in_notebook:
                plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate the multipage Fig 3 variants PDF")
    parser.add_argument("--out-dir", default=FP.DIAGNOSTIC_DIR, help="Output directory")
    parser.add_argument("--out-name", default="Fig3S4_bucketings.pdf")
    parser.add_argument("--parquet", default=FP.PARQUET_PATH,
                        help=f"Path to all_fourier_df.parquet (default: {FP.PARQUET_PATH})")
    parser.add_argument("--pages", type=int, nargs="+",
                        default=list(range(1, 12)),
                        help="Which pages to render, in order (default: 1-11)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help="Bootstrap replicates for the ECDF CI bands (page 11)")
    args = parser.parse_args([] if in_notebook else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.parquet} ...")
    all_fourier_df = pd.read_parquet(args.parquet)
    build_pages(all_fourier_df, args.pages, out_dir / args.out_name, n_boot=args.n_boot)


if __name__ == "__main__":
    if in_notebook:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    main()
