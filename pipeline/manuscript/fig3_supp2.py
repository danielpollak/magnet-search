"""Fig 3 supplement 2 -- q-value reproducibility across occurrence-waves
("recordings"), split by species.

Ports the analysis from the archived pipeline's
`MagnetSearch/code/NWB/DP_bayes_mixed_model.ipynb` (cells 23-28) onto the
current NWB-based pipeline. Backs the manuscript claim: "we collected all the
neurons that emerged as suspects in one recording, and asked whether they
also appeared to be modulated in a second independent recording ... If
responses to the magnetic stimulus exist, then this pool should be highly
enriched in such responsive neurons. Figure~\\ref{fig:q-values} shows that
there is no evidence for that."

For each species, every neuron's q-value (Storey-corrected, computed
*within that species* -- unlike fig3.py's per-wave q-values, which pool
across all species in a wave, and see QVAL_SCOPE_HELP for which neurons each
correction runs over) is tabulated across its occurrence-waves
("buckets": 1st recording, 2nd, 3rd, ...) into a wide table (one row per
neuron, one column per bucket), then plotted as a hand-rolled grid of
scatterplots (see `_plot_triangle_grid`): each panel scatters one bucket's
q-value against another's, for every pair of buckets, in the strictly lower
triangle for the negative-result (magnetic) population and the strictly
upper triangle for the positive-control (visual/audio/oddball/...)
population -- two independent analyses sharing one page per species, since
each is a completely separate Storey correction and completely separate set
of neurons (see `build_wave_pivot`'s `population` argument). No correlation /
no enrichment near q=0 in the second bucket among suspects from the first
bucket is the (correctly) expected null result for the magnetic population;
the visual/audio population is shown for contrast and is expected to show
real enrichment.

Species/bucket combinations are only plotted if there are at least
`MIN_WAVES` occurrence-buckets with at least `MIN_NEURONS_PER_WAVE` neurons
each -- this is a genuine data-availability filter, not a stylistic choice:
under the current NWB pipeline's annotations, some species that had multiple
usable waves in the old CSV-era export no longer do (e.g. zebrafish's `ID`
column is "unknown" for every session -- each fish is annotated per-`date`
with no cross-session identity, so `_split_into_waves`'s
`(species, ID, date, id)` grouping never detects a repeat), and Owl has no
`p_value` at all (it's a precomputed-Bayesian-coefficient species, never
run through the Fourier p-value pipeline -- see CLAUDE.md's aggregate.py
species-coverage notes). Reproducing the *old* hardcoded species list
(Pigeon / zebra finch / zebrafish) verbatim would silently produce an empty
or degenerate zebrafish page under current data; selecting species
dynamically means this script keeps working (and stays honest about what it
can show) as annotations evolve. Species/waves dropped by the filter are
logged, not silently omitted.

Two versions of the same grid are written. `Fig3_supp2.pdf` scatters one dot
per neuron; `Fig3_supp2_density.pdf` replaces each panel's dots with a 2D
histogram of the same points (see the DENSITY_* constants). The density
version exists because q-values pile up hard against q=1 -- exactly where the
scatter overplots itself into a solid block -- so the dots alone can't say
whether a cluster is symmetric or how much of a panel sits in its densest
corner. Both carry the same annotations: per-bucket neuron totals under each
x label, and each panel's pairwise-complete count plus its two pi0 estimates
in that panel's title.

Which recording ends up in which bucket is controlled by `--bucket-order`; the
default randomizes it, so that a bucket is a random one of each neuron's own
recordings rather than a stand-in for one particular recording. See
BUCKET_ORDER_HELP -- this matters a great deal for how a panel should be read.

Requires:
  data/manuscript/all_fourier_df.parquet  (run python pipeline/aggregate.py first)

Usage:
    python pipeline/manuscript/fig3_supp2.py
    python pipeline/manuscript/fig3_supp2.py --mode density
    python pipeline/manuscript/fig3_supp2.py --qval-scope bucket
    python pipeline/manuscript/fig3_supp2.py --bucket-order recording
    python pipeline/manuscript/fig3_supp2.py --out-dir figs/paper
"""
import argparse
from collections import Counter
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd

from magpyneto2 import statistics
from fig3 import _split_into_waves

import format_parameters as FP


WAVE_GROUPS = ["species", "ID", "date", "id"]

# Minimum neurons in a given occurrence-wave, and minimum number of such
# waves, for a species to get its own pairplot page -- see module docstring.
MIN_NEURONS_PER_WAVE = 20
MIN_WAVES = 2

QVAL_SCOPES = ("intersection", "bucket")

BUCKET_ORDERS = ("random", "recording")
RANDOM_SEED = 0

BUCKET_ORDER_HELP = """\
How a neuron's rows are assigned to buckets.

"random" (default): the rows of each population are permuted before
`_split_into_waves` counts them off, which gives every neuron a uniformly
random ordering of its own recordings. Bucket k is then "a random one of this
neuron's recordings", and the same recording lands in different buckets for
different neurons.

"recording": the original behaviour -- count off in parquet row order, so a
neuron's bucket index is the position of that recording within its own session.
This makes bucket index a proxy for recording identity, which is what the
figure's asymmetry was tracking rather than anything about neurons. On Pigeon's
positive-control page the recordings sort alphabetically by orientation suffix,
so bucket 2 was "the 180 degree grating", bucket 3 "225 degrees", and so on out
to bucket 17 = "WN_SameCh"; a panel then compared two specific and often quite
different stimuli (one panel pits a 3 Hz grating against a white-noise block,
a ~5x difference in spike count and hence in detection power) instead of two
interchangeable repeats. Under "random" the two axes of a panel are
statistically exchangeable, so a symmetric cloud is the correct null
expectation and a departure from symmetry means something.

Bucket SIZES are identical either way -- a neuron with k rows occupies buckets
0..k-1 whatever order they are in -- so only each bucket's recording
composition changes, not the per-bucket or per-panel n."""

QVAL_SCOPE_HELP = """\
Which set of neurons each axis' Storey correction runs over.

"intersection" (default): exactly the neurons drawn in that panel, i.e. the
pairwise-complete set present in BOTH of its buckets. Each axis is corrected
separately over that set.

"bucket": the original behaviour -- correct once over every neuron in the
bucket, then subset to the panel. This makes a neuron's q-value a property of
its column alone (identical everywhere it appears in the grid), which reads
nicely across a row, but compares apples to oranges: the correction population
is not the plotted population. A Storey q-value is a rescaling of a p-value by
that population's own pi0 (and pi0 is a hard ceiling on q, since
q_max = pi0 * p_max), so two axes calibrated on two different populations are
not on a common scale, and neither is calibrated on the points you can see.
On Pigeon's positive-control page the gap is large: bucket 0's pi0 is 0.845
over all 3005 of its neurons but 0.965 over the 1167 it shares with bucket 6,
because bucket 0's signal sits disproportionately in the oddball-session units
that drop out of that pairing. Under "bucket" that panel looks symmetric
(46.9% of points above the diagonal) purely because two opposite biases cancel;
under "intersection" its real asymmetry (11.1%) is visible.

The cost of "intersection" is that a neuron's q-value now varies from panel to
panel, so only within-panel comparisons are meaningful -- which is the only
comparison these panels are actually making."""


def build_wave_pivot(all_fourier_df, pval_col="p_value", population="neg",
                     bucket_order="random", seed=RANDOM_SEED):
    """Raw p-values pivoted into one row per neuron (`WAVE_GROUPS` key) and one
    column per occurrence-wave (a "bucket").

    `bucket_order` selects how rows are counted off into buckets; see
    BUCKET_ORDER_HELP. `seed` only matters for `bucket_order="random"`.

    Deliberately stops short of the Storey correction: which neurons a q-value
    should be corrected over depends on the panel it is drawn in, not on the
    bucket alone, so the correction happens per-panel in `_panel_qvals` (see
    QVAL_SCOPE_HELP). What is pooled is still never more than one species at a
    time -- unlike `fig3.split_into_occurrence_waves`, which pools every
    species together within a wave (appropriate for the main Fig3 panels'
    population-level uniformity check). The question here is neuron-level
    reproducibility across recordings of the same individual, so mixing species
    into one Storey correction would let one species' p-value distribution
    distort another's q-values for no reason connected to the actual comparison
    being made; the per-species split is enforced by the callers, which slice
    this pivot by species (`_species_subset`) before correcting.

    `population`: "neg" for the negative-result (magnetic) population, "pos"
    for the positive-control (visual/audio/oddball/...) population -- see
    `statistics.get_poscontrols_negresults`. The two are never pooled into one
    Storey correction (same reasoning as the species split above -- see
    fig3.py's `plot_uniform_p` docstring on why occurrence-wave index isn't
    comparable across populations either).
    """
    all_neg_res, all_pos_control, _ = statistics.get_poscontrols_negresults(all_fourier_df)
    source_df = all_neg_res if population == "neg" else all_pos_control
    if bucket_order == "random":
        # One global permutation is enough: restricted to any single neuron's
        # rows it is a uniformly random ordering of exactly those rows, which
        # is what `_split_into_waves`'s per-neuron cumcount then counts off.
        source_df = source_df.sample(frac=1.0, random_state=seed)
    waves = _split_into_waves(source_df, wave_groups=WAVE_GROUPS)

    wave_df_l = [w.loc[w[pval_col].notna()] for w in waves]
    wave_df_l = [w for w in wave_df_l if len(w)]
    if not wave_df_l:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[]] * len(WAVE_GROUPS), names=WAVE_GROUPS))
    all_wave_df = pd.concat(wave_df_l, ignore_index=True)
    return all_wave_df.pivot(index=WAVE_GROUPS, columns="occurrence", values=pval_col)


def _storey(vals):
    """`statistics.storey_qvalues` on `vals`, tolerating an empty input (it
    indexes `order[-1]` unconditionally and so would raise). Returns
    `(qvals, pi0)`."""
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return vals, np.nan
    return statistics.storey_qvalues(vals, lambda_=0.5)


def _bucket_qvals(species_p_df, col, cache):
    """`(qvals_series, pi0)` for every neuron in bucket `col`, corrected over
    the whole bucket. Cached in `cache`, since each bucket appears in many
    panels and the correction is identical in all of them."""
    if col not in cache:
        p = species_p_df[col].dropna()
        q, pi0 = _storey(p.values)
        cache[col] = (pd.Series(q, index=p.index), pi0)
    return cache[col]


def _panel_qvals(species_p_df, col, row, scope, cache):
    """`(qvals_df, pi0_x, pi0_y)` for panel (`col` on x, `row` on y).

    `qvals_df` is indexed by the pairwise-complete neurons -- the only ones
    that can be plotted at all -- with one column per axis. `scope` selects
    which population each axis' Storey correction runs over; see
    QVAL_SCOPE_HELP.
    """
    both = species_p_df[[col, row]].dropna()
    if scope == "intersection":
        (qx, pi0x), (qy, pi0y) = _storey(both[col].values), _storey(both[row].values)
        return pd.DataFrame({col: qx, row: qy}, index=both.index), pi0x, pi0y
    qx, pi0x = _bucket_qvals(species_p_df, col, cache)
    qy, pi0y = _bucket_qvals(species_p_df, row, cache)
    return pd.DataFrame({col: qx.loc[both.index], row: qy.loc[both.index]}), pi0x, pi0y


def _species_subset(pivot_df, species):
    """Rows of `pivot_df` for `species`, or an empty (but column-compatible)
    frame if that species has no rows in this population at all -- lets the
    caller treat "no visual data for this species" and "visual data present
    but too few buckets" the same way.
    """
    if "species" not in pivot_df.index.names or species not in pivot_df.index.get_level_values("species"):
        return pivot_df.iloc[0:0]
    return pivot_df.xs(species, level="species", drop_level=False)


def _valid_cols(species_df):
    """Occurrence-buckets with >= MIN_NEURONS_PER_WAVE neurons, plus the raw
    per-bucket counts (for logging skipped species/buckets)."""
    counts = species_df.notna().sum()
    return sorted(c for c in counts.index if counts[c] >= MIN_NEURONS_PER_WAVE), counts


LABEL_FONTSIZE = FP.FS_BODY * 2.5  # bucket axis labels -- 2-3x body size, for readability
TICK_FONTSIZE = FP.FS_BODY * 2     # "0.0"/"1.0" tick labels
COUNT_FONTSIZE = LABEL_FONTSIZE * 0.7   # per-bucket neuron counts, under each "bucket N"
PANEL_TITLE_FONTSIZE = LABEL_FONTSIZE * 0.6  # per-panel "n=... pi0 .../...%" title
PANEL_TITLE_PAD_PT = 4.0
# Vertical space one panel title occupies above its axes. Reserved in the top
# margin, and added to the offset of any bucket label drawn along the TOP of
# the grid so the label clears the title underneath it.
TITLE_BLOCK_PT = PANEL_TITLE_FONTSIZE * 1.32 + PANEL_TITLE_PAD_PT

# Distance (in points) from an axes' bottom/left edge out past its tick labels.
# The bucket labels are drawn as figure-level text rather than with
# ax.set_xlabel/set_ylabel (see the label-alignment note below), so their
# offset from the axes has to be specified rather than handled by matplotlib.
TICK_PAD_PT = TICK_FONTSIZE * 1.9
# Same, for a column labelled along the top of the grid (no tick labels to
# clear there), plus the baseline-to-baseline spacing of a stacked label block.
TOP_LABEL_PAD_PT = 14.0
LINE_SPACING = 1.32

# Temporary diagnostic (2026-09-02): the number of neurons actually plotted in
# each panel -- i.e. present in BOTH of that panel's buckets, which is very
# much smaller than either bucket's own total. Together with the per-bucket
# totals under each x label, this is what makes "why does this panel have so
# many more outliers / such an asymmetric cluster than that one" answerable by
# eye instead of by guesswork.
SHOW_PANEL_N = True

# Each axis' estimated pi0, as a percentage. Worth showing because pi0 is a
# hard ceiling on that axis' q-values (q_max = pi0 * p_max), so a panel whose
# two pi0 values differ is one whose two axes stop at different heights -- the
# single most common reason a cluster looks lopsided about the diagonal. Under
# QVAL_SCOPE "intersection" these vary panel to panel, which is exactly the
# information the old bucket-wide correction hid. Ordered x/y, matching the
# axes.
SHOW_PANEL_PI0 = True

# A column's "bucket N" label normally sits under that column's bottom-most
# occupied cell. Because the two populations don't reach the same edge of the
# grid, a column can bottom out one row early and leave a single label floating
# above the rest of the line (Pigeon capped to 11 buckets: every other column's
# lowest cell is the magnetic one on row 9, but column 9 has no magnetic cell
# at all -- that would need row 10, which isn't a magnetic bucket -- so its
# lowest cell is the visual one on row 8). Columns within this many rows of the
# grid's main label line are pulled down onto it. Ones further away keep their
# own line: on the uncapped Pigeon page the visual-only columns 10-17 form a
# staircase whose labels are up to 7 rows above the main line, and dropping
# those to the bottom would strand them in whitespace far from their data.
LABEL_ALIGN_MAX_GAP = 1

# -- Density mode -------------------------------------------------------------
# q-values live on [0, 1], so every panel's 2D histogram shares fixed bin edges.
# Counts are converted to a fraction of that panel's own points before
# colouring, since panel n ranges over two orders of magnitude (Pigeon's
# thousands vs. mouse's tens) and raw counts would render the small panels
# uniformly blank rather than comparable. The norm is logarithmic because the
# q~1 pile-up holds a large fraction of every panel on its own -- on a linear
# norm everything else is one flat colour.
DENSITY_BINS = 25
DENSITY_VMIN = 1e-4
# Blues/Oranges echo FP.COLOR_MAG (steelblue) / FP.COLOR_VIS (coral), the same
# per-population sequential families fig3.py uses for its occurrence shades.
# Truncated at the pale end so the sparsest occupied bin is still clearly
# distinguishable from an empty one (empty bins are masked -> set_bad white).
DENSITY_CMAPS = {"neg": "Blues", "pos": "Oranges"}
DENSITY_CMAP_FLOOR = 0.18


def _density_cmap(name):
    """`name` truncated to start at DENSITY_CMAP_FLOOR, with empty (masked)
    bins rendered white -- see DENSITY_CMAPS."""
    cmap = LinearSegmentedColormap.from_list(
        f"{name}_truncated",
        matplotlib.colormaps[name](np.linspace(DENSITY_CMAP_FLOOR, 1.0, 256)))
    cmap.set_bad("white")
    return cmap


def _modal_line(indices, outward):
    """The most frequent value in `indices`, ties broken by `outward` (max for
    a bottom edge, min for a left edge)."""
    counts = Counter(indices)
    best = max(counts.values())
    return outward(i for i, c in counts.items() if c == best)


def _offset_trans(fig, dx_pt, dy_pt):
    """Figure-fraction transform shifted by a fixed offset in points."""
    return fig.transFigure + ScaledTranslation(dx_pt / 72, dy_pt / 72, fig.dpi_scale_trans)


def _plot_triangle_grid(plot_df_neg, plot_df_pos, cols_neg, cols_pos, title, mode="scatter",
                        qval_scope="intersection"):
    """Full NxN grid of bucket-vs-bucket q-value panels: the strictly lower
    triangle (row > col) shows the negative-result (magnetic) population, the
    strictly upper triangle (row < col) shows the positive-control
    (visual/audio/...) population, and the diagonal is left blank --
    deliberately NOT `sns.pairplot`, for the same "no wasted blank diagonal
    stripe" reason as the original single-triangle version of this plot.

    `plot_df_neg`/`plot_df_pos` hold raw p-values (see `build_wave_pivot`);
    each panel's q-values are computed here, from the population `qval_scope`
    selects -- see QVAL_SCOPE_HELP.

    `mode` is "scatter" (one dot per neuron) or "density" (a per-panel 2D
    histogram, see the DENSITY_* constants) -- the same grid, same cells, same
    populations, differing only in how each panel's points are drawn. Dots
    overplot heavily near q=1, which is exactly where these distributions pile
    up, so the scatter version systematically understates how lopsided that
    pile-up is; the density version is the one to read for that.

    `cols_neg`/`cols_pos` are each population's own set of valid occurrence-
    buckets (see MIN_NEURONS_PER_WAVE/MIN_WAVES) -- they need not be the same
    length or contain the same values, since "how many usable magnetic
    recordings a species has" and "how many usable visual/audio recordings it
    has" are independent facts. A bucket pair missing from a population's
    valid set is simply left blank on that population's side of the grid
    (no axes are drawn there at all, same treatment as the diagonal) rather
    than forcing the two triangles to share one bucket count.

    Returns `None` if neither triangle actually has anything to plot.
    """
    cols_neg, cols_pos = set(cols_neg), set(cols_pos)
    n = max(max(cols_neg, default=-1), max(cols_pos, default=-1)) + 1
    if n < 2:
        return None

    # Every (row, col) cell that will actually hold a panel -- used below to
    # decide which cell in each column/row carries the tick labels, and where
    # that column's/row's bucket label goes.
    cells = [(row, col) for row in range(n) for col in range(n)
             if row != col and
             ((row > col and row in cols_neg and col in cols_neg) or
              (row < col and row in cols_pos and col in cols_pos))]
    if not cells:
        return None
    bottom_row_for_col = {c: max(r for r, cc in cells if cc == c) for c in {cc for _, cc in cells}}
    top_row_for_col = {c: min(r for r, cc in cells if cc == c) for c in {cc for _, cc in cells}}
    left_col_for_row = {r: min(c for rr, c in cells if rr == r) for r in {rr for rr, _ in cells}}
    right_col_for_row = {r: max(c for rr, c in cells if rr == r) for r in {rr for rr, _ in cells}}
    # The grid's main label lines: the row / column that the most panels
    # already bottom out on / start from, ties broken outwards. NOT the
    # outermost such row/column -- where the upper triangle runs past the lower
    # one, its extra buckets form a staircase hugging the diagonal whose tip
    # reaches further down and further left than the main block does, and
    # taking the extreme would drag the whole label line out there behind a
    # single column. See LABEL_ALIGN_MAX_GAP.
    x_label_row = _modal_line(bottom_row_for_col.values(), outward=max)
    y_label_col = _modal_line(left_col_for_row.values(), outward=min)
    # Columns that can't join the main label line are labelled along the TOP of
    # the grid instead of under their own bottom-most cell. A label may only be
    # pushed DOWN onto the line (the rows below a column's last panel are empty
    # by construction, so there is nothing there to collide with) and only by
    # LABEL_ALIGN_MAX_GAP; a column reaching *past* the line can never be
    # pulled up, since the line then falls in the middle of its own panels.
    # Both exclusions pick out the same thing in practice: the staircase of
    # buckets that only one population has, hugging the diagonal, where the
    # space under a cell is already taken by the rotated "bucket N" of the row
    # diagonally below it.
    top_labelled = {c for c, r in bottom_row_for_col.items()
                    if not 0 <= x_label_row - r <= LABEL_ALIGN_MAX_GAP}
    # Same for rows, which go to the RIGHT edge instead. Both relocations are
    # needed: a staircase row's label sits immediately left of the cell
    # diagonally below the one a staircase column's label sits under, so moving
    # only the columns just shifts the same overlap one step down the diagonal.
    right_labelled = {r for r, c in left_col_for_row.items()
                      if not 0 <= c - y_label_col <= LABEL_ALIGN_MAX_GAP}

    # Bucket index -> gridspec index. An entirely empty row or column is
    # dropped rather than reserved: the last row of an NxN grid never holds a
    # panel (its only candidates are diagonal or above-diagonal-and-magnetic),
    # and when the two populations' bucket counts differ the trailing rows are
    # empty too (medaka: 3 magnetic buckets vs. 4 visual, so rows 3 AND 4 of a
    # naive 4x4 grid are blank). Left in, those become a band of interior
    # whitespace between the panels and their own x labels that
    # bbox_inches="tight" can't crop.
    used_rows = sorted({r for r, _ in cells})
    used_cols = sorted({c for _, c in cells})
    row_ix = {r: i for i, r in enumerate(used_rows)}
    col_ix = {c: i for i, c in enumerate(used_cols)}
    n_rows, n_cols = len(used_rows), len(used_cols)

    # Margins are sized in inches, not figure fractions, so they stay correct
    # as the grid (and therefore the figure) grows; savefig(bbox_inches="tight")
    # crops whatever slack is left over. Sizing them this way also makes the
    # gridspec region exactly cell_size * n_cols by cell_size * n_rows, i.e.
    # every cell is already square before set_box_aspect touches it -- which is
    # what lets the label placement below read positions straight off the
    # gridspec (un-aspect-adjusted) rather than needing a draw pass first.
    cell_size = 2.4
    left_in = 1.15
    right_in = 0.15 + ((TOP_LABEL_PAD_PT + LABEL_FONTSIZE * LINE_SPACING) / 72 if right_labelled else 0.0)
    label_block_in = (TICK_PAD_PT + (LABEL_FONTSIZE + 2 * COUNT_FONTSIZE) * LINE_SPACING) / 72 + 0.15
    top_label_in = (TOP_LABEL_PAD_PT + (LABEL_FONTSIZE + 2 * COUNT_FONTSIZE) * LINE_SPACING) / 72
    top_in = 1.1 + TITLE_BLOCK_PT / 72 + (top_label_in if top_labelled else 0.0)
    cbar_block_in = 1.35 if mode == "density" else 0.0
    bottom_in = label_block_in + cbar_block_in
    figw = cell_size * n_cols + left_in + right_in
    figh = cell_size * n_rows + top_in + bottom_in
    fig = plt.figure(figsize=(figw, figh))
    # wspace and hspace are deliberately kept EQUAL -- they are fractions of
    # the average axes width/height, so unequal values would make the cells
    # non-square and break the read-straight-off-the-gridspec label placement
    # described above. Both were widened from 0.18 to leave room for the
    # per-panel titles (TITLE_BLOCK_PT) in the inter-row gap.
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.28, hspace=0.28,
                          left=left_in / figw, right=1 - right_in / figw,
                          bottom=bottom_in / figh, top=1 - top_in / figh)

    n_by_bucket = {"neg": plot_df_neg.notna().sum(), "pos": plot_df_pos.notna().sum()}
    edges = np.linspace(0.0, 1.0, DENSITY_BINS + 1)
    norm = LogNorm(vmin=DENSITY_VMIN, vmax=1.0)
    cmaps = {pop: _density_cmap(name) for pop, name in DENSITY_CMAPS.items()}

    q_cache = {"neg": {}, "pos": {}}
    for row, col in cells:
        pop = "neg" if row > col else "pos"
        df = plot_df_neg if pop == "neg" else plot_df_pos
        ax = fig.add_subplot(gs[row_ix[row], col_ix[col]])
        # Only neurons present in BOTH buckets can be plotted at all; this
        # pairwise-complete count is what SHOW_PANEL_N reports, and under
        # QVAL_SCOPE "intersection" it is also the population each axis'
        # Storey correction is calibrated on.
        both, pi0x, pi0y = _panel_qvals(df, col, row, qval_scope, q_cache[pop])
        if mode == "density":
            counts, _, _ = np.histogram2d(both[col].clip(0, 1), both[row].clip(0, 1),
                                          bins=[edges, edges])
            if counts.sum():
                frac = np.ma.masked_where(counts == 0, counts / counts.sum())
                ax.pcolormesh(edges, edges, frac.T, cmap=cmaps[pop], norm=norm,
                              rasterized=True)
        else:
            ax.scatter(both[col], both[row], s=FP.MS_SMALL, alpha=FP.ALPHA_TRACE,
                       color=FP.COLOR_MAG if pop == "neg" else FP.COLOR_VIS,
                       rasterized=True)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xticks([0.0, 1.0])
        ax.set_yticks([0.0, 1.0])
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=TICK_FONTSIZE,
                       labelbottom=(bottom_row_for_col[col] == row),
                       labelleft=(left_col_for_row[row] == col))
        if SHOW_PANEL_N:
            # NB: not `title` -- that is this function's own parameter, holding
            # the page's suptitle.
            panel_title = f"n={len(both)}"
            if SHOW_PANEL_PI0 and len(both):
                panel_title += f"   $\\pi_0$ {pi0x * 100:.0f}/{pi0y * 100:.0f}%"
            ax.set_title(panel_title, fontsize=PANEL_TITLE_FONTSIZE, color="0.35",
                         pad=PANEL_TITLE_PAD_PT)

    # -- Bucket labels + per-bucket neuron counts -----------------------------
    # Drawn as figure text rather than ax.set_xlabel/set_ylabel: a label pulled
    # onto the main label line (LABEL_ALIGN_MAX_GAP) can land on a grid cell
    # that holds no axes at all, so there is nothing to hang set_xlabel on.
    for col in sorted(bottom_row_for_col):
        # How many neurons that bucket holds in each population -- the panel-n
        # in the middle of any given panel is the pairwise-complete
        # intersection of two of these, always smaller than either.
        count_lines = [(f"{tag} n={int(n_by_bucket[pop].get(col, 0))}", COUNT_FONTSIZE, color)
                       for pop, pop_cols, color, tag in (("neg", cols_neg, FP.COLOR_MAG, "mag"),
                                                         ("pos", cols_pos, FP.COLOR_VIS, "vis"))
                       if col in pop_cols]
        name_line = (f"bucket {col}", LABEL_FONTSIZE, "black")
        if col in top_labelled:
            cell = gs[row_ix[top_row_for_col[col]], col_ix[col]].get_position(fig)
            # Reading downward towards the panels: name, then counts.
            y, va, sign, stack, offset = (cell.y1, "bottom", 1, count_lines[::-1] + [name_line],
                                          TOP_LABEL_PAD_PT + TITLE_BLOCK_PT)
        else:
            cell = gs[row_ix[x_label_row], col_ix[col]].get_position(fig)
            y, va, sign, stack, offset = cell.y0, "top", -1, [name_line] + count_lines, TICK_PAD_PT
        x = 0.5 * (cell.x0 + cell.x1)
        for text, fontsize, color in stack:
            fig.text(x, y, text, ha="center", va=va, fontsize=fontsize, color=color,
                     transform=_offset_trans(fig, 0, sign * offset))
            offset += fontsize * LINE_SPACING

    for row in sorted(left_col_for_row):
        if row in right_labelled:
            cell = gs[row_ix[row], col_ix[right_col_for_row[row]]].get_position(fig)
            x, ha, dx = cell.x1, "left", TOP_LABEL_PAD_PT
        else:
            cell = gs[row_ix[row], col_ix[y_label_col]].get_position(fig)
            x, ha, dx = cell.x0, "right", -TICK_PAD_PT
        fig.text(x, 0.5 * (cell.y0 + cell.y1), f"bucket {row}",
                 ha=ha, va="center", rotation=90, fontsize=LABEL_FONTSIZE,
                 transform=_offset_trans(fig, dx, 0))

    if mode == "density":
        # Explicit cax rather than the repo's usual fig.colorbar(..., ax=axes):
        # stealing space from the grid axes would break the exactly-square cell
        # geometry the label placement above relies on. The band below the
        # bucket labels was reserved by cbar_block_in.
        #
        # Both bars and their text are sized in inches off the grid width and
        # then clamped, rather than taken as a plain fraction of it: as a
        # fraction, the pair stays legible on the 11-bucket Pigeon page but
        # overruns the figure (and collides with itself) on the 4-bucket
        # medaka one, whose grid is a third as wide while COUNT_FONTSIZE is
        # the same absolute size on every page.
        grid_left_in = gs[0, 0].get_position(fig).x0 * figw
        grid_w_in = gs[0, n_cols - 1].get_position(fig).x1 * figw - grid_left_in
        tags = ("magnetic (lower triangle)", "positive control (upper triangle)")
        cb_w_in = float(np.clip(0.28 * grid_w_in, 3.0, 7.0))
        cb_fs = float(np.clip(cb_w_in * 72 / (0.55 * max(len(t) for t in tags)),
                              9.0, COUNT_FONTSIZE))
        gap_in = 0.7
        x0_in = grid_left_in + 0.5 * (grid_w_in - (2 * cb_w_in + gap_in))
        for i, (pop, tag) in enumerate(zip(("neg", "pos"), tags)):
            cax = fig.add_axes([(x0_in + i * (cb_w_in + gap_in)) / figw, 0.62 / figh,
                                cb_w_in / figw, 0.20 / figh])
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmaps[pop]), cax=cax,
                              orientation="horizontal")
            cb.set_label(tag, fontsize=cb_fs)
            cb.ax.tick_params(labelsize=cb_fs * 0.85)
        # Shared caption, above the pair -- there is no room for it under the
        # bars' own tick labels and titles.
        fig.text((x0_in + cb_w_in + 0.5 * gap_in) / figw, 0.92 / figh,
                 "fraction of the panel's neurons per bin",
                 ha="center", va="bottom", fontsize=cb_fs)

    fig.suptitle(title, fontsize=FP.FS_BODY + 4, fontweight="bold",
                 y=1 - 0.35 / figh, va="top")
    return fig


def plot_fig3_supp2(all_fourier_df, out_dir: Path, out_name="Fig3_supp2.pdf",
                    mode="scatter", pivots=None, qval_scope="intersection",
                    bucket_order="random", seed=RANDOM_SEED):
    """One page per species (plus a "capped" second page, see below), written
    to `out_dir / out_name`. `mode` and `qval_scope` are passed straight
    through to `_plot_triangle_grid`. `pivots` optionally supplies an
    already-built `(neg_pivot_df, pos_pivot_df)` pair -- these are raw
    p-values and so are independent of `mode` and `qval_scope` (though NOT of
    `bucket_order`/`seed`, which decide the pivots' columns), which is what
    lets the scatter and density versions share one build.
    """
    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY}
    matplotlib.rc("font", **font)

    if pivots is None:
        pivots = tuple(build_wave_pivot(all_fourier_df, population=p,
                                        bucket_order=bucket_order, seed=seed)
                       for p in ("neg", "pos"))
    neg_pivot_df, pos_pivot_df = pivots

    species_list = sorted(set(neg_pivot_df.index.get_level_values("species")) |
                           set(pos_pivot_df.index.get_level_values("species")))

    out_path = out_dir / out_name
    n_pages = 0
    with PdfPages(out_path) as pdf:
        for species in species_list:
            neg_species_df = _species_subset(neg_pivot_df, species)
            pos_species_df = _species_subset(pos_pivot_df, species)

            cols_neg, counts_neg = _valid_cols(neg_species_df)
            cols_pos, counts_pos = _valid_cols(pos_species_df)

            if len(cols_neg) < MIN_WAVES and len(cols_pos) < MIN_WAVES:
                print(f"  Skipping {species}: neither population has >= {MIN_WAVES} bucket(s) with "
                      f">= {MIN_NEURONS_PER_WAVE} neurons (magnetic counts: {counts_neg.to_dict()}, "
                      f"visual counts: {counts_pos.to_dict()})")
                continue

            fig = _plot_triangle_grid(
                neg_species_df, pos_species_df, cols_neg, cols_pos,
                f"{species}: comparing q-values across buckets\n"
                f"magnetic n={len(neg_species_df)}, visual n={len(pos_species_df)}"
                f"  (q-values corrected per {qval_scope}; buckets ordered at {bucket_order})",
                mode=mode, qval_scope=qval_scope)
            if fig is None:
                print(f"  Skipping {species}: no plottable bucket pairs after all filters")
                continue
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            n_pages += 1
            print(f"  {species}: magnetic buckets {cols_neg}, visual buckets {cols_pos}")

            # Second, "capped" page: the two populations' bucket counts are
            # often wildly lopsided (e.g. Pigeon: 10 magnetic vs. 18 visual --
            # see the module docstring), which stretches the uncapped grid
            # into a mostly-empty rectangle rather than a clean square.
            # Capping both sides to whichever bucket count is smaller (+1, so
            # a min-side of N buckets still gets one extra bucket of headroom
            # rather than being pinned to exactly its own size) keeps the two
            # triangles closer to balanced without hiding the asymmetry
            # entirely -- this is a second, additional page, not a
            # replacement, since the uncapped page is the one that shows
            # every available bucket.
            capped_n = min(len(cols_neg), len(cols_pos)) + 1
            cols_neg_capped = [c for c in cols_neg if c < capped_n]
            cols_pos_capped = [c for c in cols_pos if c < capped_n]
            fig_capped = _plot_triangle_grid(
                neg_species_df, pos_species_df, cols_neg_capped, cols_pos_capped,
                f"{species}: comparing q-values across buckets (capped to {capped_n} buckets)\n"
                f"magnetic n={len(neg_species_df)}, visual n={len(pos_species_df)}"
                f"  (q-values corrected per {qval_scope}; buckets ordered at {bucket_order})",
                mode=mode, qval_scope=qval_scope)
            if fig_capped is None:
                print(f"  Skipping {species} capped page: no plottable bucket pairs at {capped_n} buckets")
                continue
            pdf.savefig(fig_capped, bbox_inches="tight")
            plt.close(fig_capped)
            n_pages += 1
            print(f"  {species} (capped to {capped_n}): magnetic buckets {cols_neg_capped}, "
                  f"visual buckets {cols_pos_capped}")

    print(f"Saved {out_path} ({n_pages} species page(s))")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Fig 3 supplement 2 (q-value reproducibility across recordings, by species)",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out-dir", default=FP.OUT_DIR, help="Output directory for PDFs")
    parser.add_argument("--parquet", default=FP.PARQUET_PATH,
                        help=f"Path to all_fourier_df.parquet (default: {FP.PARQUET_PATH})")
    parser.add_argument("--mode", choices=["scatter", "density", "both"], default="both",
                        help="Which version(s) to render (default: both)")
    parser.add_argument("--qval-scope", choices=QVAL_SCOPES, default="intersection",
                        help=QVAL_SCOPE_HELP)
    parser.add_argument("--bucket-order", choices=BUCKET_ORDERS, default="random",
                        help=BUCKET_ORDER_HELP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Permutation seed for --bucket-order random (default: {RANDOM_SEED})")
    args = parser.parse_args([] if in_notebook else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.parquet} ...")
    all_fourier_df = pd.read_parquet(args.parquet)

    pivots = tuple(build_wave_pivot(all_fourier_df, population=p,
                                    bucket_order=args.bucket_order, seed=args.seed)
                   for p in ("neg", "pos"))
    modes = ["scatter", "density"] if args.mode == "both" else [args.mode]
    for mode in modes:
        out_name = "Fig3_supp2.pdf" if mode == "scatter" else "Fig3_supp2_density.pdf"
        print(f"Rendering {mode} version (q-values corrected per {args.qval_scope}, "
              f"buckets ordered at {args.bucket_order}) ...")
        plot_fig3_supp2(all_fourier_df, out_dir, out_name=out_name, mode=mode, pivots=pivots,
                        qval_scope=args.qval_scope, bucket_order=args.bucket_order,
                        seed=args.seed)


if __name__ == "__main__":
    if in_notebook:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    main()
