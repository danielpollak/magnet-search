"""Fig 4B diagnostic -- NFC against S*a, the predicted detectability.

Panel B of Fig4 plots NFC against firing rate, one cloud per modulation
amplitude, and the clouds are offset from each other and internally spread.
Both are expected: a unit's NFC should scale with its own baseline
sensitivity S as well as with the imposed amplitude a. This diagnostic asks
whether the product S*a accounts for both -- i.e. whether every unit at every
amplitude collapses onto one curve once its own S is folded in, or whether
some systematic difference in S survives (between sites, or as a residual
dependence on S itself).

S is fig4.compute_sensitivity's per-unit proxy, spk_count / T / 2 / sigma,
measured on the UNMODULATED trains at this file's own freq/Q.

Measured result: S*a predicts NFC with a proportionality constant of ~1. The
median NFC/(S*a) is 1.1 overall and converges to 1.0 from above as amplitude
grows -- 2.6 at a=0.05, 1.5 at a=0.1, 1.2 at a=0.2, then 1.0-1.1 for a>=0.3.
So S*a is a directly calibrated prediction of NFC, not merely proportional to
it, and the panels' reference line is drawn at the empirical median rather
than at any assumed constant. (An earlier version of this docstring predicted
the constant should be T=350 s, reasoning from |c_s| ~ a*N/2 plus the 1/T in
S. That is wrong -- fourier_analysis's own normalization already absorbs it --
and it is why the line is taken from the data instead.)

The inflation at small a is a floor effect, not a failure of the collapse:
NFC cannot fall below the null distribution's own scale (whose 99th
percentile is the grey line), so once S*a drops under ~1 the ratio rises
mechanically while NFC just sits on the floor. Read panels A and C above that
floor; below it the points measure the null, not the modulation. The same
effect drives most of panel C's negative slope (Spearman rho = -0.39 for
NFC/(S*a) vs S), since low-S units are exactly those pinned to the floor at
every amplitude.

Three panels:

  A  NFC vs S*a, log-log, one hue per amplitude. Collapse onto a single
     line is the result being tested. The dashed line is the empirical
     median NFC/(S*a) across all points; the grey horizontal line is the
     eps-corrected 99th percentile of the null NFC, the same threshold
     Fig4's own panel B draws.
  B  The same axes, colored by recording SITE instead -- this is the
     "systematic differences in S" check. Sites separating here means S*a
     has not absorbed a per-site difference.
  C  The ratio NFC/(S*a) against S, log-log. Flat at the panel-A dashed
     level = the collapse is complete; any slope is a residual dependence on
     S that S*a alone does not capture.

The PDF holds two pages of these same three panels, differing only in axis
scaling: page 1 log-log, page 2 fully linear. The log page is the one the
collapse is meant to be read on -- NFC ~ S*a is a straight slope-1 line there,
and NFC spans two decades (panel C's ratio more), so the linear page
necessarily crowds most of the points toward the origin.

The a=0 condition has S*a identically 0 and so cannot appear on a log axis.
Its NFC distribution is drawn instead as a shaded band (median and IQR)
spanning panels A and B, which is what "no modulation" looks like on the same
y scale.

Amplitudes default to a denser set than Fig4's own three, because a collapse
test with two non-zero amplitudes is weak. Each is one full-population
fourier_analysis, they are mutually independent, and results are cached --
pass --workers to fan them out, --recompute to rebuild.

Usage:
    python pipeline/manuscript/fig4B_nfc_vs_Sa.py --workers 24
    python pipeline/manuscript/fig4B_nfc_vs_Sa.py --amplitudes 0.1 0.3 0.6
    python pipeline/manuscript/fig4B_nfc_vs_Sa.py --recompute --workers 24
"""
import argparse
import hashlib
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
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

from magpyneto2 import statistics

import fig4
import format_parameters as FP
from fig4B_diagnostics import load_keyed_pseudopopulation

OUT_NAME = "Fig4B_nfc_vs_Sa.pdf"

# Denser than fig4's own (0, 0.3, 0.6): the point of this page is whether
# points at DIFFERENT amplitudes land on one curve, which two non-zero
# amplitudes can barely show. 0.0 is included so the unmodulated baseline
# band (see module docstring) comes from the same computation as the rest.
DEFAULT_AMPLITUDES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8)

# Sequential palette for the amplitude hue (panel A) -- amplitude is an
# ordered quantity, so a sequential ramp reads correctly where Dark2's
# categorical hues would not. Panel B's sites ARE categorical, so it keeps
# the same Dark2 the other fig4B diagnostics use for sites.
PALETTE_AMPLITUDE = "viridis"
PALETTE_SITE      = "Dark2"

# Square axis limit for the linear page's panels A and B. The full S*a range
# runs to ~300, but that is a thin high-sensitivity tail; cropping at 100
# gives the bulk of the cloud most of the panel. No NFC in this sweep reaches
# 100, so the y crop hides nothing -- the count of points lost to the x crop
# is printed in the page title.
LINEAR_AXIS_MAX = 100


def null_nfc_99(FOURIER_Q):
    """The eps-corrected 99th percentile of the null NFC, as a plain float.

    statistics.inverse_Rayleigh_CDF returns a length-1 ARRAY, which is fine
    for the ax.hlines call fig4.plot_fig4 passes it to but breaks a pandas
    Series comparison ("Lengths must match to compare"), so it is coerced
    once here rather than at each use.
    """
    return float(np.asarray(
        statistics.inverse_Rayleigh_CDF(0.99, eps=statistics.get_epsilon(FOURIER_Q))
    ).ravel()[0])


def amplitude_cache_path(amplitudes):
    """Cache path stamped with fig4's own window tag AND a hash of the
    amplitude list.

    The window tag alone is not enough: two runs with different
    --amplitudes produce frames with different `mod` values, and reloading
    one as the other would silently plot a different sweep than the axis
    labels claim. The hash is truncated to 8 hex chars, which is ample for
    distinguishing a handful of hand-passed lists.
    """
    tag = hashlib.md5(repr(tuple(float(a) for a in amplitudes)).encode()).hexdigest()[:8]
    return (fig4._REPO_ROOT / "data" / "manuscript" /
            f"fig4B_nfc_vs_Sa_{fig4._WINDOW_TAG}_amps{len(amplitudes)}_{tag}.pkl")


def compute_nfc_by_amplitude(spks, FOURIER_Q, amplitudes, workers=1):
    """Per-unit NFC at each amplitude, as a long frame (id, mod, NFC).

    Reuses fig4._fr_cell -- the same worker fig4.compute_fr_df uses -- so
    these NFCs are computed by exactly the code path Fig4's own panel B
    goes through, one task per amplitude.
    """
    tasks = [(float(a),) for a in amplitudes]
    results = fig4._run_parallel(tasks, fig4._fr_cell, spks, fig4.FREQ, FOURIER_Q,
                                 workers, desc=f"NFC at {len(tasks)} amplitudes")
    rows = [pd.DataFrame({"id": np.arange(len(spks)), "mod": a, "NFC": NFCs})
            for a, NFCs in results]
    return pd.concat(rows, ignore_index=True)


def build_frame(keys, spks, FOURIER_Q, amplitudes, workers=1, recompute=False):
    """One row per (unit, amplitude): its S, its NFC, and the product S*a."""
    cache = amplitude_cache_path(amplitudes)
    if not recompute and cache.exists():
        print(f"Loading cached NFC-by-amplitude frame from {cache}")
        nfc_df = pd.read_pickle(cache)
    else:
        print(f"Computing NFC at {len(amplitudes)} amplitudes...")
        nfc_df = compute_nfc_by_amplitude(spks, FOURIER_Q, amplitudes, workers=workers)
        cache.parent.mkdir(parents=True, exist_ok=True)
        nfc_df.to_pickle(cache)
        print(f"Cached -> {cache}")
    if len(nfc_df) != len(spks) * len(amplitudes):
        raise ValueError(
            f"Cached frame at {cache} holds {len(nfc_df)} rows but this "
            f"population would give {len(spks) * len(amplitudes)} "
            f"({len(spks)} units x {len(amplitudes)} amplitudes) -- stale "
            f"cache; re-run with --recompute."
        )

    # S is measured once, on the unmodulated trains -- it is a property of
    # the unit, not of the amplitude applied to it, which is the whole
    # premise of plotting against S*a.
    sens = fig4.compute_sensitivity(spks, FOURIER_Q, freq=fig4.FREQ)
    meta = pd.DataFrame({
        "id": np.arange(len(spks)),
        "unit": keys,
        "experiment": [k.split(":")[0] for k in keys],
        "S": sens,
        "FR": fig4.unit_firing_rates(spks),
    })
    df = nfc_df.merge(meta, on="id")
    df["Sa"] = df["S"] * df["mod"]
    # Categorical label for the amplitude hue. With a numeric `mod` column
    # seaborn treats the hue as CONTINUOUS and labels the legend with its own
    # evenly-spaced levels (0.15, 0.30, 0.45, ...) -- values that are not in
    # the sweep at all, which misreads as if those amplitudes had been run.
    # An ordered categorical keeps the viridis ramp (amplitude is ordered)
    # while listing the real values.
    labels = [f"{a:g}" for a in sorted(df["mod"].unique())]
    df["a_label"] = pd.Categorical([f"{a:g}" for a in df["mod"]],
                                   categories=labels, ordered=True)
    return df


def print_summary(df, FOURIER_Q, nfc_99):
    """Median NFC/(S*a) overall, per amplitude and per site, plus rank
    correlations -- the ECDF-free numeric read of whether the collapse
    holds and at what constant.
    """
    pos = df[df["Sa"] > 0].copy()
    pos["ratio"] = pos["NFC"] / pos["Sa"]
    med = pos["ratio"].median()
    print(f"\nEmpirical NFC / (S*a): median {med:.1f}, "
          f"IQR {pos['ratio'].quantile(0.25):.1f}-{pos['ratio'].quantile(0.75):.1f}")
    # Restricted to points clear of the null floor, where the ratio is
    # actually measuring the modulation rather than the noise scale
    # (see module docstring).
    above = pos[pos["NFC"] > nfc_99]
    if not above.empty:
        print(f"  restricted to NFC > the null 99th pct ({nfc_99:.2f}), "
              f"n={len(above)}: median {above['ratio'].median():.2f}")

    print("\nby amplitude:")
    print(pos.groupby("mod")["ratio"]
             .agg(n="size", median="median",
                  p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75))
             .to_string(float_format=lambda v: f"{v:.1f}"))
    print("\nby site:")
    print(pos.groupby("experiment")["ratio"]
             .agg(n="size", median="median",
                  p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75))
             .to_string(float_format=lambda v: f"{v:.1f}"))

    print("\nSpearman rank correlation with NFC (a>0 rows only):")
    for col in ("Sa", "S", "mod", "FR"):
        sub = pos[[col, "NFC"]].replace([np.inf, -np.inf], np.nan).dropna()
        res = sp_stats.spearmanr(sub[col], sub["NFC"])
        print(f"  {col:<4} rho={res.statistic:+.3f}  p={res.pvalue:.2e}  n={len(sub)}")
    # If S*a captured everything, the ratio would be independent of S.
    sub = pos[["S", "ratio"]].replace([np.inf, -np.inf], np.nan).dropna()
    res = sp_stats.spearmanr(sub["S"], sub["ratio"])
    print(f"\nresidual: NFC/(S*a) vs S  rho={res.statistic:+.3f}  "
          f"p={res.pvalue:.2e}  (0 = S*a fully absorbs S)")


def _baseline_band(ax, df):
    """Shade the a=0 NFC median/IQR across the panel -- what "unmodulated"
    looks like on this y scale. a=0 has S*a == 0 exactly and so has no place
    on a log x axis, which is why it is a band rather than a hue.
    """
    zero = df.loc[df["mod"] == 0, "NFC"].dropna()
    if zero.empty:
        return
    ax.axhspan(zero.quantile(0.25), zero.quantile(0.75), color="0.85", zorder=0,
               label=f"a=0 IQR (median {zero.median():.2f})")


def _scatter_panel(ax, df, hue, palette, title, ratio_med, nfc_99, legend_title,
                   hue_order=None, logscale=True, axis_max=None):
    pos = df[(df["Sa"] > 0) & (df["NFC"] > 0)]
    _baseline_band(ax, df)
    sns.scatterplot(data=pos, x="Sa", y="NFC", hue=hue, hue_order=hue_order,
                    palette=palette, s=4,
                    linewidth=0, alpha=FP.ALPHA_SCATTER, ax=ax)
    # Proportional reference at the EMPIRICAL median ratio, drawn across the
    # panel's own x range -- on log-log a pure proportionality is a slope-1
    # line, so any curvature in the cloud away from it is a real departure
    # from NFC ~ S*a and not a scaling choice.
    # Sampled rather than drawn as a 2-point segment: matplotlib joins points
    # with straight lines in DISPLAY space, so a 2-point line is only correct
    # when both axes carry the same scale. They do on both pages here, but the
    # sampling is kept so a future page mixing a log axis with a linear one
    # cannot silently draw a straight dash in place of this curve.
    xs = np.geomspace(pos["Sa"].min(), pos["Sa"].max(), 256)
    ax.plot(xs, ratio_med * xs, "--", color="black", linewidth=1,
            label=f"NFC = {ratio_med:.0f} (S a)")
    ax.axhline(nfc_99, color="grey", linewidth=1)
    scale = "log" if logscale else "linear"
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    if axis_max is not None:
        # Cropped view. Both axes carry the same scale of quantity here (the
        # reference line is slope 1), so the limit is square to keep that line
        # at 45 degrees. Only x actually loses points -- _n_cropped counts
        # them into the page title, so the crop is never silent.
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
    ax.set_xlabel(r"$S \cdot a$  (sensitivity $\times$ modulation)")
    ax.set_ylabel("NFC")
    ax.set_title(title, fontsize=FP.FS_TITLE)
    ax.legend(fontsize=FP.FS_LEGEND, title=legend_title,
              title_fontsize=FP.FS_LEGEND, loc="upper left", markerscale=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _residual_panel(ax, df, ratio_med, logscale=True):
    pos = df[(df["Sa"] > 0) & (df["NFC"] > 0)].copy()
    pos["ratio"] = pos["NFC"] / pos["Sa"]
    a_order = [c for c in pos["a_label"].cat.categories if float(c) > 0]
    sns.scatterplot(data=pos, x="S", y="ratio", hue="a_label", hue_order=a_order,
                    palette=sns.color_palette(PALETTE_AMPLITUDE, n_colors=len(a_order)),
                    s=4, linewidth=0, alpha=FP.ALPHA_SCATTER, ax=ax, legend=False)
    ax.axhline(ratio_med, color="black", linestyle="dashed", linewidth=1)
    scale = "log" if logscale else "linear"
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_xlabel(r"$S$  (baseline sensitivity)")
    ax.set_ylabel(r"NFC / ($S \cdot a$)")
    ax.set_title("Residual dependence on S (flat = full collapse;\n"
                 "low-S rise is the null floor, not a real trend)",
                 fontsize=FP.FS_TITLE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _n_cropped(df, axis_max):
    """How many plotted (a>0) points fall outside an axis_max-square crop."""
    pos = df[(df["Sa"] > 0) & (df["NFC"] > 0)]
    return int(((pos["Sa"] > axis_max) | (pos["NFC"] > axis_max)).sum()), len(pos)


def _nfc_vs_Sa_page(df, ratio_med, nfc_99, logscale=True, axis_max=None):
    """One page of the three panels, log-log (logscale) or fully linear.

    Both pages show the same points; only the axis mapping changes. The log
    page is the one on which a pure NFC ~ S*a proportionality reads as a
    straight slope-1 line, and on which the low-S*a bulk is legible at all.
    The linear page shows the untransformed values -- expect the bulk crowded
    near the origin, since both S*a and NFC are heavy-tailed here. axis_max,
    if given, crops panels A and B to a square window; it is not applied to
    panel C, whose axes are a different quantity and a ratio.
    """
    crop_note = ""
    if axis_max is not None:
        n_out, n_tot = _n_cropped(df, axis_max)
        crop_note = (f", panels A/B cropped to {axis_max:g}"
                     f" ({n_out}/{n_tot} points outside)")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    # a=0 is excluded from the hue order: it has Sa == 0 exactly, so it never
    # appears in the scatter (it is the grey band instead), and leaving it in
    # would put a legend key next to no points.
    a_order = [c for c in df["a_label"].cat.categories if float(c) > 0]
    _scatter_panel(axes[0], df, "a_label",
                   sns.color_palette(PALETTE_AMPLITUDE, n_colors=len(a_order)),
                   "Colored by modulation amplitude", ratio_med, nfc_99, "a",
                   hue_order=a_order, logscale=logscale, axis_max=axis_max)
    _scatter_panel(axes[1], df, "experiment", PALETTE_SITE,
                   "Colored by recording site", ratio_med, nfc_99, "site",
                   logscale=logscale, axis_max=axis_max)
    _residual_panel(axes[2], df, ratio_med, logscale=logscale)
    for ax, letter in zip(axes, "ABC"):
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontfamily="arial",
                fontsize=FP.FS_PANEL, ha="left", va="bottom")

    fig.suptitle(
        f"Fig4 panel B re-plotted against predicted detectability S*a "
        f"({len(df['id'].unique())} pigeon-HP units, "
        f"{df['mod'].nunique()} amplitudes) -- "
        f"{'log-log' if logscale else 'LINEAR'} axes{crop_note}\n"
        f"grey line = 99th percentile of the null NFC; grey band = the a=0 IQR",
        fontsize=FP.FS_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def plot_nfc_vs_Sa(df, FOURIER_Q, out_path: Path):
    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY_XL}
    matplotlib.rc("font", **font)

    pos = df[df["Sa"] > 0]
    ratio_med = float((pos["NFC"] / pos["Sa"]).median())
    # eps-corrected exactly as fig4.plot_fig4 does -- these NFCs come from
    # fig4._fr_cell at this same FOURIER_Q.
    nfc_99 = null_nfc_99(FOURIER_Q)

    # The linear page is cropped square at LINEAR_AXIS_MAX; the log page is
    # not, since a log axis already fits the full range legibly.
    figs = [_nfc_vs_Sa_page(df, ratio_med, nfc_99, logscale=logscale,
                            axis_max=None if logscale else LINEAR_AXIS_MAX)
            for logscale in (True, False)]
    with PdfPages(out_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=FP.DPI_FIG4, bbox_inches="tight")
    print(f"\nSaved -> {out_path} ({len(figs)} pages: log-log, then linear)")
    if in_notebook:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="NFC vs S*a (sensitivity x modulation) for Fig4's pseudopopulation")
    parser.add_argument("--out-dir", default=FP.DIAGNOSTIC_DIR, help="Output directory for the PDF")
    parser.add_argument("--data-dir", default=FP.DATA_DIR,
                        help="Directory containing pipeline output .nwb files")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Explicit experiment names to pool (default: fig4's own "
                             "auto-discovery of species=='Pigeon' & area=='HP')")
    parser.add_argument("--amplitudes", nargs="*", type=float, default=list(DEFAULT_AMPLITUDES),
                        help=f"Modulation amplitudes to sweep (default: "
                             f"{' '.join(f'{a:g}' for a in DEFAULT_AMPLITUDES)})")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers; one task per amplitude")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute the NFC-by-amplitude frame even if cached")
    args = parser.parse_args([] if in_notebook else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = args.experiments or fig4.discover_pigeon_hp_experiments(fig4._EXPERIMENTS_DIR)
    print(f"Pooling pigeon-HP pseudopopulation from: {experiments}")
    keys, spks, Q_frac = load_keyed_pseudopopulation(args.data_dir, experiments)

    FOURIER_Q = fig4.fourier_Q_from_frac(spks, fig4.FREQ, Q_frac,
                                         context=f"[fig4B NFC vs S*a @ {fig4.FREQ}Hz] ")
    print(f"  Q_frac={Q_frac} -> FOURIER_Q={FOURIER_Q} bins at {fig4.FREQ} Hz")

    df = build_frame(keys, spks, FOURIER_Q, args.amplitudes,
                     workers=args.workers, recompute=args.recompute)
    print_summary(df, FOURIER_Q, null_nfc_99(FOURIER_Q))
    plot_nfc_vs_Sa(df, FOURIER_Q, out_dir / OUT_NAME)


if __name__ == "__main__":
    if in_notebook:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    main()
