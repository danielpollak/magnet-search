"""Fig 2 variants -- one multipage PDF of alternative populations for panel C.

Page 1  Panel C (magnetic p-value ECDF - uniform, bootstrap 95% band) for four
        populations: all magnetic units (canonical), without zebrafish, without
        medaka, without any GCaMP imaging (zebrafish + medaka). Each drawn by `fig2._plot_pvalue_ecdf_deviation`
        itself, so every panel is rendered exactly as canonical panel C is. A
        fifth panel overlays the four centre lines on one axis for direct
        comparison.

Asks how much of panel C's bulk excess comes from the imaging units, whose
deviation is ~3.6x the ephys units' (see docs/nfc_finite_sample_bias.md,
"How much of the real deviation does this explain?").

Requires:
  data/manuscript/all_fourier_df.parquet  (run python pipeline/aggregate.py first)

Usage:
    python pipeline/manuscript/fig2_variants.py
    python pipeline/manuscript/fig2_variants.py --out-dir figs/paper

Page 2  Single-number uniformity tests on each of page 1's four populations
        (one row per neuron -- first occurrence -- since every test assumes
        i.i.d. p-values): Kolmogorov-Smirnov, Cramer-von Mises,
        Anderson-Darling, Kuiper, Fisher's combined test, and Donoho-Jin
        higher criticism. Each observed statistic is drawn against its Monte
        Carlo null distribution under TWO nulls:
          U   exact Uniform(0,1) i.i.d., same N;
          FS  finite-sample null: each ephys unit's p-value is drawn from
              homogeneous-Poisson simulations at that unit's own spike count
              (docs/nfc_finite_sample_bias/simulate.py's simulator, which is
              self-tested against the production NFC); imaging units stay
              Uniform, as there is no spike-count model for them.
        FS is the honest null for "is there more here than finite spike counts
        produce on their own"; U is the textbook one.
Page 3  The same results as a table: statistic, U and FS null 95th
        percentiles, and Monte Carlo p-values (plus scipy's analytic
        p-value under U where one exists, as a cross-check on the MC).

The FS null needs a library of simulated null p-values per spike count; it is
built once (~minutes) and cached, untracked, at
data/manuscript/fig2_variants_fs_null_library.npz.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import stats

from magpyneto2 import statistics
import fig2

import format_parameters as FP

# The finite-sample simulator lives with the report that validates it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "docs" / "nfc_finite_sample_bias"))
import simulate as fs_sim  # noqa: E402

DEDUP_KEY = ["species", "ID", "date", "id"]
IMAGING = ["zebrafish", "medaka"]

# Categorical slots 1-4 (validated: dataviz validate_palette.js, light mode).
C_VARIANTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
C_NULL_U = "#8a8a8a"
C_NULL_FS = "#2a78d6"
C_INK = "#333333"

N_NULL = 2000   # Monte Carlo null replicates per (population, null)
FS_LIBRARY = _REPO_ROOT / "data" / "manuscript" / "fig2_variants_fs_null_library.npz"
# Log-spaced spike counts; real counts above the top are mapped to it, where
# the finite-sample bias has decayed into noise (see the report's sweep).
FS_GRID = np.unique(np.round(np.geomspace(50, 4000, 24)).astype(int))
FS_DRAWS = 10_000  # simulated null units per grid point


def magnetic_variants(all_neg_res):
    """(label, pre-dedup frame) for each population on page 1."""
    sp = all_neg_res["species"]
    return [
        ("All magnetic (canonical)", all_neg_res),
        ("Without zebrafish", all_neg_res[sp != "zebrafish"]),
        ("Without medaka", all_neg_res[sp != "medaka"]),
        ("Without GCaMP (ephys only)", all_neg_res[~sp.isin(["zebrafish", "medaka"])]),
    ]


def page_imaging_exclusions(pdf, all_neg_res):
    variants = magnetic_variants(all_neg_res)

    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], wspace=0.28, hspace=0.45)
    small = [fig.add_subplot(gs[0, 0])]
    small += [fig.add_subplot(gs[r, c], sharey=small[0]) for r, c in [(0, 1), (1, 0), (1, 1)]]
    ax_all = fig.add_subplot(gs[:, 2], sharey=small[0])

    ax_all.axhline(0, color=FP.COLOR_NULL, linestyle="--", linewidth=FP.LW_REFERENCE, alpha=0.6)
    print("\nPage 1: panel C with/without imaging")
    for ax, colr, (label, df) in zip(small, C_VARIANTS, variants):
        fig2._plot_pvalue_ecdf_deviation(ax, df, colr)
        n = len(df.drop_duplicates(subset=DEDUP_KEY))
        ax.set_title(f"{label}\nN={n}", fontsize=FP.FS_TITLE)

        # Overlay: the same centre line the small panel draws (seed=0, so the
        # bootstrap call reproduces it exactly).
        x, _, _, center = statistics.bootstrap_occurrence_ecdf_band(
            df, value_col="p_value", alpha=0.05, seed=0)

        # Null sampling band: under H0 the count of p-values below x is
        # Binomial(N, x), so ECDF(x) - x has mean 0 and variance x(1-x)/N;
        # +/-1.96 SD is its pointwise 95% interval. N = neurons that have a
        # finite p-value (one per neuron per draw, as the curve itself uses).
        # The coloured band is something else entirely: the spread over WHICH
        # observation each repeatedly-recorded neuron contributes (see
        # statistics.bootstrap_occurrence_ecdf_band) -- it carries no
        # sampling noise at all.
        n_fin = df[np.isfinite(df["p_value"])].drop_duplicates(subset=DEDUP_KEY).shape[0]
        half = 1.96 * np.sqrt(x * (1 - x) / n_fin)
        ax.fill_between(x, -half, half, color=C_NULL_U, alpha=0.25, linewidth=0, zorder=0)
        ax.plot(x, half, color=C_INK, linewidth=0.7, linestyle=":", zorder=1)
        ax.plot(x, -half, color=C_INK, linewidth=0.7, linestyle=":", zorder=1)

        ax_all.plot(x, center, color=colr, linewidth=1.5, label=f"{label} (N={n})")
        dev = float(np.interp(0.5, x, center))
        sd = np.sqrt(0.25 / n_fin)
        outside = np.mean(np.abs(center[1:-1]) > half[1:-1])
        print(f"  {label:26} N={n:5d} (finite p: {n_fin})  ECDF-uniform @ p=0.5: {dev:+.4f}"
              f"  = {dev / sd:.1f} null SD  (null 95% +/-{1.96 * sd:.4f});"
              f"  curve outside null band on {outside:.0%} of p")
    from matplotlib.patches import Patch
    small[0].legend(handles=[
        Patch(facecolor=C_VARIANTS[0], alpha=FP.ALPHA_CONFIDENCE,
              label="which-observation sensitivity (95%)"),
        Patch(facecolor=C_NULL_U, alpha=0.4, edgecolor=C_INK, linestyle=":",
              label="null sampling 95%: ±1.96·√(x(1−x)/N)")],
        fontsize=FP.FS_LEGEND - 1, frameon=False, loc="upper left")

    for ax in small[1::2]:
        ax.set_ylabel("")
    for ax in small[:2]:
        ax.set_xlabel("")

    ax_all.set_xlim(0, 1)
    ax_all.set_xlabel("p-value")
    ax_all.set_title("Overlay (bootstrap centre lines)", fontsize=FP.FS_TITLE)
    ax_all.legend(fontsize=FP.FS_LEGEND, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.1), ncol=1)
    ax_all.spines["top"].set_visible(False)
    ax_all.spines["right"].set_visible(False)

    fig.suptitle("Fig 2C variants: magnetic p-value ECDF deviation, with and without GCaMP imaging",
                 fontsize=FP.FS_TITLE + 1)
    pdf.savefig(fig, bbox_inches="tight", dpi=FP.DPI)
    plt.close(fig)


# ------------------------------------------------------------ uniformity tests
# name -> short description. Every statistic here is "large = evidence against
# the null", so the MC p-value is P(null stat >= observed).
TESTS = {
    "KS D": "max |ECDF(p) - p|",
    "CvM W2": "N * integral (ECDF - p)^2 dp",
    "AD A2": "same, weighted 1/[p(1-p)]",
    "Kuiper V": "max(ECDF-p) + max(p-ECDF)",
    "Fisher X2": "-2 sum log p (excess small p)",
    "HC*": "higher criticism (sparse small p)",
}


def uniformity_stats(p):
    """All six statistics for one sample of p-values. `p` may be 2-D (rows = samples)."""
    p = np.sort(np.clip(np.atleast_2d(p), 1e-300, 1 - 1e-16), axis=1)
    n = p.shape[1]
    i = np.arange(1, n + 1)
    d_plus = (i / n - p).max(1)
    d_minus = (p - (i - 1) / n).max(1)
    w2 = 1 / (12 * n) + (((2 * i - 1) / (2 * n) - p) ** 2).sum(1)
    a2 = -n - ((2 * i - 1) * (np.log(p) + np.log1p(-p[:, ::-1]))).sum(1) / n
    fisher = -2 * np.log(p).sum(1)
    # HC* (Donoho & Jin 2004): smallest half, skipping p_(i) <= 1/n, which
    # otherwise lets a single tiny p-value dominate.
    h = n // 2
    ph, ih = p[:, :h], i[:h]
    hc = np.sqrt(n) * (ih / n - ph) / np.sqrt(ph * (1 - ph))
    hc = np.where(ph > 1 / n, hc, -np.inf).max(1)
    return {"KS D": np.maximum(d_plus, d_minus), "CvM W2": w2, "AD A2": a2,
            "Kuiper V": d_plus + d_minus, "Fisher X2": fisher, "HC*": hc}


def analytic_uniform_p(p):
    """scipy's p-values under exact Uniform, where scipy has one (MC cross-check)."""
    p = p[np.isfinite(p)]
    return {"KS D": stats.kstest(p, "uniform").pvalue,
            "CvM W2": stats.cramervonmises(p, "uniform").pvalue,
            "Fisher X2": stats.combine_pvalues(np.clip(p, 1e-300, 1), method="fisher").pvalue}


def fs_null_library():
    """(grid spike counts, (n_grid, FS_DRAWS) simulated null p-values), cached."""
    if FS_LIBRARY.exists():
        z = np.load(FS_LIBRARY)
        if np.array_equal(z["grid"], FS_GRID) and z["pvals"].shape[1] == FS_DRAWS:
            return z["grid"], z["pvals"]
    print(f"  building finite-sample null library ({len(FS_GRID)} spike counts x {FS_DRAWS})")
    fs_sim._selftest()
    pv = np.empty((len(FS_GRID), FS_DRAWS))
    for k, n_spk in enumerate(FS_GRID):
        rng = np.random.default_rng(1000 + k)
        chunk = max(1, min(FS_DRAWS, 4_000_000 // n_spk))   # bounds the working array
        nfc = np.concatenate([fs_sim.nfc_poisson(min(chunk, FS_DRAWS - s), n_spk, rng)
                              for s in range(0, FS_DRAWS, chunk)])
        pv[k] = fs_sim.nfc_to_p(nfc)
        print(f"    {n_spk:5d} spikes: dev@0.5 = {fs_sim.dev_at_half(pv[k]):+.4f}")
    FS_LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FS_LIBRARY, grid=FS_GRID, pvals=pv)
    return FS_GRID, pv


def null_distributions(df, grid, lib, seed):
    """MC null statistics for population `df` (one row per neuron) under U and FS."""
    n = len(df)
    rng = np.random.default_rng(seed)
    ephys = ~df["species"].isin(IMAGING).values
    # Nearest grid point in log spike count; beyond the grid clamps to its ends.
    lg = np.log10(grid)
    spk = np.log10(np.clip(df["spk_count"].values[ephys], grid[0], grid[-1]))
    bin_idx = np.abs(spk[:, None] - lg[None, :]).argmin(1)

    out = {"U": {k: [] for k in TESTS}, "FS": {k: [] for k in TESTS}}
    batch = max(1, 2_000_000 // n)
    for s in range(0, N_NULL, batch):
        b = min(batch, N_NULL - s)
        for key in out:
            p = rng.random((b, n))
            if key == "FS" and ephys.any():
                p[:, ephys] = lib[bin_idx[None, :],
                                  rng.integers(lib.shape[1], size=(b, ephys.sum()))]
            for k, v in uniformity_stats(p).items():
                out[key][k].append(v)
    return {key: {k: np.concatenate(v) for k, v in d.items()} for key, d in out.items()}


def mc_p(null, obs):
    return (1 + np.sum(null >= obs)) / (len(null) + 1)


def _fmt_p(p):
    return f"<{1 / (N_NULL + 1):.0e}" if p <= 1 / (N_NULL + 1) + 1e-12 else f"{p:.2g}"


def run_tests(all_neg_res):
    grid, lib = fs_null_library()
    rows, nulls = [], {}
    print("\nUniformity tests (one row per neuron)")
    for j, (label, df) in enumerate(magnetic_variants(all_neg_res)):
        d = df.drop_duplicates(subset=DEDUP_KEY, keep="first")
        d = d[np.isfinite(d["p_value"])]
        obs = {k: float(v[0]) for k, v in uniformity_stats(d["p_value"].values).items()}
        ana = analytic_uniform_p(d["p_value"].values)
        nd = null_distributions(d, grid, lib, seed=j)
        nulls[label] = (obs, nd)
        for k in TESTS:
            rows.append(dict(population=label, N=len(d), test=k, statistic=obs[k],
                             U_q95=np.quantile(nd["U"][k], 0.95),
                             FS_q95=np.quantile(nd["FS"][k], 0.95),
                             p_U=mc_p(nd["U"][k], obs[k]), p_FS=mc_p(nd["FS"][k], obs[k]),
                             p_U_analytic=ana.get(k, np.nan)))
            r = rows[-1]
            print(f"  {label:27} {k:9} stat={r['statistic']:10.4g}  p_U={_fmt_p(r['p_U']):>7}"
                  f"  p_FS={_fmt_p(r['p_FS']):>7}  (analytic U: {r['p_U_analytic']:.2g})")
    return pd.DataFrame(rows), nulls


def page_test_nulls(pdf, nulls):
    labels = list(nulls)
    fig, axes = plt.subplots(len(TESTS), len(labels), figsize=(11, 13))
    for c, label in enumerate(labels):
        obs, nd = nulls[label]
        for r, k in enumerate(TESTS):
            ax = axes[r, c]
            u, f = nd["U"][k], nd["FS"][k]
            lo = min(u.min(), f.min(), obs[k])
            hi = max(u.max(), f.max(), obs[k])
            bins = np.linspace(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo), 60)
            ax.hist(u, bins=bins, histtype="stepfilled", color=C_NULL_U, alpha=0.45,
                    label="null U (exact uniform)")
            ax.hist(f, bins=bins, histtype="step", color=C_NULL_FS, linewidth=1.4,
                    label="null FS (finite-sample)")
            ax.axvline(obs[k], color=C_INK, linewidth=1.8, label="observed")
            ax.text(0.98, 0.95, f"p_U = {_fmt_p(mc_p(u, obs[k]))}\n"
                                f"p_FS = {_fmt_p(mc_p(f, obs[k]))}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7, color=C_INK)
            ax.set_yticks([])
            ax.tick_params(labelsize=7)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title(label, fontsize=FP.FS_TITLE)
            if c == 0:
                ax.set_ylabel(f"{k}\n{TESTS[k]}", fontsize=7, rotation=0, ha="right",
                              va="center", labelpad=6)
    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=3, frameon=False,
               fontsize=FP.FS_LEGEND, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Fig 2C variants: uniformity tests, observed statistic vs Monte Carlo nulls "
                 f"({N_NULL} replicates each; one row per neuron)", fontsize=FP.FS_TITLE + 1)
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    pdf.savefig(fig, dpi=FP.DPI)
    plt.close(fig)


def page_test_table(pdf, res):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    cols = ["population", "N", "test", "statistic", "U 95%", "FS 95%",
            "p (U, MC)", "p (U, scipy)", "p (FS, MC)"]
    cells = [[r.population, f"{r.N}", r.test, f"{r.statistic:.4g}", f"{r.U_q95:.4g}",
              f"{r.FS_q95:.4g}", _fmt_p(r.p_U),
              "-" if not np.isfinite(r.p_U_analytic) else f"{r.p_U_analytic:.2g}",
              _fmt_p(r.p_FS)] for r in res.itertuples()]
    tab = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center",
                   colWidths=[0.22, 0.06, 0.09, 0.1, 0.09, 0.09, 0.1, 0.1, 0.1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    tab.scale(1, 1.25)
    for (ri, ci), cell in tab.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if ri == 0:
            cell.set_text_props(weight="bold")
        # Shade by population block so the four groups read apart.
        elif (ri - 1) // len(TESTS) % 2 == 1:
            cell.set_facecolor("#f4f4f2")
    ax.set_title("Uniformity tests: statistics, null 95th percentiles, p-values\n"
                 "U = exact Uniform(0,1) null; FS = finite-sample null (ephys units drawn "
                 "from Poisson simulations at their own spike count;\nimaging units uniform). "
                 f"MC p-values floor at 1/(B+1), B = {N_NULL}. N counts neurons with a finite "
                 "p-value (page 1's N also counts owl units, which have none).",
                 fontsize=FP.FS_TITLE, loc="left")
    pdf.savefig(fig, dpi=FP.DPI)
    plt.close(fig)


def plot_fig2_variants(all_fourier_df, out_dir: Path):
    all_neg_res, _, _ = statistics.get_poscontrols_negresults(all_fourier_df)
    matplotlib.rc("font", family=FP.FONT_FAMILY, size=FP.FS_BODY_LG)
    res, nulls = run_tests(all_neg_res)
    csv = _REPO_ROOT / "data" / "manuscript" / "fig2_variants_uniformity_tests.csv"
    res.to_csv(csv, index=False)
    print(f"Saved {csv}")

    out_path = out_dir / "Fig2_variants.pdf"
    with PdfPages(out_path) as pdf:
        page_imaging_exclusions(pdf, all_neg_res)
        page_test_nulls(pdf, nulls)
        page_test_table(pdf, res)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 2 variants (panel C populations)")
    parser.add_argument("--out-dir", default=FP.OUT_DIR, help="Output directory for PDFs")
    parser.add_argument("--parquet", default=FP.PARQUET_PATH,
                        help=f"Path to all_fourier_df.parquet (default: {FP.PARQUET_PATH})")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.parquet} ...")
    plot_fig2_variants(pd.read_parquet(args.parquet), out_dir)


if __name__ == "__main__":
    main()
