"""Grid search over (P(iscell), npix) inclusion thresholds for the zebrafish imaging units.

    python docs/zebrafish_pvalue_excess/threshold_grid.py                 # compute + plot
    python docs/zebrafish_pvalue_excess/threshold_grid.py --figures-only  # replot from CSVs

Why this is cheap
-----------------
fit_Fourier treats every trace independently, and flatline removal is a
per-trace test too, so a ROI's NFC/p-value does not depend on which OTHER
ROIs were included. We therefore compute p-values once for EVERY ROI
(thresholds off), and each grid cell is just a boolean mask over that table.
`_selftest` checks that masking at each recording's production thresholds
reproduces the production p-values in all_fourier_df.

Why sham frequencies
--------------------
A real magnetic response would ALSO push magnetic-frequency p-values off
uniform, so picking thresholds that flatten the magnetic-frequency ECDF could
tune a genuine effect away. Each ROI is therefore also fit at two SHAM
frequencies (SHAM_RATIOS x the magnetic frequency, non-harmonic, where no
stimulus was presented). Null calibration is judged on the sham p-values;
the magnetic frequency is only read off at whatever thresholds the sham
calibration supports.

Outputs (next to this script):
    all_rois.csv        one row per (rec, ROI, frequency kind): p_value, p_iscell, npix, ...
    grid.csv            one row per (population, frequency kind, iscell_thr, npix_thr)
    fig_threshold_grid.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "pipeline"))

from magpyneto2 import statistics as st  # noqa: E402
from magpyneto2.engert_helpers import fit_Fourier  # noqa: E402
from schema import load_experiment  # noqa: E402
from analysis_stages.engert import _load_from_nwb  # noqa: E402

PARQUET = _REPO / "data" / "manuscript" / "all_fourier_df.parquet"
SHAM_RATIOS = (0.71, 0.83)
# Floor at 0.5: below it suite2p rates the ROI more likely NOT a cell, which
# is never an acceptable inclusion rule regardless of calibration.
ISCELL_THR = np.round(np.arange(0.5, 0.951, 0.05), 2)
NPIX_THR = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100])
NEURON_KEY = ["ID", "date", "id"]   # id = suite2p ROI index (post id-fix)
MIN_N = 200                          # grid cells with fewer units are left blank


def batch_of(freq):
    return {0.4: "2022 Q1 (0.4 Hz)", 0.3: "0.3 Hz batch", 0.1: "0.1 Hz batch"}.get(
        round(float(freq), 3), f"{freq} Hz")


# ------------------------------------------------------------------ compute
def pool_recordings():
    """The zebrafish recordings (and their magnetic frequency) in the Fig2C pool."""
    df = pd.read_parquet(PARQUET)
    neg, _, _ = st.get_poscontrols_negresults(df)
    z = neg[neg["species"] == "zebrafish"]
    return z, z.groupby("rec")["freq"].first()


def fit_all_rois(rec, f_mag):
    cfg = load_experiment(str(_REPO / "experiments" / f"{rec}.yml"))
    # Thresholds of -1 keep every ROI; flatline removal still runs (per trace).
    F, roi_df, kept, _ = _load_from_nwb(cfg.nwb_path(), -1.0, -1.0)
    roi_idx = np.where(kept)[0]
    out = []
    for kind, f in [("magnetic", f_mag)] + [(f"sham {r}", f_mag * r) for r in SHAM_RATIOS]:
        try:
            NFC, _, _, _, M, _ = fit_Fourier(F, T=cfg.sample_period, f=f, Q_frac=cfg.analysis.Q_frac)
        except Exception as e:   # e.g. too few off-frequency bins at a low sham frequency
            print(f"    {rec}: skipping {kind} ({f:.3f} Hz): {e}")
            continue
        NFC = np.asarray(NFC)
        out.append(pd.DataFrame(dict(
            rec=rec, kind=kind, freq=f, id=roi_idx, p_value=st.corrected_pvalues(NFC, M),
            NFC=NFC, p_iscell=roi_df["p_iscell"].values[roi_idx],
            npix=roi_df["npix"].values[roi_idx], prod_iscell_thr=cfg.iscell_threshold,
            prod_npix_thr=cfg.npix_threshold)))
    return pd.concat(out, ignore_index=True)


def _selftest(allr, z):
    """Production thresholds applied to the all-ROI table must reproduce all_fourier_df."""
    m = allr[(allr["kind"] == "magnetic") & (allr["p_iscell"] > allr["prod_iscell_thr"])
             & (allr["npix"] > allr["prod_npix_thr"])]
    j = m.merge(z[["rec", "id", "p_value"]], on=["rec", "id"], suffixes=("", "_prod"),
                how="outer", indicator=True)
    missing = (j["_merge"] != "both").sum()
    err = np.nanmax(np.abs(j["p_value"] - j["p_value_prod"]))
    print(f"  self-test: {len(m)} masked rows vs {len(z)} production rows; "
          f"unmatched={missing}; max |dp|={err:.2e}")
    # corrected_pvalues sizes its null grid from the batch's max NFC, so the
    # all-ROI fit and the production fit differ only at the interpolation level.
    assert missing == 0, "masked all-ROI table does not match production's ROI set"
    assert err < 1e-3, f"p-values differ from production by up to {err:.2e}"


# ------------------------------------------------------------------ grid
def grid_metrics(allr, z_meta):
    allr = allr.merge(z_meta, on="rec", how="left")
    rows = []
    pops = [("all zebrafish", allr)] + list(allr.groupby("batch"))
    for pop, d in pops:
        for kind, dk in d.groupby("kind"):
            for ti in ISCELL_THR:
                for tn in NPIX_THR:
                    s = dk[(dk["p_iscell"] > ti) & (dk["npix"] > tn)]
                    s = s.drop_duplicates(subset=NEURON_KEY, keep="first")
                    p = s["p_value"].values
                    p = p[np.isfinite(p)]
                    n = len(p)
                    r = dict(population=pop, kind=kind, iscell_thr=ti, npix_thr=tn, n=n)
                    if n >= MIN_N:
                        dev = np.mean(p <= 0.5) - 0.5
                        r.update(dev_at_half=dev, z=dev / np.sqrt(0.25 / n),
                                 ks_p=stats.kstest(p, "uniform").pvalue,
                                 cvm_p=stats.cramervonmises(p, "uniform").pvalue)
                    rows.append(r)
        print(f"  grid done: {pop}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figure
def _heat(ax, g, col, title, cmap, vmin, vmax, prod=None, contour=None):
    piv = g.pivot(index="iscell_thr", columns="npix_thr", values=col)
    im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=6, rotation=90)
    ax.set_yticks(range(0, len(piv.index), 2))
    ax.set_yticklabels(piv.index[::2], fontsize=6)
    if contour is not None:
        c = g.pivot(index="iscell_thr", columns="npix_thr", values=contour[0]).values
        ax.contour(np.nan_to_num(c, nan=0.0), levels=[contour[1]], colors="k", linewidths=0.8)
    for (ti, tn) in prod or []:
        if ti in piv.index and tn in piv.columns:
            ax.plot(list(piv.columns).index(tn), list(piv.index).index(ti), marker="*",
                    color="white", markeredgecolor="k", markersize=9)
    ax.set_title(title, fontsize=7)
    return im


def figure(grid, prod, out):
    pops = ["all zebrafish", "2022 Q1 (0.4 Hz)", "0.3 Hz batch", "0.1 Hz batch"]
    pops = [p for p in pops if p in set(grid["population"])]
    sham = sorted(k for k in grid["kind"].unique() if k.startswith("sham"))
    # Sham calibration: average the two sham frequencies (independent fits).
    sh = (grid[grid["kind"].isin(sham)]
          .groupby(["population", "iscell_thr", "npix_thr"], as_index=False)
          .agg(dev_at_half=("dev_at_half", "mean"), z=("z", "mean"), ks_p=("ks_p", "min"),
               n=("n", "first")))
    mag = grid[grid["kind"] == "magnetic"]

    # Colour = dev@0.5 itself (effect size), NOT its z: N shrinks steeply with
    # stricter thresholds, so z falls with N even where dev doesn't move, and
    # would make losing units look like fixing them. |z|=2 is drawn as a
    # contour so significance is still visible.
    rows = [("n", "units retained", "Greys", 0, None, mag),
            ("dev_at_half", "SHAM: mean dev@0.5\n(contour |z|=2)", "RdBu_r", -0.06, 0.06, sh),
            ("ks_p", "SHAM: min KS p\n(contour p=0.05)", "viridis", 0, 1, sh),
            ("dev_at_half", "MAGNETIC: dev@0.5\n(read-out only; contour |z|=2)", "RdBu_r",
             -0.06, 0.06, mag)]
    fig, axes = plt.subplots(len(rows), len(pops), figsize=(3.1 * len(pops), 11.5))
    for c, pop in enumerate(pops):
        for r, (col, lab, cmap, vmin, vmax, src) in enumerate(rows):
            g = src[src["population"] == pop]
            vmax_ = vmax if vmax is not None else g["n"].max()
            if col == "dev_at_half":
                g = g.assign(absz=g["z"].abs())
                contour = ("absz", 2)
            elif col == "ks_p":
                contour = ("ks_p", 0.05)
            else:
                contour = None
            im = _heat(axes[r, c], g, col, f"{pop}\n{lab}" if r == 0 else lab, cmap, vmin,
                       vmax_, prod=prod.get(pop, prod["all zebrafish"]), contour=contour)
            fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
            if c == 0:
                axes[r, c].set_ylabel("P(iscell) threshold (>)", fontsize=7)
            if r == len(rows) - 1:
                axes[r, c].set_xlabel("npix threshold (>)", fontsize=7)
    fig.suptitle("Zebrafish inclusion-threshold grid. Star = production thresholds (the YAMLs use P(iscell) > 0.55 or 0.6, npix > 10). "
                 f"Blank = fewer than {MIN_N} units.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ------------------------------------------------------------------ main
def main():
    z, recs = pool_recordings()
    z_meta = z.groupby("rec").agg(ID=("ID", "first"), date=("date", "first"),
                                  freq_mag=("freq", "first")).reset_index()
    z_meta["batch"] = z_meta["freq_mag"].map(batch_of)

    if "--figures-only" in sys.argv:
        allr = pd.read_csv(_HERE / "all_rois.csv")
        grid = pd.read_csv(_HERE / "grid.csv")
        grid = grid[grid["iscell_thr"].round(2).isin(ISCELL_THR)]
    else:
        print(f"Fitting all ROIs in {len(recs)} recordings (magnetic + {len(SHAM_RATIOS)} sham)")
        allr = pd.concat([fit_all_rois(rec, f) for rec, f in recs.items()], ignore_index=True)
        _selftest(allr, z)
        allr.to_csv(_HERE / "all_rois.csv", index=False)
        grid = grid_metrics(allr, z_meta[["rec", "ID", "date", "batch"]])
        grid.to_csv(_HERE / "grid.csv", index=False)

    prod_thr = (allr.merge(z_meta, on="rec").groupby("batch")[["prod_iscell_thr", "prod_npix_thr"]]
                .agg(lambda s: sorted(set(s))))
    prod = {b: [(ti, tn) for ti in r.prod_iscell_thr for tn in r.prod_npix_thr]
            for b, r in prod_thr.iterrows()}
    prod["all zebrafish"] = sorted({x for v in prod.values() for x in v})
    print("  production thresholds by batch:", prod)
    figure(grid, prod, _HERE / "fig_threshold_grid.png")


if __name__ == "__main__":
    main()
