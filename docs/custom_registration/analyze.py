"""What correcting 20220915_fish1's drift does to its ROI traces: nonstationarity and the
production Fourier test, suite2p's registration vs the custom one.

    python docs/custom_registration/measure.py      # magneto2: drift per frame -> shifts.npz
    python docs/custom_registration/extract.py      # suite2p1.0, GPU: F_custom.npy
    python docs/custom_registration/analyze.py      # magneto2: this, ~3 min
    python docs/custom_registration/analyze.py --figures-only

Four trace sets, each over each trial's frames, all with production's ROI selection (P(iscell)
> 0.5, npix >= 10, inside the fish outline, flatline removal):
    suite2p           production's ROIs and full-resolution traces (the NWB file / F_full.npy)
    custom            the same ROIs and the same warp, with the measured drift added (extract.py)
    reseg_raw         segment.py's x64 run on the raw frames: its ROIs and traces
    reseg_corrected   segment.py's x64 run on the drift-corrected frames: its ROIs and traces
The re-segmentations are loaded by load_segmentation, which applies production's selection to a
plane0 folder; check_loader asserts it reproduces production's ROIs and traces exactly.

Population: the six 20220915_fish1 recordings (experiments/engert_20220915_fish1_*.yml), the one
session whose drift is large (14-20 px).

Outputs: results_rois.csv (per ROI and trace set: nonstationarity p, Fourier p at the magnetic
frequency and, for the grating trials, at 1/60 Hz), results_summary.csv, fig_*.png.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "docs" / "nonstationarity"))
import nonstationarity as ns  # noqa: E402
from magpyneto2 import engert_helpers as eh  # noqa: E402

SESSION = "2022_09_15-fish1_cytoGCaMP"
DATA = Path(f"C:/Users/dan/suite2p_v1_custom_reg/{SESSION}")
PLANE0 = f"C:/Users/dan/suite2p_v1_production/{SESSION}/output/suite2p/plane0"
NAMES = [f"engert_20220915_fish1_{c}_{t}" for c in ("magneto", "no_magneto") for t in (0, 1, 2)]
PARQUET = ns.PARQUET    # NS_DATA_DIR (see nonstationarity.py) when run from a worktree
VISUAL_F, VISUAL_QFRAC = 0.016667, 0.5
ALPHA_NS, ALPHA_F = 0.05, 0.01
C_S2P, C_CUS = "#8a8a8a", "#2a78d6"
SET_STYLE = {"suite2p": (C_S2P, -0.27, "production ROIs, suite2p registration"),
             "custom": (C_CUS, -0.09, "production ROIs, drift corrected"),
             "reseg_raw": ("#e8590c", 0.09, "x64 re-segmentation, raw frames"),
             "reseg_corrected": ("#7a3fb0", 0.27, "x64 re-segmentation, drift corrected")}
C_INK, C_MUTED = "#333333", "#8a8a8a"
_orig_load = ns.sv.load


def trial_slice(name):
    from pipeline import ophys_extraction as ox
    tiff = name.replace("engert_20220915_fish1", "2022_09_15_fish1") + ".tif"
    for f, start, n in ox.trial_files(ox.load_ops(PLANE0)):
        if f == tiff:
            return start, n
    raise KeyError(tiff)


def load_custom(name, **kw):
    """sv.load with the traces replaced by the custom-registration ones (same ROIs, frames)."""
    cfg, F, roi = _orig_load(name, **kw)
    start, n = trial_slice(name)
    Fc = np.load(DATA / "F_custom.npy", mmap_mode="r")
    return cfg, np.asarray(Fc[roi, start:start + F.shape[1]], float), roi


SEGMENTATIONS = {      # trace set -> (plane0, trace file, divisor to raw photon units)
    "suite2p": (Path(PLANE0), "F_full.npy", 1.0),
    "reseg_raw": (DATA / "s2p_raw_x64/output/suite2p/plane0", "F.npy", 32.0),
    "reseg_corrected": (DATA / "s2p_corrected_x64/output/suite2p/plane0", "F.npy", 32.0),
}


def load_segmentation(tag, name, outline=True, series="full"):
    """sv.load for a segmentation of this session given by its plane0 folder: production's ROI
    selection (P(iscell) and npix above the YAML's thresholds, strict >; inside the fish outline
    drawn with the YAML's parameters on that run's own mean image; flatline removal on the
    trial's traces), applied to that run's ROIs and traces. segment.py's runs read tiffs x64,
    which suite2p halves, so their F.npy / 32 is in raw photon units like F_full.npy."""
    from pipeline import body_outline
    from magpyneto2.engert_helpers import remove_flatlines
    plane0, trace_file, div = SEGMENTATIONS[tag]
    cfg = ns.sv.schema.load_experiment(str(_REPO / "experiments" / f"{name}.yml"))
    stat = np.load(plane0 / "stat.npy", allow_pickle=True)
    p = np.load(plane0 / "iscell.npy")[:, 1]
    npix = np.array([s["npix"] for s in stat])
    mask = (p > cfg.iscell_threshold) & (npix > cfg.npix_threshold)
    if outline:
        mean = np.load(plane0 / "ops.npy", allow_pickle=True).item()["meanImg"]
        body = body_outline.body(mean, body_outline.params_for(cfg.body_outline))
        share = body_outline.inside_share([(s["ypix"], s["xpix"]) for s in stat], body)
        mask &= share >= body_outline.INSIDE_SHARE
    start, n = trial_slice(name)
    F = np.load(plane0 / trace_file, mmap_mode="r")
    F = np.asarray(F[mask, start:start + n], float) / div
    F, _, _, keep = remove_flatlines(F)
    roi = np.where(mask)[0][keep]
    N = min(int(120 * (F.shape[1] // 60)), F.shape[1])
    return cfg, F[:, :N], roi


def check_loader():
    """Self-test: load_segmentation on production's plane0 must give production's ROIs and
    traces (sv.load reads them from the NWB file the pipeline wrote)."""
    for name in NAMES:
        cfg, F, roi = _orig_load(name, outline=True, series="full")
        _, F2, roi2 = load_segmentation("suite2p", name)
        assert np.array_equal(roi, roi2), f"{name}: ROI sets differ ({len(roi)} vs {len(roi2)})"
        err = float(np.abs(F - F2).max())
        assert err < 1e-3, f"{name}: traces differ by {err}"
    print("  check: load_segmentation on production's plane0 reproduces production's ROIs and "
          "traces in all six recordings")


def fourier_p(F, cfg, f, qf):
    NFC, _, _, _, M, _ = eh.fit_Fourier(F, T=cfg.sample_period, f=f, Q_frac=qf)
    return ns.sv.pvals(np.asarray(NFC, float), M)


def job(args):
    name, tag = args
    if tag == "custom":
        ns.sv.load = load_custom
    elif tag.startswith("reseg"):
        ns.sv.load = lambda n, **kw: load_segmentation(tag, n, **kw)
    d, _ = ns.run(name)
    d = d[~d.synthetic].copy()
    cfg, F, roi = ns.sv.load(name, outline=True, series="full")
    assert np.array_equal(roi, d.roi.values)
    d["p_mag"] = fourier_p(F, cfg, cfg.analysis.f, cfg.analysis.Q_frac)
    d["p_vis"] = (fourier_p(F, cfg, VISUAL_F, VISUAL_QFRAC) if cfg.analysis.visual_f > 0
                  else np.nan)
    d["trace_set"] = tag
    return d


TRACE_SETS = ("suite2p", "custom", "reseg_raw", "reseg_corrected")


def compute():
    check_loader()
    with ProcessPoolExecutor(24) as pool:
        d = pd.concat(pool.map(job, [(n, t) for n in NAMES for t in TRACE_SETS]),
                      ignore_index=True)
    keep = ["experiment", "trace_set", "roi", "visual", "coil", "p_mag", "p_vis"] + \
        [c for c in d.columns if c.startswith("p_block_")]
    d = d[keep]
    d.to_csv(_HERE / "results_rois.csv", index=False)
    return d


def reconcile(d):
    """Tripwire, stated before running: the harness's production-trace p-values must reproduce
    the parquet's for every ROI (same fit_Fourier, same null); if not, nothing below is
    trusted."""
    pq = pd.read_parquet(PARQUET, columns=["rec", "freq", "id", "p_value"])
    s = d[d.trace_set == "suite2p"]
    worst = 0.0
    for name, g in s.groupby("experiment"):
        f = ns.sv.schema.load_experiment(str(_REPO / "experiments" / f"{name}.yml")).analysis.f
        ref = pq[(pq.rec == name) & np.isclose(pq.freq, f)].set_index("id").p_value
        worst = max(worst, float(np.abs(g.set_index("roi").p_mag - ref.reindex(g.roi)).max()))
        n_h, n_p = int((g.p_mag < ALPHA_F).sum()), int((ref < ALPHA_F).sum())
        assert n_h == n_p, f"{name}: {n_h} suspects in the harness, {n_p} in the parquet"
    # Not bit-identical: sv.pvals rounds the null grid's upper end up to an integer, production
    # uses max(NFC) * 1.05, so the numerical CDF differs in the 7th decimal.
    print(f"  harness vs parquet, magnetic-frequency p-values: max |difference| {worst:.2g}; "
          f"suspect counts identical in all six recordings")
    assert worst < 1e-5, "the harness disagrees with production"


def summary(d):
    rows = []
    for (name, tag), g in d.groupby(["experiment", "trace_set"]):
        rows.append(dict(experiment=name, trace_set=tag, rois=len(g),
                         rms_10=(g.p_block_rms_10 < ALPHA_NS).mean(),
                         band_10=(g.p_block_band_10 < ALPHA_NS).mean(),
                         cv2_10=(g.p_block_cv2_10 < ALPHA_NS).mean(),
                         suspects_mag=int((g.p_mag < ALPHA_F).sum()),
                         responders_1_60=int((g.p_vis < ALPHA_F).sum()) if g.p_vis.notna().any() else -1))
    t = pd.DataFrame(rows)
    # Chance check, approximate: suspects ~ Binomial(rois, ALPHA_F) if every ROI followed the null
    from scipy.stats import binom
    t["suspects_p_binom"] = binom.sf(t.suspects_mag - 1, t.rois, ALPHA_F)
    t.to_csv(_HERE / "results_summary.csv", index=False)
    t["coil"] = np.where(t.experiment.str.contains("no_magneto"), "off", "on")
    g = t.groupby(["trace_set", "coil"])[["rois", "suspects_mag"]].sum()
    g["share"] = g.suspects_mag / g.rois
    g["p_binom"] = binom.sf(g.suspects_mag - 1, g.rois, ALPHA_F)
    vis = t[t.responders_1_60 >= 0].groupby("trace_set")[["rois", "responders_1_60"]].sum()
    vis["share"] = vis.responders_1_60 / vis.rois
    print(g.round(4).to_string())
    print("  grating trials (_1, _2):")
    print(vis.round(3).to_string())
    pd.set_option("display.width", 200)
    print(t.drop(columns="coil").round(3).to_string(index=False))
    return t.drop(columns="coil")


# ------------------------------------------------------------------ figures
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_INK, labelsize=7)


def fig_shifts(out):
    """suite2p's stored shift vs the measured displacement, every bin of every trial."""
    from pipeline import ophys_extraction as ox
    ops = ox.load_ops(PLANE0)
    assert np.ptp(ops["yoff"]) == 0 and np.ptp(ops["xoff"]) == 0, "suite2p's rigid shift varies"
    assert np.ptp(ops["yoff1"]) == 0 and np.ptp(ops["xoff1"]) == 0, "suite2p's block shifts vary"
    b = pd.read_csv(_HERE / "results_bins.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.2), sharey=True)
    for ax, (fi, g) in zip(axes.ravel(), b.groupby("file_index")):
        t = (g.frame.values - g.frame.values[0] + 5) / 60
        for col, c, lab in (("dy", "#2a78d6", "y"), ("dx", "#e8590c", "x")):
            ax.plot(t, g[col], color=c, lw=1.0, label=f"measured, {lab}")
            ax.fill_between(t, g[col] - g[f"prec_{col}"], g[col] + g[f"prec_{col}"], color=c,
                            alpha=0.25, lw=0)
        ax.axhline(0, color=C_INK, lw=1.2, ls="--", label="suite2p (same shift every frame)")
        ax.set_title(g.file.iloc[0].replace("2022_09_15_fish1_", "").replace(".tif", ""),
                     fontsize=8, loc="left", color=C_INK)
        ax.set_xlabel("time in the trial tiff (min)", fontsize=7)
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("displacement (px)", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _prep(x):
    x = gaussian_filter(np.asarray(x, float), 1.5)
    x = x - gaussian_filter(x, 20)
    lo, hi = np.percentile(x, [1, 99.5])
    return np.clip((x - lo) / (hi - lo), 0, 1)


def fig_images(out):
    """First minute of magneto_0 (magenta) over minutes 10-21 (green), registered space:
    misaligned anatomy shows as separate magenta and green, aligned as white."""
    z = np.load(DATA / "registered_means.npz")
    late = _prep(z["settled"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, k, ttl in ((axes[0], "first_min_s2p", "suite2p's registration"),
                       (axes[1], "first_min_custom", "custom registration")):
        early = _prep(z[k])
        ax.imshow(np.dstack([early, late, early]), interpolation="nearest")
        ax.set_title(ttl, fontsize=9, color=C_INK)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.02, "magenta: first minute of magneto_0    green: minutes 10-21    "
             "white: both", ha="center", fontsize=8, color=C_INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_corr(out):
    """Each bin's correlation with the reference, before and after correction."""
    b = pd.read_csv(_HERE / "results_bins.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for ax, fi in zip(axes, (0, 3)):
        g = b[b.file_index == fi]
        t = (g.frame.values - g.frame.values[0] + 5) / 60
        ax.plot(t, g.corr_before, color=C_S2P, lw=1.1, label="as suite2p left it")
        ax.plot(t, g.corr_after, color=C_CUS, lw=1.1, label="after the custom correction")
        ax.set_title(g.file.iloc[0].replace("2022_09_15_fish1_", "").replace(".tif", ""),
                     fontsize=8, loc="left", color=C_INK)
        ax.set_xlabel("time in the trial tiff (min)", fontsize=7)
        _style(ax)
    axes[0].set_ylabel("correlation of a 10-frame mean\nwith the session reference", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_effect(t, out):
    """Share of ROIs nonstationary, suspect and responding, per trial and trace set."""
    t = t.copy()
    t["trial"] = t.experiment.str.replace("engert_20220915_fish1_", "")
    order = [n.replace("engert_20220915_fish1_", "") for n in NAMES]
    cols = [("rms_10", "RMS nonstationary"), ("band_10", "ref.-window power nonstationary"),
            ("cv2_10", "CV² nonstationary"),
            ("suspect_share", "share of suspects at 0.3 Hz (p < 0.01)"),
            ("responder_share", "share of 1/60 Hz responders (p < 0.01)")]
    t["suspect_share"] = t.suspects_mag / t.rois
    t["responder_share"] = np.where(t.responders_1_60 >= 0, t.responders_1_60 / t.rois, -1)
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 3.8))
    x = np.arange(len(order))
    for ax, (c, lab) in zip(axes, cols):
        for tag, (col, dx, lab_) in SET_STYLE.items():
            if tag not in set(t.trace_set):
                continue
            v = t[t.trace_set == tag].set_index("trial").reindex(order)[c].astype(float)
            if c == "responder_share":
                v[v < 0] = np.nan
            ax.scatter(x + dx, v, color=col, s=18, label=lab_, zorder=3)
        if c == "suspect_share":
            ax.axhline(ALPHA_F, color=C_INK, lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
        ax.set_title(lab, fontsize=8, color=C_INK)
        if c == "responder_share":
            ax.text(0.5, 0.97, "n/a for trial 0 (no grating)", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.5, color=C_MUTED)
        _style(ax)
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    if a.figures_only:
        d = pd.read_csv(_HERE / "results_rois.csv")
    else:
        d = compute()
        reconcile(d)
    t = summary(d)
    fig_shifts(_HERE / "fig_shifts.png")
    fig_images(_HERE / "fig_images.png")
    fig_corr(_HERE / "fig_corr.png")
    fig_effect(t, _HERE / "fig_effect.png")


if __name__ == "__main__":
    main()
