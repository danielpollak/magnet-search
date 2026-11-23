"""Does the measured z drift predict each ROI's slow brightness change? (20221001_fish1)

    python docs/zplane/roi_z.py          # magneto2 env, after zpos.py; local disk only

For every selected ROI of the production segmentation (x64, whole-pixel drift correction):
  - its stack profile I(z): the ROI's lam-weighted brightness in each anatomy-stack plane, in
    photons per pixel per frame, after aligning the plane laterally to the run's mean image;
  - its observed brightness in each 60-frame chunk of the six trial tiffs (F_full / 64 - 100);
  - its predicted brightness in each chunk: I(z) at the chunk's z from zpos.py.
Only chunks whose z is inside the stack (not at its edge) and whose best plane correlates at
least MIN_CORR are used.

Bleaching or a laser-power change is a GLOBAL factor: it changes every ROI the same way. A z
drift is ROI-specific: as the plane moves, some ROIs brighten and others dim, in the direction
their stack profile predicts. The tests therefore compare ROIs with each other:
  1. across ROIs, the observed trend over the chunks against the predicted trend (both as
     relative change per stack unit of z; a global factor shifts every observed trend by the
     same amount and can't create this correlation);
  2. per ROI, the share of the chunk-to-chunk variance explained by the prediction, by the
     global factor (mean of all ROIs' relative brightness), and by both;
  3. the ceiling: the share that is reproducible at all (odd vs even frames of each chunk).

Outputs: results_roi_z.csv (one row per ROI), fig_roi_z.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import shift as nd_shift

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE.parent / "custom_registration"))
sys.path.insert(0, str(_HERE.parent / "resegmentation"))
import measure  # noqa: E402
import common  # noqa: E402
from pipeline import ophys_extraction as ox, schema  # noqa: E402

SESSION = "2022_10_01-fish1_cytoGCamp"
STACK = "anatomy stack no magneto"
TAG = "reseg_corrected_int"           # production since 2026-10-12
CHUNK = 60
TRIALS = [("magneto", c, f"2022_10_01_fish1_magneto_{c}.tif") for c in range(3)] + \
         [("no-magneto", c, f"2022_10_01_fish1_no_magneto_{c}.tif") for c in range(3)]
OFFSET = 100.0
MIN_CORR = 0.35      # a chunk's z counts only if its best plane correlates at least this well
ALIGN_DZ = 10
RINGS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 1000)]   # px, centroid distance


def stack_planes():
    d = np.load(f"C:/Users/dan/zplane_cache/{SESSION}.npz", allow_pickle=True)
    meta = pd.DataFrame(d["meta"])
    st = meta[(meta.kind == "stack") & (meta.folder == STACK)]
    zs = np.sort(st.z.unique())
    full = d["full"]
    return zs, np.stack([full[st.index[st.z == z]].mean(0) for z in zs])


def align(planes, ref, zs, z_ref):
    """Shift every plane onto ref (the run's mean image) by ONE lateral shift, measured on the
    planes within ALIGN_DZ of z_ref (the planes far from the recording's z don't resemble it, and
    their cross-correlation peak is noise); the stack is one 16-minute acquisition."""
    win = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    rf = np.conj(np.fft.fft2(measure.prep(ref, win)))
    near = np.abs(zs - z_ref) <= ALIGN_DZ
    sh = np.array([measure.xcorr_shift(rf, measure.prep(p, win)) for p in planes[near]])
    dy, dx = np.median(sh, 0)
    return np.stack([nd_shift(p, (-dy, -dx), order=1, mode="nearest") for p in planes]), sh


def chunk_means(F, n_chunks):
    """(ROI, chunk) means over all, odd and even frames."""
    C = F[:, :n_chunks * CHUNK].reshape(len(F), n_chunks, CHUNK)
    return C.mean(-1), C[..., 0::2].mean(-1), C[..., 1::2].mean(-1)


def r2(y, X):
    """Share of y's variance explained by an OLS fit on the columns of X (with intercept)."""
    A = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ b
    return 1 - res.var() / y.var()


def main():
    p0 = str(common.plane0(TAG, SESSION))
    ops = ox.load_ops(p0)
    stat = np.load(Path(p0) / "stat.npy", allow_pickle=True)
    cfg = schema.load_experiment(str(_REPO / "experiments" / "engert_20221001_fish1_magneto_0.yml"))
    sel = np.flatnonzero(common.selection(TAG, SESSION, cfg))

    zs, planes = stack_planes()
    ref = ops["meanImg"]
    if planes.shape[1:] != ref.shape:
        raise ValueError(f"stack {planes.shape[1:]} vs mean image {ref.shape}")
    Z = pd.read_csv(_HERE / "results_zpos.csv")
    Z = Z[(Z.session == SESSION) & (Z.stack_dir == STACK)]
    good = (~Z.at_edge) & (Z.peak_corr >= MIN_CORR)
    planes, sh = align(planes, ref, zs, Z.z_est[good].median())
    print("stack -> mean image lateral shift, planes near the recording (dy, dx):", np.round(sh, 1).tolist())
    # stack profile per ROI: lam-weighted, photons per pixel per frame
    prof = np.zeros((len(sel), len(zs)))
    for i, r in enumerate(sel):
        s = stat[r]
        # suite2p's own cell mask (extraction.masks.create_cell_mask, allow_overlap=False): the
        # ROI's pixels that no other ROI claims, lam normalised; F and F_full use exactly these
        m = ~s["overlap"]
        w = s["lam"][m] / s["lam"][m].sum()
        prof[i] = (planes[:, s["ypix"][m], s["xpix"][m]] - OFFSET) @ w

    obs, odd, even, zc, tc, cond = [], [], [], [], [], []
    for folder, c, tif in TRIALS:
        (_, start, n), = ox.trial_files(ops, tif)
        F = np.load(Path(p0) / "F_full.npy", mmap_mode="r")[sel, start:start + n] / 64.0 - OFFSET
        k = n // CHUNK
        m, mo, me = chunk_means(np.asarray(F, np.float64), k)
        # the trial tiff is the first k chunks of this condition, in acquisition order
        ch = Z[(Z.folder == folder) & (Z.cond == c)].sort_values("chunk").head(k)
        assert len(ch) == k, (tif, len(ch), k)
        obs.append(m); odd.append(mo); even.append(me)
        zc.append(np.where(ch.at_edge | (ch.peak_corr < MIN_CORR), np.nan, ch.z_est)); tc.append(ch.mtime.values)
        cond.append(np.full(k, f"{folder}_{c}"))
    obs, odd, even = (np.concatenate(a, axis=1) for a in (obs, odd, even))
    zc, tc, cond = np.concatenate(zc), np.concatenate(tc), np.concatenate(cond)
    use = ~np.isnan(zc)
    print(f"chunks with z inside the stack: {use.sum()} of {len(zc)}; "
          f"z {np.nanmin(zc):.1f} .. {np.nanmax(zc):.1f}; conditions {sorted(set(cond[use]))}")

    o, od, ev, z = obs[:, use], odd[:, use], even[:, use], zc[use]
    pred = np.array([np.interp(z, zs, p) for p in prof])
    mo_ = o.mean(1, keepdims=True)
    rel = o / mo_                                 # relative brightness per chunk
    glob = np.nanmean(rel, 0)                     # the global factor (bleaching, power)
    zc_ = z - z.mean()
    slope = lambda Y: (Y - Y.mean(1, keepdims=True)) @ zc_ / (zc_ @ zc_)
    obs_tr = slope(rel)                           # relative change per stack unit
    pred_tr = slope(pred / pred.mean(1, keepdims=True))
    rows = []
    for i in range(len(sel)):
        y = o[i]
        rows.append(dict(roi=sel[i], npix=stat[sel[i]]["npix"], mean=mo_[i, 0],
                         obs_trend=obs_tr[i], pred_trend=pred_tr[i],
                         r2_z=r2(y, pred[i]), r2_global=r2(y, glob), r2_both=r2(y, np.column_stack([pred[i], glob])),
                         split_r=np.corrcoef(od[i], ev[i])[0, 1]))
    R = pd.DataFrame(rows)
    # controls for the across-ROI test: the prediction of the NEAREST other ROI (a spatially smooth
    # trend field, e.g. a tilt or vignetting change, would predict as well from a neighbour)
    cy = np.array([np.median(stat[r]["ypix"]) for r in sel])
    cx = np.array([np.median(stat[r]["xpix"]) for r in sel])
    dist = np.hypot(cy[:, None] - cy[None], cx[:, None] - cx[None])
    np.fill_diagonal(dist, np.inf)
    nn = dist.argmin(1)
    R["pred_trend_nearest"] = R.pred_trend.values[nn]
    R["nearest_dist"] = dist.min(1)
    # per-ROI variance explained, with regressors: global factor, trial offsets, and either this
    # ROI's z prediction, another ROI's (random), or a linear time trend. z(t) falls steadily with
    # time, so any per-ROI trend regressor explains variance; only the comparison is informative.
    D = np.column_stack([(cond[use] == c).astype(float) for c in sorted(set(cond[use]))][1:])
    tt = (tc[use] - tc[use].min()) / 60
    perm = np.random.default_rng(0).permutation(len(sel))
    base = np.column_stack([glob, D])
    R["r2_global_trial"] = [r2(o[i], base) for i in range(len(sel))]
    R["r2_global_trial_z"] = [r2(o[i], np.column_stack([base, pred[i]])) for i in range(len(sel))]
    R["r2_global_trial_zother"] = [r2(o[i], np.column_stack([base, pred[perm[i]]])) for i in range(len(sel))]
    R["r2_global_trial_time"] = [r2(o[i], np.column_stack([base, tt])) for i in range(len(sel))]
    # the same with an ROI drawn at random from rings of increasing distance: a cell-specific
    # (z) prediction should lose its power within a cell diameter or two
    rng = np.random.default_rng(1)
    ring_rows = []
    for lo, hi in RINGS:
        pick = np.full(len(sel), -1)
        for i in range(len(sel)):
            c = np.flatnonzero((dist[i] >= lo) & (dist[i] < hi))
            if len(c):
                pick[i] = rng.choice(c)
        ok = pick >= 0
        r_, p_ = stats.spearmanr(R.obs_trend[ok], R.pred_trend.values[pick[ok]])
        ring_rows.append(dict(ring_px=f"{lo}-{hi}", n=int(ok.sum()), spearman=r_, p=p_))
    ring = pd.DataFrame(ring_rows)
    ring.to_csv(_HERE / "results_roi_z_rings.csv", index=False)
    print(ring.round(4).to_string(index=False))
    R.to_csv(_HERE / "results_roi_z.csv", index=False)
    rho, p = stats.spearmanr(R.obs_trend, R.pred_trend)
    rho_n, p_n = stats.spearmanr(R.obs_trend, R.pred_trend_nearest)
    print(f"{len(R)} ROIs; observed vs predicted trend across ROIs: Spearman {rho:.3f} (p = {p:.1e}); "
          f"with the nearest ROI's prediction (median {R.nearest_dist.median():.0f} px away): "
          f"{rho_n:.3f} (p = {p_n:.1e})")
    print(f"median trend, observed {R.obs_trend.median():+.4f}, predicted {R.pred_trend.median():+.4f} per unit z")
    # reliability ceiling: Spearman-Brown of the odd/even split
    R["ceiling"] = (2 * R.split_r / (1 + R.split_r)).clip(lower=0)
    print(R[["r2_z", "r2_global", "r2_both", "r2_global_trial", "r2_global_trial_z",
             "r2_global_trial_zother", "r2_global_trial_time", "ceiling"]].median().round(3).to_string())
    np.savez(_HERE / "results_roi_z_series.npz", z=z, t=tc[use], cond=cond[use], obs=o, pred=pred,
             zs=zs, prof=prof, roi=sel)


if __name__ == "__main__":
    main()
