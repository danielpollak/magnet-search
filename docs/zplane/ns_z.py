"""How much of the imaging traces' nonstationarity is the focal-plane (z) drift? (20221001_fish1)

    python docs/zplane/ns_z.py --workers 8          # magneto2 env, after zpos.py; local disk only
    python docs/zplane/ns_z.py --figures-only

Step 1, a constrained gain model. For ROI r and 1-minute chunk k of the coil-on trial tiffs,
the relative brightness y_rk = m_rk / mean_k(m_rk) is modelled as
    y_rk = g_k * (1 + beta * d_rk),
  g_k   a shared factor (bleaching, laser power, anything common to all ROIs), free per chunk;
  d_rk  the ROI's stack-predicted relative change at chunk k's z: pred_rk / mean_k(pred_rk) - 1,
        pred = the ROI's stack profile I(z) at the chunk's z (roi_z.py);
  beta  ONE coefficient for all ROIs (1 = the stack's prediction at face value).
Fitted by alternating least squares. Unlike roi_z.py's per-ROI regressions, the ROI can't pick
its own coefficient, so a profile that belongs to another ROI can only help as much as its
direction and size happen to match: the control refits the model with the profiles permuted
across ROIs. The z term's share of an ROI's chunk-to-chunk variance is
1 - var(e - beta d) / var(e), e = y / g - 1 (the ROI's variation net of the shared factor).

Attenuation: the stack profile is one 60-frame average per plane, so d is noisy and beta (and
the share) are biased towards 0. The profile is rebuilt from the stack's odd and from its even
frames; the split-half correlation across ROIs of the predicted trend (slope of the prediction
against z), stepped up by Spearman-Brown, is the reliability rel of the full-frame profile's
trend. Disattenuated: beta / rel, share / rel (classical measurement error in d).

Step 2, the nonstationarity metric (docs/nonstationarity/nonstationarity.py: chunk_metrics,
dispersion = std / mean across K chunks, block-shuffle null with 10 s blocks, 1,000 replicates,
p = (1 + #null >= observed) / 1001) on corrected traces of the three coil-on trial tiffs,
restricted to the frames whose 1-minute chunk has a z (the trailing usable chunks of each tiff).
Traces are F_full / 64 - 100 (photons per pixel per frame) of production's ROIs (selection +
flatline removal, as the pipeline). Versions:
  a       raw
  b       / g_k                         (shared factor)
  c       / (g_k * (1 + beta d_rk))      (shared factor and the ROI's own z gain)
  d<i>    / (g_k * (1 + beta d_pk))      (same beta, a permuted ROI p's profile: control)
  cdis, ddis<i>   the same with the disattenuated beta
  const   a / 1.37                       (sanity: must change no p-value)
The gain is applied piecewise constant per 60-frame chunk: each chunk is one contiguous
acquisition minute, while neighbouring chunks of a trial tiff are ~3 min apart in acquisition
time (interleaved conditions), so a gain change between chunks happens at the seam, and
interpolating across it would smear a real step. All versions of a trial use the same block
permutations (common random numbers).

Note on cv2: nonstationarity.chunk_metrics whitens each ordinate by its mean over axis -3,
which for its (..., ROI, K, freq) arrays is the ROI axis (the population's mean ordinate in that
chunk), not the ROI's own mean over the K chunks as its docstring says. So a ROI's cv2 depends
on the other rows of the array; here every version's array holds the same ROIs, all corrected.

Checks (asserted): this script's metric and null code, on nonstationarity.run's own input (NWB
traces plus its AR(1) control rows), reproduces run's observed dispersions and block-shuffle
p-values exactly, every metric and K, all three trials; this script's loader (F_full of the
suite2p run) equals the NWB traces; the restricted traces' per-chunk band power equals the
corresponding chunks of the whole trace; a constant gain changes no p-value (beyond a float32
tie, one replicate = 1/1001).

Outputs (docs/zplane/): results_ns_z_fit.csv (step 1), results_ns_z_roi.csv (step 1, per ROI),
results_ns_z_permbeta.npy (step 1, beta per permutation), results_ns_z_frames.csv (step 2,
frames per trial), results_ns_z_p.csv (step 2, per ROI and version), results_ns_z_shares.csv
(step 2, share p < 0.05), results_ns_z_checks.csv, fig_ns_z.png.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import zlib
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import shift as nd_shift
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "nonstationarity"))
import roi_z  # noqa: E402  (also puts common, measure and pipeline on the path)
import nonstationarity as ns  # noqa: E402
import common  # noqa: E402
from pipeline import ophys_extraction as ox, schema  # noqa: E402

SESSION, TAG, CHUNK, OFFSET = roi_z.SESSION, roi_z.TAG, roi_z.CHUNK, roi_z.OFFSET
SCALE = common.SCALE
TRIALS = [t for t in roi_z.TRIALS if t[0] == "magneto"]
NAMES = [f"engert_20221001_fish1_magneto_{c}" for c, _ in enumerate(TRIALS)]
KS = (7, 10, 14)        # 120, 80 and 60 frames (12, 8, 6 cycles of 0.1 Hz); 14 = one per minute
K_MAIN = 14
METRICS = ("rms", "band", "cv2")
N_PERM_FIT = 100        # permutations for step 1
N_BOOT = 200
CONST = 1.37
C_INK, C_MUTED = "#333333", "#8a8a8a"
C_OWN = "#2a78d6"
OUT = dict(fit=_HERE / "results_ns_z_fit.csv", roi=_HERE / "results_ns_z_roi.csv",
           p=_HERE / "results_ns_z_p.csv", shares=_HERE / "results_ns_z_shares.csv", g=_HERE / "results_ns_z_g.csv",
           dmat=_HERE / "results_ns_z_d.npz", fig_d=_HERE / "fig_ns_z_d.png",
           var=_HERE / "results_ns_z_variance.csv",
           checks=_HERE / "results_ns_z_checks.csv", fig=_HERE / "fig_ns_z.png")


# ------------------------------------------------------------------ data
def stack_halves():
    """Stack planes from all, odd and even frames (repeats of a plane averaged), as roi_z."""
    d = np.load(f"C:/Users/dan/zplane_cache/{SESSION}.npz", allow_pickle=True)
    meta = pd.DataFrame(d["meta"])
    st = meta[(meta.kind == "stack") & (meta.folder == roi_z.STACK)]
    zs = np.sort(st.z.unique())
    return zs, [np.stack([d[k][st.index[st.z == z]].mean(0) for z in zs]) for k in ("full", "odd", "even")]


def profiles(planes, stat, sel):
    """(ROI, plane) stack brightness over suite2p's own cell mask (non-overlapping pixels, lam
    normalised; extraction.masks.create_cell_mask), photons per pixel per frame (as roi_z)."""
    out = np.zeros((len(sel), len(planes)))
    for i, r in enumerate(sel):
        s = stat[r]
        m = ~s["overlap"]
        out[i] = (planes[:, s["ypix"][m], s["xpix"][m]] - OFFSET) @ (s["lam"][m] / s["lam"][m].sum())
    return out


def load():
    """Selection, chunk means and z of the coil-on trials, and the ROI profiles (full, odd, even
    stack frames, all moved by roi_z.align's one lateral shift)."""
    p0 = common.plane0(TAG, SESSION)
    ops = ox.load_ops(str(p0))
    stat = np.load(p0 / "stat.npy", allow_pickle=True)
    cfg = schema.load_experiment(str(_REPO / "experiments" / f"{NAMES[0]}.yml"))
    sel = np.flatnonzero(common.selection(TAG, SESSION, cfg))
    zs, (full, odd, even) = stack_halves()
    Z = pd.read_csv(_HERE / "results_zpos.csv")
    Z = Z[(Z.session == SESSION) & (Z.stack_dir == roi_z.STACK)]
    good = (~Z.at_edge) & (Z.peak_corr >= roi_z.MIN_CORR)
    full, sh = roi_z.align(full, ops["meanImg"], zs, Z.z_est[good].median())
    dy, dx = np.median(sh, 0)
    odd, even = ([nd_shift(p, (-dy, -dx), order=1, mode="nearest") for p in h] for h in (odd, even))
    prof = {k: profiles(np.asarray(v), stat, sel) for k, v in (("full", full), ("odd", odd), ("even", even))}
    F_all = np.load(p0 / "F_full.npy", mmap_mode="r")
    obs, halves, rows = [], [], []
    for t, (folder, c, tif) in enumerate(TRIALS):
        (_, start, n), = ox.trial_files(ops, tif)
        k = n // CHUNK
        m, mo, me = roi_z.chunk_means(np.asarray(F_all[sel, start:start + n], np.float64) / SCALE - OFFSET, k)
        ch = Z[(Z.folder == folder) & (Z.cond == c)].sort_values("chunk").head(k)
        assert len(ch) == k
        ok = (~ch.at_edge.values) & (ch.peak_corr.values >= roi_z.MIN_CORR)
        obs.append(m[:, ok])
        halves.append((mo[:, ok], me[:, ok]))
        rows += [dict(trial=t, pos=j, z=z, chunk=cc, mtime=mt) for j, z, cc, mt in
                 zip(np.flatnonzero(ok), ch.z_est.values[ok], ch.chunk.values[ok], ch.mtime.values[ok])]
    ck = pd.DataFrame(rows)
    odd_even = tuple(np.concatenate([h[i] for h in halves], 1) for i in (0, 1))
    return sel, zs, prof, np.concatenate(obs, 1), odd_even, ck


def rel_pred(prof, zs, z):
    """d_rk = pred_rk / mean_k pred_rk - 1, and the predicted trend (slope of pred / mean vs z)."""
    pred = np.array([np.interp(z, zs, p) for p in prof])
    q = pred / pred.mean(1, keepdims=True)
    zc = z - z.mean()
    return q - 1, (q - q.mean(1, keepdims=True)) @ zc / (zc @ zc)


# ------------------------------------------------------------------ step 1
def fit_gain(y, d, g_fixed=None, iters=500, tol=1e-12):
    """Least squares y_rk ~ g_k (1 + beta d_rk), one beta; g free per chunk unless g_fixed."""
    g = y.mean(0) if g_fixed is None else g_fixed
    beta = 0.0
    for _ in range(iters):
        gd = g * d
        b_new = float((gd * (y - g)).sum() / (gd ** 2).sum())
        if g_fixed is None:
            u = 1 + b_new * d
            g = (y * u).sum(0) / (u * u).sum(0)
        if abs(b_new - beta) < tol:
            beta = b_new
            break
        beta = b_new
    return beta, g


def z_share(y, g, d, beta, within_trial=None):
    """Per ROI: share of the variance of e = y / g - 1 explained by beta d, and the pooled share.
    within_trial: chunk labels; each ROI's trial means are removed from e and beta d first."""
    e, zt = y / g - 1, beta * d
    if within_trial is not None:
        for t in np.unique(within_trial):
            s = within_trial == t
            e[:, s] -= e[:, s].mean(1, keepdims=True)
            zt[:, s] -= zt[:, s].mean(1, keepdims=True)
    ec = e - e.mean(1, keepdims=True)
    rc = (e - zt) - (e - zt).mean(1, keepdims=True)
    ss_e, ss_r = (ec ** 2).sum(1), (rc ** 2).sum(1)
    return 1 - ss_r / ss_e, 1 - ss_r.sum() / ss_e.sum()


def reliability(a, b):
    r = np.corrcoef(a, b)[0, 1]
    return r, 2 * r / (1 + r)


def reproducible(odd_even, g):
    """Per ROI: the reproducible part of the chunk-to-chunk sum of squares of e = y / g - 1, the
    covariance of e from the chunks' odd frames with e from their even frames (photon noise is
    independent between them; neural activity and the slow change are not)."""
    eo, ee = ((h / h.mean(1, keepdims=True)) / g - 1 for h in odd_even)
    eo, ee = (x - x.mean(1, keepdims=True) for x in (eo, ee))
    return (eo * ee).sum(1)


def step1(sel, zs, prof, obs, odd_even, ck):
    y = obs / obs.mean(1, keepdims=True)
    z = ck.z.values
    d, tr = rel_pred(prof["full"], zs, z)
    d_o, tr_o = rel_pred(prof["odd"], zs, z)
    d_e, tr_e = rel_pred(prof["even"], zs, z)
    r_half, rel = reliability(tr_o, tr_e)
    # the same on d itself, centred across ROIs per chunk (the shared part is absorbed by g)
    dc = lambda x: (x - x.mean(0)).ravel()
    r_half_d, rel_d = reliability(dc(d_o), dc(d_e))
    rng = np.random.default_rng(0)
    perms = [rng.permutation(len(sel)) for _ in range(N_PERM_FIT)]
    rows, roi = [], dict(roi=sel)
    for gmode in ("free", "ROI mean"):
        gf = None if gmode == "free" else y.mean(0)
        beta, g = fit_gain(y, d, gf)
        sh, pooled = z_share(y, g, d, beta)
        sh_w, pooled_w = z_share(y, g, d, beta, ck.trial.values)
        e = y / g - 1
        ss_e = ((e - e.mean(1, keepdims=True)) ** 2).sum()
        rep = reproducible(odd_even, g).sum()
        # z term's share of the reproducible variation: explained SS / reproducible SS
        pooled_rep, rep_frac = pooled * ss_e / rep, rep / ss_e
        boot = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(sel), len(sel))
            boot.append(fit_gain(y[i], d[i], gf)[0])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pb, psh, psh_w, ppool = [], [], [], []
        for p in perms:
            b_, g_ = fit_gain(y, d[p], gf)
            s_, pl_ = z_share(y, g_, d[p], b_)
            pb.append(b_); psh.append(np.median(s_)); ppool.append(pl_)
            psh_w.append(np.median(z_share(y, g_, d[p], b_, ck.trial.values)[0]))
        base = dict(g_model=gmode, n_roi=len(sel), n_chunk=y.shape[1])
        rows.append(dict(base, profile="own", estimate="raw", beta=beta, beta_ci_lo=lo, beta_ci_hi=hi,
                         share_median=np.median(sh), share_q25=np.percentile(sh, 25),
                         share_q75=np.percentile(sh, 75), share_pooled=pooled,
                         share_within_trial_median=np.median(sh_w), share_within_trial_pooled=pooled_w,
                         reproducible_fraction=rep_frac, share_of_reproducible_pooled=pooled_rep))
        rows.append(dict(base, profile="own", estimate="disattenuated", beta=beta / rel,
                         beta_ci_lo=lo / rel, beta_ci_hi=hi / rel,
                         share_median=np.median(sh) / rel, share_q25=np.percentile(sh, 25) / rel,
                         share_q75=np.percentile(sh, 75) / rel, share_pooled=pooled / rel,
                         share_within_trial_median=np.median(sh_w) / rel,
                         share_within_trial_pooled=pooled_w / rel, reproducible_fraction=rep_frac,
                         share_of_reproducible_pooled=pooled_rep / rel))
        rows.append(dict(base, profile=f"permuted (median of {N_PERM_FIT})", estimate="raw",
                         beta=np.median(pb), beta_ci_lo=np.percentile(pb, 2.5),
                         beta_ci_hi=np.percentile(pb, 97.5), share_median=np.median(psh),
                         share_pooled=np.median(ppool), share_within_trial_median=np.median(psh_w),
                         beta_abs_max=np.max(np.abs(pb)), share_median_max=np.max(psh),
                         p_perm_beta=(1 + np.sum(np.abs(pb) >= abs(beta))) / (1 + len(pb))))
        if gmode == "free":
            roi.update(share=sh, share_within_trial=sh_w, pred_trend=tr, pred_trend_odd=tr_o,
                       pred_trend_even=tr_e)
            # one permuted share per ROI, for the figure: ROI r's share with ROI perm[r]'s profile
            b0, g0 = fit_gain(y, d[perms[0]])
            roi["share_perm0"] = z_share(y, g0, d[perms[0]], b0)[0]
            fit = dict(beta=beta, g=g, d=d, perms=perms, rel=rel, perm_betas=np.array(pb),
                       perm_share_medians=np.array(psh), share=sh)
    F = pd.DataFrame(rows)
    F["rel_trend_split_half"], F["rel_trend_SB"] = r_half, rel
    F["rel_d_split_half"], F["rel_d_SB"] = r_half_d, rel_d
    return F, pd.DataFrame(roi), fit


# ------------------------------------------------------------------ step 2
class _Ks:
    """Temporarily set nonstationarity.KS (chunk_length nests the chunks over max(KS))."""
    def __init__(self, ks):
        self.ks = tuple(ks)

    def __enter__(self):
        self.old, ns.KS = ns.KS, self.ks

    def __exit__(self, *a):
        ns.KS = self.old


def ns_test(X, T, f, qf, Ks, seed, nested=None, n_rep=ns.N_REP):
    """Observed dispersion, block-shuffle p and per-chunk series of each metric, as
    nonstationarity.run computes them (same metric code, same block permutations for a given
    seed: its block null draws from default_rng([seed, 1])). nested: run every K under one
    nonstationarity.KS (nested chunks, for the self-test); default each K alone."""
    X0 = np.asarray(X, np.float32)
    mu = X0.mean(1)
    N, B = X0.shape[1], int(round(ns.BLOCK_S / T))
    kss = {K: (nested if nested is not None else (K,)) for K in Ks}
    obs, series = {}, {}
    for K in Ks:
        with _Ks(kss[K]):
            m, _ = ns.chunk_metrics(X0[None], T, f, qf, mu, K)
        for k in METRICS:
            obs[k, K], series[k, K] = ns.dispersion(m[k][0]), m[k][0]
    null = {(k, K): np.empty((n_rep, len(X0)), np.float32) for k in METRICS for K in Ks}
    r = np.random.default_rng([seed, 1])
    for b0 in range(0, n_rep, ns.BATCH):
        idx = np.stack([ns.permutation("block", N, B, r) for _ in range(ns.BATCH)])
        Xp = X0[:, idx].transpose(1, 0, 2)
        for K in Ks:
            with _Ks(kss[K]):
                m, _ = ns.chunk_metrics(Xp, T, f, qf, mu, K)
            for k in METRICS:
                null[k, K][b0:b0 + ns.BATCH] = ns.dispersion(m[k])
    p = {}
    for key, o in obs.items():
        p[key] = (1 + (null[key] >= o).sum(0)) / (1 + n_rep)
        p[key][~np.isfinite(o)] = np.nan
    return obs, p, series


def trial_data(t):
    """(cfg, photons (ROI, frame) of the whole trial, ROI indices, first usable frame)."""
    cfg, F, roi = common.load(TAG, NAMES[t])
    return cfg, F - OFFSET, roi


def _job(args):
    t, version, X, T, f, qf, seed, roi = args
    obs, p, _ = ns_test(X, T, f, qf, KS, seed)
    d = pd.DataFrame(dict(trial=t, experiment=NAMES[t], version=version, roi=roi, N=X.shape[1]))
    for (k, K), v in obs.items():
        d[f"obs_{k}_{K}"] = v
        d[f"p_{k}_{K}"] = p[k, K]
    print(f"  trial {t} {version}: {len(roi)} ROIs, {X.shape[1]} frames", flush=True)
    return d


def _selftest_job(t):
    """ns_test against nonstationarity.run on the whole trial, on run's own input: the NWB traces
    plus its AR(1) control rows, rebuilt with its seeds (chunk_metrics' cv2 whitens by the mean
    over the ROI axis, so cv2 depends on which other rows are in the array). And this script's
    loader (F_full of the suite2p run, photons) against the NWB traces."""
    name = NAMES[t]
    d_ref, _ = ns.run(name)
    d_ref = d_ref[~d_ref.synthetic].set_index("roi")
    cfg, F, roi_nwb = ns.sv.load(name, outline=True, series="full")
    seed = zlib.crc32(name.encode())
    pick = np.unique(np.linspace(0, len(F) - 1, min(ns.N_SYNTH, len(F))).astype(int))
    S, _ = ns.ar1(F[pick], np.random.default_rng([seed, 99]))
    X0 = np.concatenate([F, S]).astype(np.float32)
    _, X, roi = trial_data(t)
    assert np.array_equal(d_ref.index.values, roi) and np.array_equal(roi_nwb, roi), "ROI sets differ"
    rows = [dict(check=f"whole trial {t}: this loader's photons vs NWB traces / {SCALE} - {OFFSET:g}",
                 n=X.size, max_abs_diff_photons=float(np.max(np.abs(X - (F / SCALE - OFFSET)))))]
    obs, p, _ = ns_test(X0, cfg.sample_period, cfg.analysis.f, cfg.analysis.Q_frac, ns.KS, seed,
                        nested=ns.KS)
    obs = {key: v[:len(F)] for key, v in obs.items()}
    p = {key: v[:len(F)] for key, v in p.items()}
    for k in METRICS:
        for K in ns.KS:
            a, b = obs[k, K], d_ref[f"obs_{k}_{K}"].values
            ok = np.isfinite(a) & np.isfinite(b)
            pa, pb = p[k, K], d_ref[f"p_block_{k}_{K}"].values
            rows.append(dict(check=f"whole trial {t}: {k} K={K} vs nonstationarity.run",
                             n=int(ok.sum()), max_rel_diff_obs=float(np.max(np.abs(a[ok] - b[ok]) / np.abs(b[ok]))),
                             max_abs_diff_p=float(np.nanmax(np.abs(pa - pb))),
                             n_p_differ=int(np.sum(pa[ok] != pb[ok]))))
    return pd.DataFrame(rows)


def build_versions(fit, sel, ck, n_perm):
    """Per trial: the usable frames and every version's traces (as job argument tuples)."""
    beta, g, d, rel = fit["beta"], fit["g"], fit["d"], fit["rel"]
    pos_of = {s: i for i, s in enumerate(sel)}
    jobs, info, chunk_checks = [], [], []
    for t in range(len(TRIALS)):
        cfg, X, roi = trial_data(t)
        T, f, qf = cfg.sample_period, cfg.analysis.f, cfg.analysis.Q_frac
        c = ck[ck.trial == t]
        first = int(c.pos.min())
        assert np.array_equal(c.pos.values, np.arange(first, first + len(c))), "usable chunks not contiguous"
        assert X.shape[1] == (first + len(c)) * CHUNK
        Xu = X[:, first * CHUNK:]
        cols = np.flatnonzero(ck.trial.values == t)            # this trial's columns of g, d
        rows = np.array([pos_of[r] for r in roi])
        step = lambda v: np.repeat(v, CHUNK, axis=-1)          # piecewise constant per minute
        G = step(g[cols])
        seed = zlib.crc32(NAMES[t].encode())
        V = {"a": Xu, "b": Xu / G, "const": Xu / CONST,
             "c": Xu / (G * (1 + beta * step(d[rows][:, cols]))),
             "cdis": Xu / (G * (1 + beta / rel * step(d[rows][:, cols])))}
        for i in range(n_perm):
            dp = step(d[fit["perms"][i][rows]][:, cols])
            V[f"d{i}"] = Xu / (G * (1 + beta * dp))
            V[f"ddis{i}"] = Xu / (G * (1 + beta / rel * dp))
        for name, Xv in V.items():
            assert np.all(np.isfinite(Xv))
            jobs.append((t, name, Xv, T, f, qf, seed, roi))
        # check: the restricted traces' band power per 60-frame chunk equals the matching chunks
        # of the whole trace (frame indexing; band power is per chunk, so it must agree exactly)
        mu = lambda Y: np.asarray(Y, np.float32).mean(1)
        with _Ks((X.shape[1] // CHUNK,)):
            mw, _ = ns.chunk_metrics(np.asarray(X, np.float32)[None], T, f, qf, mu(X), X.shape[1] // CHUNK)
        with _Ks((Xu.shape[1] // CHUNK,)):
            mr, _ = ns.chunk_metrics(np.asarray(Xu, np.float32)[None], T, f, qf, mu(Xu), Xu.shape[1] // CHUNK)
        bw, br = mw["band"][0][:, first:], mr["band"][0]
        pos = bw > 0           # a chunk of an all-zero trace has no power
        assert np.all(br[~pos] == 0)
        chunk_checks.append(dict(check=f"trial {t}: restricted band power per chunk = whole trace's chunks {first}-{first + len(c) - 1}",
                                 n=int(pos.sum()), max_rel_diff_obs=float(np.max(np.abs(bw - br)[pos] / bw[pos]))))
        info.append(dict(trial=t, experiment=NAMES[t], n_roi=len(roi), frames_trial=X.shape[1],
                         first_usable_frame=first * CHUNK, frames_usable=Xu.shape[1],
                         **{f"frames_used_K{K}": _used(Xu.shape[1], K, f, T) for K in KS}))
    return jobs, pd.DataFrame(info), pd.DataFrame(chunk_checks)


def _used(N, K, f, T):
    with _Ks((K,)):
        return K * ns.chunk_length(N, K, f, T)[0]


def shares(P):
    """Share of ROIs with p < 0.05 per version, pooled over the three trials; permuted versions
    also averaged over their permutations (mean, min, max)."""
    cols = [f"p_{k}_{K}" for k in METRICS for K in KS]
    s = P.groupby("version")[cols].apply(lambda x: (x < ns.ALPHA).sum() / x.notna().sum())
    s["group"] = [v.rstrip("0123456789") for v in s.index]
    out = s.groupby("group")[cols].agg(["mean", "min", "max"])
    out.columns = [f"{c}_{a}" for c, a in out.columns]
    return out, s


# ------------------------------------------------------------------ figure
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_INK, labelsize=7.5)


def figure(Fit, R, S_by_version, G=None):
    """beta with the ROIs' own stack profiles against refits with the profiles permuted across
    ROIs. (Per-ROI shares are in results_ns_z_roi.csv; the flagged shares per trace version are a
    table in the doc, from results_ns_z_shares.csv.)
    One data colour (the ROI's own profile); grey = controls."""
    G = pd.read_csv(OUT["g"]) if G is None else G
    fig, (axg, ax) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(width_ratios=[1.3, 1],
                                                                            wspace=0.3))
    t = (G.mtime - G.mtime.min()) / 60
    for tr, (mk, lab) in enumerate((("o", "no grating"), ("<", "grating left"), (">", "grating right"))):
        m = G.trial == tr
        axg.plot(t[m], G.g[m], mk, ms=5, color=C_OWN, mfc="white" if tr == 0 else C_OWN,
                 label=lab)
    axg.legend(fontsize=7, frameon=False, loc="upper right")
    axg.axhline(1, color=C_MUTED, lw=0.8, ls=":")
    axg.set_xlabel("acquisition time (min)", fontsize=8)
    axg.set_ylabel(r"$\hat g_k$ (relative brightness shared by all ROIs)", fontsize=8)
    axg.set_title(f"A  shared factor, one value per minute ({len(G)})", fontsize=9, loc="left",
                  color=C_INK)
    _style(axg)
    fr = Fit[Fit.g_model == "free"]
    pb = np.load(_HERE / "results_ns_z_permbeta.npy")
    ax.scatter(np.random.default_rng(0).uniform(-0.15, 0.15, len(pb)), pb, s=9, color=C_MUTED, lw=0)
    for x, est in ((1, "raw"), (2, "disattenuated")):
        r = fr[(fr.profile == "own") & (fr.estimate == est)].iloc[0]
        ax.errorbar(x, r.beta, yerr=[[r.beta - r.beta_ci_lo], [r.beta_ci_hi - r.beta]], fmt="o",
                    color=C_OWN, ms=6, capsize=3, lw=1.2)
    ax.axhline(0, color=C_INK, lw=0.6)
    ax.axhline(1, color=C_MUTED, lw=0.8, ls=":")
    ax.text(2.45, 1, "β = 1:\nthe stack's\nprediction\nas is", fontsize=6.5, va="center", color=C_MUTED)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f"permuted\nprofiles ({len(pb)})", "own,\nraw", "own,\ndisattenuated"],
                       fontsize=7.5)
    ax.set_xlim(-0.5, 3.1)
    ax.set_ylim(-0.25, 1.15)
    ax.set_ylabel("β", fontsize=8)
    ax.set_title("B  one scale factor β for all ROIs' z gains", fontsize=9, loc="left", color=C_INK)
    _style(ax)
    fig.savefig(OUT["fig"], dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_d():
    """The model's inputs as ROI x minute matrices, minutes in acquisition order, ROIs sorted by
    the stack's predicted trend: A the stack's predicted relative change d_rk; B the observed
    relative change net of the shared factor, y_rk / g_k - 1; C the fitted z term beta d_rk."""
    z = np.load(OUT["dmat"])
    d, y, g, beta = z["d"], z["y"], z["g"], float(z["beta"])
    o = np.argsort(z["mtime"])
    t = (z["mtime"][o] - z["mtime"].min()) / 60
    tc = t - t.mean()
    order = np.argsort((d[:, o] - d[:, o].mean(1, keepdims=True)) @ tc)
    mats = [(d[:, o][order], "A  stack's prediction $d_{rk}$"),
            ((y / g - 1)[:, o][order], "B  observed, net of the shared factor: $y_{rk}/\\hat g_k - 1$"),
            ((beta * d)[:, o][order], f"C  fitted z term $\\hat\\beta\\, d_{{rk}}$ ($\\hat\\beta$ = {beta:.2f})")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, (m, title) in zip(axes, mats):
        im = ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
                       extent=(-0.5, m.shape[1] - 0.5, m.shape[0] - 0.5, -0.5))
        ax.set_title(title, fontsize=9, loc="left", color=C_INK)
        ax.set_xlabel("minute, in acquisition order (43)", fontsize=8)
        _style(ax)
    axes[0].set_ylabel(f"ROI ({d.shape[0]}), sorted by the stack's predicted trend\n"
                       "(top: predicted to dim; bottom: to brighten)", fontsize=8)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cb.set_label("relative change (0 = the ROI's average; clipped at ±1)", fontsize=8)
    fig.savefig(OUT["fig_d"], dpi=150, bbox_inches="tight")
    plt.close(fig)


def variance_table(n_perm=20):
    """Share of the ROIs' minute-to-minute variance (y_rk around its mean, 1) accounted for by the
    fitted model's parts, with the jointly fitted g and beta (no per-ROI coefficients):
        shared factor only    y ~ g_k
        z term only           y ~ 1 + beta d_rk
        both                  y ~ g_k (1 + beta d_rk)
    share = 1 - SS(residual) / SS(y - 1), pooled over ROIs (sums first) and per ROI (median).
    Control: both, refitted with the profiles shuffled across ROIs (median of n_perm)."""
    z = np.load(OUT["dmat"])
    y, d = z["y"], z["d"]
    beta, g = fit_gain(y, d)
    tot = ((y - 1) ** 2).sum(1)

    def shares(fit):
        res = ((y - fit) ** 2).sum(1)
        return 1 - res.sum() / tot.sum(), float(np.median(1 - res / tot))
    rows = [("the shared factor only", np.broadcast_to(g, y.shape)),
            ("the z term only", 1 + beta * d),
            ("both", g * (1 + beta * d))]
    out = [dict(part=n, **dict(zip(("pooled", "median_roi"), shares(f)))) for n, f in rows]
    rng = np.random.default_rng(0)
    pp, pm = [], []
    for _ in range(n_perm):
        dp = d[rng.permutation(len(d))]
        b_, g_ = fit_gain(y, dp)
        a, m = shares(g_ * (1 + b_ * dp))
        pp.append(a); pm.append(m)
    out.append(dict(part="both, profiles shuffled across ROIs (control)", pooled=np.median(pp),
                    median_roi=np.median(pm)))
    T = pd.DataFrame(out)
    T.to_csv(OUT["var"], index=False)
    print(T.round(3).to_string(index=False))
    return T


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=5, help="permuted-profile versions in step 2")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--step1-only", action="store_true", help="refit step 1 and redraw the figure")
    a = ap.parse_args()
    if a.figures_only:
        figure(pd.read_csv(OUT["fit"]), pd.read_csv(OUT["roi"]),
               shares(pd.read_csv(OUT["p"]))[1])
        return

    sel, zs, prof, obs, odd_even, ck = load()
    print(f"{len(sel)} ROIs, {len(ck)} chunks with z, per trial {ck.groupby('trial').size().to_dict()}")
    Fit, R, fit = step1(sel, zs, prof, obs, odd_even, ck)
    Fit.to_csv(OUT["fit"], index=False)
    R.to_csv(OUT["roi"], index=False)
    np.save(_HERE / "results_ns_z_permbeta.npy", fit["perm_betas"])
    G = ck.assign(g=fit["g"])
    G.to_csv(OUT["g"], index=False)
    y = obs / obs.mean(1, keepdims=True)
    np.savez_compressed(OUT["dmat"], d=fit["d"], y=y, g=fit["g"], beta=fit["beta"],
                        mtime=ck.mtime.values, trial=ck.trial.values, roi=sel)
    if a.step1_only:
        figure(Fit, R, None, G)
        figure_d()
        variance_table()
        return
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(Fit.round(4).to_string(index=False))

    jobs, info, chunk_checks = build_versions(fit, sel, ck, a.n_perm)
    print(info.to_string(index=False))
    with ProcessPoolExecutor(a.workers) as pool:
        selftests = [pool.submit(_selftest_job, t) for t in range(len(TRIALS))]
        P = pd.concat(pool.map(_job, jobs), ignore_index=True)
        checks = pd.concat([s.result() for s in selftests] + [chunk_checks], ignore_index=True)
    # constant gain: obs dispersion and p must be unchanged
    pa = P[P.version == "a"].set_index(["trial", "roi"])
    pc = P[P.version == "const"].set_index(["trial", "roi"]).loc[pa.index]
    for k in METRICS:
        for K in KS:
            o1, o2 = pa[f"obs_{k}_{K}"].values, pc[f"obs_{k}_{K}"].values
            ok = np.isfinite(o1)
            checks.loc[len(checks)] = dict(
                check=f"constant gain {CONST}: {k} K={K}", n=int(ok.sum()),
                max_rel_diff_obs=float(np.max(np.abs(o1[ok] - o2[ok]) / np.abs(o1[ok]))),
                max_abs_diff_p=float(np.nanmax(np.abs(pa[f"p_{k}_{K}"].values - pc[f"p_{k}_{K}"].values))),
                n_p_differ=int(np.sum(pa[f"p_{k}_{K}"].values[ok] != pc[f"p_{k}_{K}"].values[ok])))
    # vs nonstationarity.run: identical input, so identical p-values; the constant gain may flip
    # a replicate tie at float32 rounding (one replicate = 1/1001)
    exact = checks.check.str.contains("vs nonstationarity.run")
    checks["passed"] = ((checks.max_rel_diff_obs.fillna(0) < 1e-4)
                        & (checks.max_abs_diff_p.fillna(0) <= 2 / 1001)
                        & ~(exact & (checks.n_p_differ != 0))
                        & (checks.max_abs_diff_photons.fillna(0) < 1e-3))
    checks.to_csv(OUT["checks"], index=False)
    print(checks.to_string(index=False))
    P.to_csv(OUT["p"], index=False)
    summ, by_version = shares(P)
    summ.to_csv(OUT["shares"])
    info.to_csv(_HERE / "results_ns_z_frames.csv", index=False)
    print(summ[[c for c in summ.columns if c.endswith("_mean")]].round(3).T.to_string())
    figure(Fit, R, by_version)
    assert checks.passed.all(), "a check failed"


if __name__ == "__main__":
    main()
