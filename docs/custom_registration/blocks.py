"""Is 20220915_fish1's motion a whole-field translation, or does the image also rotate, stretch
or deform? Measures the displacement left over in local blocks after the rigid correction.

    python docs/custom_registration/blocks.py          # magneto2 env, ~1 min, CPU
    python docs/custom_registration/blocks.py --figures-only

Method: each 60-frame chunk of the stitched trial tiffs (122 chunks; a chunk is one minute of
continuous acquisition) is averaged, shifted back by its mean rigid displacement from
measure.py, and split into a GRID x GRID grid of overlapping BLOCK-px blocks. Each block is
cross-correlated with the same block of measure.py's reference (same filtering, a Hann taper per
block, parabolic subpixel peak). A pure translation leaves every block at ~0. A rotation leaves a
residual field that turns around the centre, a scale change one that points outward, a local
deformation one that no affine map fits.

Precision: odd and even frames of each chunk are measured separately; half their difference is
the block's measurement error. Blocks whose error exceeds MAX_PREC px (no structure to lock on to)
are not used.

Per chunk, an affine map r(p) = t + M (p - centre) is fitted by least squares to the block
residuals. Reported: the rotation (deg), the scale change (%), the shear, the residual RMS
before and after the affine fit, and the RMS expected from the measurement error alone. The
affine map is then applied to the chunk mean, and its correlation with the reference compared
with the rigid correction's.

Outputs:
    results_blocks.csv        one row per (chunk, block): residual, precision
    results_affine.csv        one row per chunk: affine parameters, RMS, correlations
    fig_blocks.png            residual fields of example chunks, and the affine parameters over
                              the session
"""
import os
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import affine_transform, shift as nd_shift

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import measure as ms  # noqa: E402

GRID, BLOCK = 5, 128
MARGIN = 24          # px kept clear of the frame edge, where the rigid shift (up to 20 px) fills in
MAXS_B = 12          # px search range per block (the rigid shift is already removed)
MAX_PREC = 0.5       # px; blocks measured worse than this are not used
MIN_MATCH = 0.4      # blocks whose best match correlates less than this with the reference are not used
EARLY = 7            # chunks 0..EARLY-1 of a trial tiff are its unsettled start (Figure 3)
OUTLIER = 2.0        # px; blocks this far off the affine fit are dropped and the fit repeated
CHUNK = ms.CHUNK


def load_chunks(path, n):
    """Mean, odd-frame mean and even-frame mean of each 60-frame chunk of one tiff."""
    mov = tifffile.imread(path).astype(np.float32)
    nc = n // CHUNK
    c = mov[:nc * CHUNK].reshape(nc, CHUNK, *mov.shape[1:])
    return c.mean(1), c[:, 0::2].mean(1), c[:, 1::2].mean(1)


def block_starts(L):
    return np.linspace(MARGIN, L - MARGIN - BLOCK, GRID).round().astype(int)


def block_shifts(ref, img, win):
    ref_f = np.conj(np.fft.fft2(ms.prep(ref, win)))
    c = np.fft.fftshift(np.real(np.fft.ifft2(ref_f * np.fft.fft2(ms.prep(img, win)))))
    cy, cx = np.array(c.shape) // 2
    sub = c[cy - MAXS_B:cy + MAXS_B + 1, cx - MAXS_B:cx + MAXS_B + 1]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)

    def para(a, b, d):
        den = a - 2 * b + d
        return 0.0 if den == 0 else 0.5 * (a - d) / den
    dy = para(*sub[iy - 1:iy + 2, ix]) if 0 < iy < sub.shape[0] - 1 else 0.0
    dx = para(*sub[iy, ix - 1:ix + 2]) if 0 < ix < sub.shape[1] - 1 else 0.0
    r = (iy - MAXS_B + dy, ix - MAXS_B + dx)
    match = ms.corr(ms.prep(nd_shift(img, [-r[0], -r[1]], order=1, mode="nearest"), win), ms.prep(ref, win))
    return (*r, match)


def measure_blocks(ref, img):
    win = np.outer(np.hanning(BLOCK), np.hanning(BLOCK))
    out = []
    for y in block_starts(img.shape[0]):
        for x in block_starts(img.shape[1]):
            out.append((y + BLOCK / 2, x + BLOCK / 2,
                        *block_shifts(ref[y:y + BLOCK, x:x + BLOCK], img[y:y + BLOCK, x:x + BLOCK], win)))
    return np.array(out)        # (n_blocks, 5): cy, cx, ry, rx, match


def fit_affine(cy, cx, ry, rx, c0):
    """r = t + M (p - c0), least squares; returns t (2,), M (2, 2), fitted r (n, 2)."""
    X = np.column_stack([np.ones_like(cy), cy - c0[0], cx - c0[1]])
    by, *_ = np.linalg.lstsq(X, ry, rcond=None)
    bx, *_ = np.linalg.lstsq(X, rx, rcond=None)
    t = np.array([by[0], bx[0]])
    M = np.array([[by[1], by[2]], [bx[1], bx[2]]])
    return t, M, np.column_stack([X @ by, X @ bx])


def one_chunk(args):
    ref, mean, odd, even, d = args
    al = [nd_shift(im, -d, order=1, mode="nearest") for im in (mean, odd, even)]
    b, bo, be = (measure_blocks(ref, im) for im in al)
    return b, np.abs(bo[:, 2:4] - be[:, 2:4]) / 2


def main():
    files = ms.trial_files()
    sh = np.load(ms.OUT_DATA / "shifts.npz")
    ref = sh["ref"]
    with Pool(len(files)) as pool:
        loaded = pool.starmap(load_chunks, [(p, n) for p, _, n in files])
    meta, M_, O_, E_, D_ = [], [], [], [], []
    for i, ((p, start, n), (m, o, e)) in enumerate(zip(files, loaded)):
        for j in range(len(m)):
            f0 = start + j * CHUNK
            meta.append(dict(file=os.path.basename(p), file_index=i, chunk_in_file=j, frame=f0))
            M_.append(m[j]); O_.append(o[j]); E_.append(e[j])
            D_.append(np.array([sh["dy"][f0:f0 + CHUNK].mean(), sh["dx"][f0:f0 + CHUNK].mean()]))
    with Pool(24) as pool:
        res = pool.map(one_chunk, [(ref, m, o, e, d) for m, o, e, d in zip(M_, O_, E_, D_)])

    Ly, Lx = ref.shape
    c0 = np.array([(Ly - 1) / 2, (Lx - 1) / 2])
    win = np.outer(np.hanning(Ly), np.hanning(Lx))
    pref = ms.prep(ref, win)
    brow, arow = [], []
    for k, ((b, prec), mt, img, d) in enumerate(zip(res, meta, M_, D_)):
        ok = (prec.max(1) < MAX_PREC) & (b[:, 4] > MIN_MATCH)
        # robust affine: refit without blocks more than OUTLIER px off the fit
        for _ in range(3):
            cy, cx, ry, rx, _m = b[ok].T
            t, M, fit = fit_affine(cy, cx, ry, rx, c0)
            full = np.column_stack([np.ones(len(b)), b[:, 0] - c0[0], b[:, 1] - c0[1]])
            off = np.hypot(*(b[:, 2:4] - np.column_stack([t[0] + full[:, 1:] @ M[0], t[1] + full[:, 1:] @ M[1]])).T)
            new = ok & (off < OUTLIER)
            if new.sum() < 6 or (new == ok).all():
                break
            ok = new
        cy, cx, ry, rx, _m = b[ok].T
        t, M, fit = fit_affine(cy, cx, ry, rx, c0)
        for q, (cy_, cx_, ry_, rx_, m_) in enumerate(b):
            brow.append(dict(chunk=k, **mt, block=q, cy=cy_, cx=cx_, ry=ry_, rx=rx_, match=m_,
                             prec_y=prec[q, 0], prec_x=prec[q, 1], used=bool(ok[q])))
        r = np.column_stack([ry, rx])
        rms_tr = np.sqrt(np.mean(np.sum((r - r.mean(0)) ** 2, 1)))     # about a pure translation
        rms_aff = np.sqrt(np.mean(np.sum((r - fit) ** 2, 1)))
        rms_noise = np.sqrt(np.mean(np.sum(prec[ok] ** 2, 1)))
        # apply: output(p) = raw(p + d + t + M (p - c0))
        rigid = nd_shift(img, -d, order=1, mode="nearest")
        aff = affine_transform(img, np.eye(2) + M, offset=d + t - M @ c0, order=1, mode="nearest")
        arow.append(dict(chunk=k, **mt, n_blocks=int(ok.sum()),
                         n_matched=int(((prec.max(1) < MAX_PREC) & (b[:, 4] > MIN_MATCH)).sum()),
                         match_median=float(np.median(b[:, 4])), dy=d[0], dx=d[1],
                         t_y=t[0], t_x=t[1],
                         rotation_deg=np.degrees((M[1, 0] - M[0, 1]) / 2),
                         scale_pct=100 * (M[0, 0] + M[1, 1]) / 2,
                         shear=(M[0, 1] + M[1, 0]) / 2, aniso=(M[0, 0] - M[1, 1]) / 2,
                         rms_translation=rms_tr, rms_affine=rms_aff, rms_noise=rms_noise,
                         corr_rigid=ms.corr(ms.prep(rigid, win), pref),
                         corr_affine=ms.corr(ms.prep(aff, win), pref)))
    B, A = pd.DataFrame(brow), pd.DataFrame(arow)
    B.to_csv(_HERE / "results_blocks.csv", index=False)
    A.to_csv(_HERE / "results_affine.csv", index=False)
    pd.set_option("display.width", 200)
    print(A[["file", "chunk_in_file", "n_matched", "n_blocks", "match_median", "rotation_deg", "scale_pct", "shear",
             "rms_translation", "rms_affine", "rms_noise", "corr_rigid", "corr_affine"]].round(3).to_string())
    figure(B, A, ref)
    summarize(B, A)
    print(A[["n_matched", "n_blocks", "match_median", "rotation_deg", "scale_pct", "shear", "aniso", "rms_translation", "rms_affine",
             "rms_noise", "corr_rigid", "corr_affine"]].describe().round(3).to_string())


def summarize(B, A):
    """The numbers quoted in custom_registration.md section 5."""
    B = B.assign(trial=B.file.map(trial))
    A = A.assign(trial=A.file.map(trial))
    matched = (B.match > MIN_MATCH) & (B[["prec_y", "prec_x"]].max(1) < MAX_PREC)
    rows = []
    for k, g in B[matched].groupby("chunk"):
        r = g[["ry", "rx"]].values
        rows.append(dict(chunk=k, rms_matched=np.sqrt(np.mean(np.sum((r - r.mean(0)) ** 2, 1)))))
    A = A.merge(pd.DataFrame(rows), on="chunk")
    A["group"] = np.where(A.trial.str.startswith("no"), "coil off", "coil on")
    A["phase"] = np.where(A.chunk_in_file < EARLY, f"chunks 0-{EARLY - 1}", f"chunks {EARLY}+")
    A["abs_rot"], A["abs_scale"] = A.rotation_deg.abs(), A.scale_pct.abs()
    A["affine_gain"] = A.corr_affine - A.corr_rigid
    print(A.groupby(["group", "phase"])[["match_median", "rms_matched", "rms_noise", "abs_rot", "abs_scale",
                                          "corr_rigid", "affine_gain"]].median().round(3).to_string())
    print(f"  chunks where the affine map raises the correlation by > 0.01: "
          f"{(A.affine_gain > 0.01).sum()} of {len(A)}; lowers it by > 0.01: {(A.affine_gain < -0.01).sum()}")
    # Is the leftover block field the same in the three trials of one acquisition at the same
    # chunk index (chunks of the three conditions are interleaved, so ~3 min apart)?
    out = []
    for grp in ("magneto", "no_magneto"):
        for j in range(B.chunk_in_file.max() + 1):
            F = [B[(B.trial == f"{grp}_{c}") & (B.chunk_in_file == j)].sort_values("block") for c in range(3)]
            if any(len(f) == 0 for f in F):
                continue
            ok = np.all([((f.match > MIN_MATCH) & (f[["prec_y", "prec_x"]].max(1) < MAX_PREC)).values for f in F], 0)
            if ok.sum() < 6:
                continue
            for a, b in ((0, 1), (0, 2), (1, 2)):
                ra, rb = (F[i][["ry", "rx"]].values[ok] for i in (a, b))
                ra, rb = ra - ra.mean(0), rb - rb.mean(0)
                out.append(dict(group="coil on" if grp == "magneto" else "coil off",
                                phase=f"chunks 0-{EARLY - 1}" if j < EARLY else f"chunks {EARLY}+",
                                r=np.corrcoef(ra.ravel(), rb.ravel())[0, 1]))
    print("  correlation of the leftover block fields between trials at the same chunk index:")
    print(pd.DataFrame(out).groupby(["group", "phase"]).r.median().round(2).to_string())


TRIAL_COLORS = {"magneto_0": "#e8590c", "magneto_1": "#f08c4a", "magneto_2": "#f5b98a",
                "no_magneto_0": "#1c5fa8", "no_magneto_1": "#4f8fd6", "no_magneto_2": "#9cc3ee"}
EXAMPLES = [("magneto_0", 1), ("magneto_0", 10)]


def trial(f):
    return f.replace("2022_09_15_fish1_", "").replace(".tif", "")


def figure(B, A, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = A.assign(trial=A.file.map(trial))
    B = B.assign(trial=B.file.map(trial))
    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.3)
    lo, hi = np.percentile(ref, [1, 99.7])
    for i, (tr, j) in enumerate(EXAMPLES):
        ax = fig.add_subplot(gs[0, i])
        a = A[(A.trial == tr) & (A.chunk_in_file == j)].iloc[0]
        b = B[(B.trial == tr) & (B.chunk_in_file == j)]
        ax.imshow(ref, cmap="gray", vmin=lo, vmax=hi)
        for used, col in ((True, "#5ad1ff"), (False, "#ff4d4d")):
            s = b[b.used == used]
            ax.quiver(s.cx, s.cy, s.rx, -s.ry, color=col, angles="xy", scale_units="xy", scale=0.25,
                      width=0.006, label="fits an affine map" if used else "not used (poor match or outlier)")
        for _, r in b.iterrows():
            ax.text(r.cx, r.cy + 14, f"{r.match:.2f}", color="w", fontsize=5.5, ha="center", va="top")
        ax.set_title(f"{tr}, chunk {j}: block match median {a.match_median:.2f},\n"
                     f"blue arrows: spread {a.rms_translation:.1f} px RMS, noise {a.rms_noise:.1f} px", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=6, loc="lower left", framealpha=0.8)
    ax = fig.add_subplot(gs[0, 2])
    for tr, g in A.groupby("trial"):
        ax.plot(g.chunk_in_file, g.match_median, "-o", ms=2.5, lw=1, color=TRIAL_COLORS[tr], label=tr)
    ax.set_xlabel("60-frame chunk of the trial tiff", fontsize=8)
    ax.set_ylabel("median block match (correlation)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    for k, (col, lab) in enumerate((("rotation_deg", "rotation (deg)"), ("scale_pct", "scale change (%)"))):
        ax = fig.add_subplot(gs[1, k])
        for tr, g in A.groupby("trial"):
            ax.plot(g.chunk_in_file, g[col], "-o", ms=2.5, lw=1, color=TRIAL_COLORS[tr])
        ax.axhline(0, color="#8a8a8a", lw=0.6)
        ax.set_xlabel("60-frame chunk of the trial tiff", fontsize=8)
        ax.set_ylabel(lab + " of the affine fit", fontsize=8)
    ax = fig.add_subplot(gs[1, 2])
    for tr, g in A.groupby("trial"):
        ax.scatter(g.corr_rigid, g.corr_affine, s=9, color=TRIAL_COLORS[tr])
    ax.plot([0.1, 0.85], [0.1, 0.85], color="#8a8a8a", lw=0.6)
    ax.set_xlabel("correlation with reference, translation only", fontsize=8)
    ax.set_ylabel("correlation with reference, affine", fontsize=8)
    for ax in fig.axes:
        ax.tick_params(labelsize=7)
    fig.savefig(_HERE / "fig_blocks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        B, A = pd.read_csv(_HERE / "results_blocks.csv"), pd.read_csv(_HERE / "results_affine.csv")
        figure(B, A, np.load(ms.OUT_DATA / "shifts.npz")["ref"])
        summarize(B, A)
    else:
        main()
