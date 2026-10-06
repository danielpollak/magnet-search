"""Full-resolution ROI traces from the raw imaging tiffs, reusing suite2p's registration and
segmentation.

Why: suite2p (0.10.1 and 0.14.4 for these sessions, and still on main as of v1.1.0) writes its
registered movie as int16: it halves uint16 tiffs (`// 2`) and truncates the registered frames
back to integers. For photon-starved movies (most pixels at the offset in any frame, one level
~ one photon) the two steps erase most of the signal between events and leave traces sitting
at a single value in most frames (docs/cv2.md, section 4).

What: for one experiment's frames, every ROI's trace is extracted twice from the raw tiff,
with suite2p's stored registration shifts (ops yoff/xoff rigid, yoff1/xoff1 nonrigid) and its
ROI weights (stat lam over each ROI's non-overlapping pixels, as allow_overlap=False):
  1. through suite2p's own lossy steps (halving, then truncation after registration). This must
     reproduce suite2p's F.npy to float precision -- the exact-match check -- or extraction
     fails, so a passing run proves the registration below does what suite2p did;
  2. without them, in float32 throughout: the full-resolution trace.
Neuropil is not subtracted (production uses raw F).

The registration transform is suite2p's (0.10-0.14): a rigid integer roll by (-yoff, -xoff),
then a piecewise-bilinear warp whose per-pixel shifts are bilinearly interpolated from the
block shifts. Its bilinear sampler is copied verbatim from suite2p so the lossy path lands on
suite2p's int16 values exactly; the rest is reimplemented (suite2p itself is not a dependency:
0.14 pulls in torch and cellpose).
"""
import os

import numpy as np
import scipy.sparse as sp
import tifffile
from numba import njit, prange

BATCH = 100                     # frames per registration batch
REPRO_RTOL = 1e-5               # exact-match tolerance, relative to the largest |F.npy|
REPRO_ATOL = 1e-4


def load_suite2p(session_path):
    """(ops, stat, F) from session_path/suite2p/plane0."""
    plane = os.path.join(session_path, "suite2p", "plane0")
    ops = np.load(os.path.join(plane, "ops.npy"), allow_pickle=True).item()
    stat = np.load(os.path.join(plane, "stat.npy"), allow_pickle=True)
    F = np.load(os.path.join(plane, "F.npy"), mmap_mode="r")
    return ops, stat, F


def trial_files(ops, tiff_name=None):
    """[(tiff basename, first frame in the suite2p movie, n frames)] for one experiment: the
    named tiff, or every tiff of the session if tiff_name is empty. From suite2p's own file
    list, which fixes the order and frame counts it concatenated."""
    files = [os.path.basename(f.replace("\\", "/")) for f in ops["filelist"]]
    fpf = [int(n) for n in ops["frames_per_file"]]
    starts = np.concatenate([[0], np.cumsum(fpf)[:-1]]).astype(int)
    if tiff_name:
        if tiff_name not in files:
            raise ValueError(f"tiff_name {tiff_name!r} is not in suite2p's file list {files}")
        k = files.index(tiff_name)
        return [(files[k], int(starts[k]), fpf[k])]
    return [(f, int(s), n) for f, s, n in zip(files, starts, fpf)]


def roi_weights(stat, Ly, Lx):
    """Sparse (n_rois, Ly*Lx) matrix of each ROI's normalised lam over its non-overlapping
    pixels: suite2p's extraction weights with allow_overlap=False."""
    rows, cols, vals = [], [], []
    for i, s in enumerate(stat):
        keep = ~s["overlap"]
        lam = s["lam"][keep]
        rows.append(np.full(keep.sum(), i))
        cols.append(s["ypix"][keep] * Lx + s["xpix"][keep])
        vals.append(lam / lam.sum())
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(len(stat), Ly * Lx))


# _map_coordinates is suite2p's registration/nonrigid.py map_coordinates (MouseLand/suite2p,
# GPL-3.0), copied verbatim: its exact float conversions decide where int16 truncation lands,
# and an equivalent rewrite differed from suite2p in ~1e-4 of pixels.
@njit(["(float32[:, :],float32[:,:], float32[:,:], float32[:,:])"], cache=True)
def _map_coordinates(I, yc, xc, Y):
    Ly, Lx = I.shape
    yc_floor = yc.astype(np.int32)
    xc_floor = xc.astype(np.int32)
    yc = yc - yc_floor
    xc = xc - xc_floor
    for i in range(yc_floor.shape[0]):
        for j in range(yc_floor.shape[1]):
            yf = min(Ly - 1, max(0, yc_floor[i, j]))
            xf = min(Lx - 1, max(0, xc_floor[i, j]))
            yf1 = min(Ly - 1, yf + 1)
            xf1 = min(Lx - 1, xf + 1)
            y = yc[i, j]
            x = xc[i, j]
            Y[i, j] = (np.float32(I[yf, xf]) * (1 - y) * (1 - x) + np.float32(I[yf, xf1]) *
                       (1 - y) * x + np.float32(I[yf1, xf]) * y * (1 - x) +
                       np.float32(I[yf1, xf1]) * y * x)


@njit(parallel=True, cache=True)
def _warp(frames, ymax1, xmax1, by, bx, my, mx):
    """suite2p's nonrigid transform_data: per-pixel shifts interpolated from the block shifts
    (block_interp), then each frame sampled at pixel + shift (shift_coordinates)."""
    n, Ly, Lx = frames.shape
    out = np.empty((n, Ly, Lx), np.float32)
    for t in prange(n):
        yup = np.empty((Ly, Lx), np.float32)
        xup = np.empty((Ly, Lx), np.float32)
        _map_coordinates(ymax1[t], by, bx, yup)
        _map_coordinates(xmax1[t], by, bx, xup)
        _map_coordinates(frames[t], my + yup, mx + xup, out[t])
    return out


def _block_centers(ops):
    """(y centres, x centres, (ny, nx)) of suite2p's nonrigid registration blocks: stored in
    ops by 0.10.x (yblock/xblock/nblocks), recomputed from block_size as suite2p's make_blocks
    does otherwise (0.14.x)."""
    if "nblocks" in ops:
        nb = ops["nblocks"]
        yb = np.array(ops["yblock"][::nb[1]]).mean(axis=1)
        xb = np.array(ops["xblock"][:nb[1]]).mean(axis=1)
        return yb, xb, (int(nb[0]), int(nb[1]))
    centres = []
    for L, bs in zip((int(ops["Ly"]), int(ops["Lx"])), ops["block_size"]):
        bs, nb = (L, 1) if bs >= L else (int(bs), int(np.ceil(1.5 * L / bs)))
        start = np.linspace(0, L - bs, nb).astype("int")
        centres.append((start + (start + bs)) / 2)
    return centres[0], centres[1], (len(centres[0]), len(centres[1]))


class Registration:
    """suite2p's stored registration for one session, applied to any batch of frames."""

    def __init__(self, ops):
        if ops.get("bidiphase", 0) and not ops.get("bidi_corrected", False):
            raise NotImplementedError("bidirectional phase correction is not reimplemented")
        self.ops = ops
        self.Ly, self.Lx = int(ops["Ly"]), int(ops["Lx"])
        self.nonrigid = bool(ops.get("nonrigid", False))
        if self.nonrigid:
            yb, xb, self.nblocks = _block_centers(ops)
            iy = np.interp(np.arange(self.Ly), yb, np.arange(yb.size)).astype(np.float32)
            ix = np.interp(np.arange(self.Lx), xb, np.arange(xb.size)).astype(np.float32)
            self.bx, self.by = np.meshgrid(ix, iy)            # block-grid coordinates per pixel
            self.mx, self.my = np.meshgrid(np.arange(self.Lx, dtype=np.float32),
                                           np.arange(self.Ly, dtype=np.float32))

    def __call__(self, frames, first):
        """Register frames (n, Ly, Lx), movie frames first..first+n, to float32."""
        n = len(frames)
        sl = slice(first, first + n)
        out = np.empty(frames.shape, np.float32)
        for i, (dy, dx) in enumerate(zip(self.ops["yoff"][sl], self.ops["xoff"][sl])):
            out[i] = np.roll(frames[i], (-int(dy), -int(dx)), axis=(0, 1))
        if not self.nonrigid:
            return out
        shape = (n,) + self.nblocks
        ymax1 = np.asarray(self.ops["yoff1"][sl], np.float32).reshape(shape)
        xmax1 = np.asarray(self.ops["xoff1"][sl], np.float32).reshape(shape)
        return _warp(out, ymax1, xmax1, self.by, self.bx, self.my, self.mx)


def _suite2p_int16(raw):
    """suite2p's conversion of tiff frames to int16 (io/tiff.py)."""
    if raw.dtype == np.uint16 or raw.dtype == np.int32:
        return (raw // 2).astype(np.int16)
    return raw.astype(np.int16)


def extract(session_path, tiff_name=None, log=print):
    """dict for one experiment's frames, every ROI of the segmentation:
        F_full    (n_rois, n_frames) float32, full-resolution traces, raw tiff units
        F_suite2p (n_rois, n_frames) float32, suite2p's F.npy over the same frames
        max_err   max |lossy-path reproduction - F.npy| (the exact-match check, passed)
    Raises if the lossy path does not reproduce F.npy."""
    ops, stat, F_s2p = load_suite2p(session_path)
    reg = Registration(ops)
    W = roi_weights(stat, reg.Ly, reg.Lx)
    full, repro, s2p = [], [], []
    for name, start, n in trial_files(ops, tiff_name):
        path = os.path.join(session_path, name)
        with tifffile.TiffFile(path) as tf:
            if len(tf.pages) != n:
                raise ValueError(f"{path}: {len(tf.pages)} frames, suite2p used {n}")
            for i in range(0, n, BATCH):
                j = min(i + BATCH, n)
                raw = np.stack([tf.pages[k].asarray() for k in range(i, j)])
                flat = lambda m: m.reshape(len(m), -1).T
                full.append(np.asarray(W @ flat(reg(raw.astype(np.float32), start + i))))
                lossy = reg(_suite2p_int16(raw).astype(np.float32), start + i).astype(np.int16)
                repro.append(np.asarray(W @ flat(lossy.astype(np.float32))))
        s2p.append(np.asarray(F_s2p[:, start:start + n], np.float32))
    F_full = np.concatenate(full, axis=1).astype(np.float32)
    F_repro = np.concatenate(repro, axis=1)
    F_suite2p = np.concatenate(s2p, axis=1)
    err = float(np.abs(F_repro - F_suite2p).max())
    tol = REPRO_RTOL * float(np.abs(F_suite2p).max()) + REPRO_ATOL
    if not err <= tol:
        raise RuntimeError(f"{session_path} {tiff_name}: reproducing suite2p's F.npy failed "
                           f"(max error {err:.3g} > {tol:.3g}); the registration "
                           f"reimplementation does not match this session's suite2p")
    log(f"  exact-match check passed: max error {err:.2g} (tolerance {tol:.2g})")
    return dict(F_full=F_full, F_suite2p=F_suite2p, max_err=err)
