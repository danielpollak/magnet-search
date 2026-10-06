"""Full-resolution ROI traces for suite2p v1 (1.1.0) sessions, and the loaders processing
uses to read them.

Why: suite2p (0.10.1 and 0.14.4, and still v1.1.0) writes its registered movie as int16: it
halves uint16 tiffs (`// 2`) and truncates the registered frames back to integers. For
photon-starved movies (most pixels at the offset in any frame, one level ~ one photon) the two
steps erase most of the signal between events and leave traces sitting at a single value in
most frames (docs/cv2.md, section 4).

Two halves, two conda envs:

* `extract_session(plane0)` -- run once per suite2p session, in the `suite2p1.0` env (needs
  torch + suite2p), as `python pipeline/ophys_extraction.py --plane0 <dir> [...]`. It applies
  suite2p's stored registration (rigid roll by ops yoff/xoff, then the nonrigid block warp
  from yoff1/xoff1) to the raw tiff frames on the GPU and extracts every ROI's trace with
  suite2p's ROI weights (stat lam over each ROI's non-overlapping pixels, allow_overlap=False):
    1. through suite2p's own lossy path -- halving, then suite2p's `transform_data`, which
       ends in `.short()`. This must reproduce suite2p's F.npy to float precision (the
       exact-match check) or extraction fails;
    2. without the lossy steps, in float32 throughout: the full-resolution trace. Its warp is
       `transform_data` minus the final `.short()`, and is checked on every batch to give
       suite2p's int16 frames bit for bit once truncated.
  Writes `F_full.npy` (n_rois, n_frames, float32, raw tiff units) and `fullres_check.json`
  next to suite2p's outputs, and `meanImg_full.npy`, the mean of the full-resolution registered
  movie (suite2p's own meanImg is the mean of the truncated movie: in photon-starved sessions it
  is almost flat and the fish outline fails on it). With `--neuropil` it also writes `Fneu_full.npy`, each ROI's
  neuropil trace (suite2p's neuropil masks, unweighted mean), checked against Fneu.npy the
  same way. Production does not subtract neuropil.

* `trial_traces(plane0, tiff_name)` -- used by processing (magneto2 env, numpy only): one
  experiment's frames of F_full.npy and F.npy, located with suite2p's own file list.

The 0.10.1/0.14.4 version of this module (registration reimplemented in numba, extraction at
processing time) is in git history before the switch to suite2p v1 (2026-10-05).
"""
import argparse
import json
import os

import numpy as np

REPRO_RTOL = 1e-5               # exact-match tolerance, relative to the largest |F.npy|
REPRO_ATOL = 1e-4
BATCH = 200                     # frames per GPU batch
FULL_FILE = "F_full.npy"
NEU_FILE = "Fneu_full.npy"
MEAN_FILE = "meanImg_full.npy"
CHECK_FILE = "fullres_check.json"


# ------------------------------------------------------------------ loaders (any env)

def load_ops(plane0):
    return np.load(os.path.join(plane0, "ops.npy"), allow_pickle=True).item()


def trial_files(ops, tiff_name=None):
    """[(tiff basename, first frame in the suite2p movie, n frames)] for one experiment: the
    named tiff, or every tiff of the session if tiff_name is empty. From suite2p's own file
    list, which fixes the order and frame counts it concatenated."""
    files = [os.path.basename(str(f).replace("\\", "/")) for f in ops["file_list"]]
    fpf = [int(n) for n in ops["frames_per_file"]]
    starts = np.concatenate([[0], np.cumsum(fpf)[:-1]]).astype(int)
    if tiff_name:
        if tiff_name not in files:
            raise ValueError(f"tiff_name {tiff_name!r} is not in suite2p's file list {files}")
        k = files.index(tiff_name)
        return [(files[k], int(starts[k]), fpf[k])]
    return [(f, int(s), n) for f, s, n in zip(files, starts, fpf)]


def mean_image(plane0):
    """Mean of the session's full-resolution registered movie (meanImg_full.npy), for the fish
    outline."""
    return np.load(os.path.join(plane0, MEAN_FILE))


def trial_traces(plane0, tiff_name=None):
    """dict for one experiment's frames, every ROI of the segmentation:
        F_full    (n_rois, n_frames) float32, full-resolution traces, raw tiff units
        F_suite2p (n_rois, n_frames) float32, suite2p's F.npy over the same frames
        max_err   the session's exact-match check, max |lossy-path reproduction - F.npy|"""
    path = os.path.join(plane0, FULL_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run `python pipeline/ophys_extraction.py "
                                f"--plane0 {plane0}` in the suite2p1.0 env first")
    with open(os.path.join(plane0, CHECK_FILE)) as f:
        check = json.load(f)
    if not check["passed"]:
        raise RuntimeError(f"{plane0}: the exact-match check did not pass")
    ops = load_ops(plane0)
    F_full = np.load(path, mmap_mode="r")
    F_s2p = np.load(os.path.join(plane0, "F.npy"), mmap_mode="r")
    sl = [slice(s, s + n) for _, s, n in trial_files(ops, tiff_name)]
    cat = lambda F: np.concatenate([np.asarray(F[:, s], np.float32) for s in sl], axis=1)
    return dict(F_full=cat(F_full), F_suite2p=cat(F_s2p), max_err=float(check["max_err"]))


# ------------------------------------------------------------------ extraction (suite2p1.0 env)

def neuropil_weights(stat, ops):
    """Sparse (n_rois, Ly*Lx) matrix averaging each ROI's suite2p neuropil mask, built by
    suite2p's own create_masks with the run's extraction settings."""
    import scipy.sparse as sp
    from suite2p.extraction.masks import create_masks
    Ly, Lx, ex = int(ops["Ly"]), int(ops["Lx"]), ops["extraction"]
    _, nmasks = create_masks(stat, Ly, Lx, lam_percentile=ex["lam_percentile"],
                             allow_overlap=ex["allow_overlap"], neuropil_extract=True,
                             inner_neuropil_radius=ex["inner_neuropil_radius"],
                             min_neuropil_pixels=ex["min_neuropil_pixels"],
                             circular_neuropil=ex["circular_neuropil"])
    rows = np.concatenate([np.full(len(m), i) for i, m in enumerate(nmasks)])
    cols = np.concatenate(nmasks)
    vals = np.concatenate([np.full(len(m), 1.0 / len(m)) for m in nmasks])
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(stat), Ly * Lx))


def roi_weights(stat, Ly, Lx):
    """Sparse (n_rois, Ly*Lx) matrix of each ROI's normalised lam over its non-overlapping
    pixels: suite2p's extraction weights with allow_overlap=False (extraction/masks.py)."""
    import scipy.sparse as sp
    rows, cols, vals = [], [], []
    for i, s in enumerate(stat):
        keep = ~s["overlap"]
        lam = s["lam"][keep]
        rows.append(np.full(keep.sum(), i))
        cols.append(s["ypix"][keep] * Lx + s["xpix"][keep])
        vals.append(lam / lam.sum())
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(len(stat), Ly * Lx))


def _transform_float(data, nblocks, xblock, yblock, ymax1, xmax1):
    """suite2p 1.1.0 registration/nonrigid.py transform_data (MouseLand/suite2p, GPL-3.0),
    CUDA/CPU branch, without the final `.short()` and without the upsampled mean image."""
    import torch
    import torch.nn.functional as F
    n_frames, Ly, Lx = data.shape
    device = data.device
    ymax1 = ymax1.reshape(-1, *nblocks)
    xmax1 = xmax1.reshape(-1, *nblocks)
    mshy, mshx = torch.meshgrid(torch.arange(Ly, dtype=torch.float, device=device),
                                torch.arange(Lx, dtype=torch.float, device=device), indexing="ij")
    yb = np.array(yblock[::nblocks[1]]).mean(axis=1).astype("int")
    xb = np.array(xblock[:nblocks[1]]).mean(axis=1).astype("int")
    Lyc, Lxc = int(yb.max() - yb.min()), int(xb.max() - xb.min())
    yxup = F.interpolate(torch.stack((ymax1, xmax1), dim=1),
                         size=(Lyc, Lxc), mode="bilinear", align_corners=True)
    yxup = F.pad(yxup, (int(xb.min()), Lx - int(xb.max()),
                        int(yb.min()), Ly - int(yb.max())), mode="replicate")
    yxup[:, 0] += mshy
    yxup[:, 1] += mshx
    yxup /= torch.Tensor([Ly - 1, Lx - 1]).to(device).unsqueeze(-1).unsqueeze(-1)
    yxup *= 2
    yxup -= 1
    yxup = yxup.permute(0, 2, 3, 1)
    fr_shift = F.grid_sample(data.float().unsqueeze(1), yxup[:, :, :, [1, 0]],
                             mode="bilinear", padding_mode="border", align_corners=True)
    return fr_shift.squeeze(1)


def extract_session(plane0, device="cuda", log=print, neuropil=False):
    """Write F_full.npy (and Fneu_full.npy if neuropil) and fullres_check.json for one suite2p
    v1 session; returns the check."""
    import tifffile
    import torch
    import suite2p
    from suite2p.registration import nonrigid

    ops = load_ops(plane0)
    stat = np.load(os.path.join(plane0, "stat.npy"), allow_pickle=True)
    F_s2p = np.load(os.path.join(plane0, "F.npy"), mmap_mode="r")
    reg = ops["registration"]
    if ops.get("bidiphase", 0) or reg.get("two_step_registration") or ops.get("nchannels", 1) != 1:
        raise NotImplementedError(f"{plane0}: bidiphase / two-step / 2-channel registration")
    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
    W = roi_weights(stat, Ly, Lx)
    Wn = neuropil_weights(stat, ops) if neuropil else None
    blocks = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=reg["block_size"]) if reg["nonrigid"] else None
    dev = torch.device(device)
    tiff_dir = [os.path.dirname(str(f)) for f in ops["file_list"]][0]

    def register(frames, first, lossy):
        """frames (n, Ly, Lx) on dev -> registered frames, as suite2p's shift_frames does."""
        n = len(frames)
        yoff, xoff = ops["yoff"][first:first + n], ops["xoff"][first:first + n]
        out = torch.stack([torch.roll(f, shifts=(-int(dy), -int(dx)), dims=(0, 1))
                           for f, dy, dx in zip(frames, yoff, xoff)], dim=0)
        if blocks is None:
            return out
        y1 = torch.from_numpy(np.ascontiguousarray(ops["yoff1"][first:first + n])).to(dev)
        x1 = torch.from_numpy(np.ascontiguousarray(ops["xoff1"][first:first + n])).to(dev)
        if lossy:
            return nonrigid.transform_data(out, blocks[2], blocks[1], blocks[0], y1, x1)
        return _transform_float(out, blocks[2], blocks[1], blocks[0], y1, x1)

    flat = lambda m: m.reshape(len(m), -1).T
    full, repro, nfull, nrepro = [], [], [], []
    mean_sum = np.zeros(Ly * Lx, np.float64)
    n_copy_checked = 0
    for name, start, n in trial_files(ops):
        path = os.path.join(tiff_dir, name)
        with tifffile.TiffFile(path) as tf:
            if len(tf.pages) != n:
                raise ValueError(f"{path}: {len(tf.pages)} frames, suite2p used {n}")
            for i in range(0, n, BATCH):
                j = min(i + BATCH, n)
                raw = np.stack([tf.pages[k].asarray() for k in range(i, j)])
                if raw.dtype != np.uint16:
                    raise NotImplementedError(f"{path}: {raw.dtype} tiff (expected uint16)")
                half = torch.from_numpy((raw // 2).astype(np.int16)).to(dev)
                lossy = register(half, start + i, lossy=True)
                if blocks is not None:      # the float copy is suite2p's warp, bit for bit
                    if not torch.equal(register(half, start + i, lossy=False).short(), lossy):
                        raise RuntimeError(f"{path}: the float warp differs from suite2p's "
                                           f"transform_data (frames {i}-{j})")
                    n_copy_checked += j - i
                fr = flat(register(torch.from_numpy(raw.astype(np.float32)).to(dev), start + i,
                                   lossy=False).cpu().numpy())
                mean_sum += fr.sum(axis=1)
                lo = flat(lossy.cpu().numpy().astype(np.float32))
                full.append(np.asarray(W @ fr, np.float32))
                repro.append(np.asarray(W @ lo))
                if neuropil:
                    nfull.append(np.asarray(Wn @ fr, np.float32))
                    nrepro.append(np.asarray(Wn @ lo))
        log(f"  {name}: {n} frames")
    F_full = np.concatenate(full, axis=1).astype(np.float32)
    F_repro = np.concatenate(repro, axis=1)
    F_ref = np.asarray(F_s2p, np.float32)
    if F_full.shape != F_ref.shape:
        raise RuntimeError(f"{plane0}: F_full {F_full.shape} vs F.npy {F_ref.shape}")
    err = float(np.abs(F_repro - F_ref).max())
    tol = REPRO_RTOL * float(np.abs(F_ref).max()) + REPRO_ATOL
    if neuropil:
        N_ref = np.load(os.path.join(plane0, "Fneu.npy")).astype(np.float32)
        nerr = float(np.abs(np.concatenate(nrepro, axis=1) - N_ref).max())
        ntol = REPRO_RTOL * float(np.abs(N_ref).max()) + REPRO_ATOL
        if not nerr <= ntol:
            raise RuntimeError(f"{plane0}: reproducing suite2p's Fneu.npy failed "
                               f"(max error {nerr:.3g} > {ntol:.3g})")
        np.save(os.path.join(plane0, NEU_FILE), np.concatenate(nfull, axis=1).astype(np.float32))
        log(f"  neuropil check passed: max error {nerr:.2g} (tolerance {ntol:.2g}) -> {NEU_FILE}")
    check = dict(passed=bool(err <= tol), max_err=err, tolerance=tol,
                 frames_float_warp_checked=n_copy_checked, n_rois=len(stat),
                 n_frames=int(F_full.shape[1]), suite2p_version=suite2p.version,
                 torch_version=torch.__version__, device=str(dev))
    if neuropil:
        check.update(neuropil_max_err=nerr, neuropil_tolerance=ntol)
    with open(os.path.join(plane0, CHECK_FILE), "w") as f:
        json.dump(check, f, indent=1)
    if not check["passed"]:
        raise RuntimeError(f"{plane0}: reproducing suite2p's F.npy failed "
                           f"(max error {err:.3g} > {tol:.3g})")
    np.save(os.path.join(plane0, FULL_FILE), F_full)
    np.save(os.path.join(plane0, MEAN_FILE), (mean_sum / F_full.shape[1]).reshape(Ly, Lx).astype(np.float32))
    log(f"  exact-match check passed: max error {err:.2g} (tolerance {tol:.2g}) -> {FULL_FILE}")
    return check


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plane0", nargs="+", required=True, help="suite2p v1 plane0 directories")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--neuropil", action="store_true", help="also write Fneu_full.npy")
    a = ap.parse_args()
    for p in a.plane0:
        print(p, flush=True)
        extract_session(p, a.device, log=lambda m: print(m, flush=True), neuropil=a.neuropil)
