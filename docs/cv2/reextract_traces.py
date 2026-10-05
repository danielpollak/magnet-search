"""Re-extract floor-clipped zebrafish ROI traces from the raw tiffs at full resolution.

Run in the suite2p conda env (it needs suite2p's own registration transform):

    /c/Users/dan/anaconda3/envs/suite2p/python.exe docs/cv2/reextract_traces.py            # the 16 recordings below, ~1 min each
    /c/Users/dan/anaconda3/envs/suite2p/python.exe docs/cv2/reextract_traces.py NAME ...   # some of them

Write-up: docs/cv2.md, section "Where the floor comes from".

suite2p (0.10.1 for these sessions) wrote its registered movie, data.bin, as
    int16( nonrigid_bilinear( rigid_shift( tiff // 2 ) ) )
and extracted F.npy from data.bin. The tiffs are uint16, offset 100, photon-starved (most
pixels sit at 100 in most frames), so the two lossy steps -- integer halving, which merges
100 and 101, and truncation to int16 after the bilinear interpolation, which drops the
fractional counts the interpolation spreads around -- erase most of the signal between events.
This script repeats suite2p's steps on the raw tiff
  1. with the halving and truncation, as a check that it reproduces F.npy exactly, and
  2. without them (float throughout), giving the full-resolution trace,
using the stored shifts (ops yoff/xoff, yoff1/xoff1) and ROI weights (stat lam, overlapping
pixels excluded, as suite2p's allow_overlap=False). Neuropil is not subtracted, as in production.

Output: docs/cv2/reextracted/{name}.npz (gitignored), every ROI of the segmentation:
  F_full  full-resolution trace, raw tiff units (offset 100)
  F_s2p   F.npy's frames for this tiff (suite2p units: raw // 2)
  max_repro_err  max |reproduced suite2p trace - F.npy| (0 up to float error if understood)
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from suite2p.registration import nonrigid, rigid

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
OUT = _HERE / "reextracted"
NAMES = ([f"engert_20220914_fish2_magneto_{i}" for i in range(3)]
         + [f"engert_20220915_fish1_magneto_{i}" for i in range(3)]
         + [f"engert_20221001_fish1_magneto_{i}" for i in range(3)]
         + [f"engert_20221001_fish2_magneto_{i}" for i in range(3)]
         + ["engert_20221002_fish1_magneto_1"]
         + [f"engert_20221002_fish2_magneto_{i}" for i in range(3)])


def yaml_field(name, key):
    text = (_REPO / "experiments" / f"{name}.yml").read_text()
    return re.search(rf'^{key}:\s*"?([^"\n]*)"?', text, re.M).group(1).strip()


def register(frames, ops, sl):
    """suite2p's shift_frames on float frames: rigid integer shift, then nonrigid bilinear."""
    for fr, dy, dx in zip(frames, ops["yoff"][sl], ops["xoff"][sl]):
        fr[:] = rigid.shift_frame(frame=fr, dy=int(dy), dx=int(dx))
    return nonrigid.transform_data(frames, nblocks=ops["nblocks"], xblock=ops["xblock"],
                                   yblock=ops["yblock"], ymax1=ops["yoff1"][sl],
                                   xmax1=ops["xoff1"][sl])


def weights(stat, Lx):
    """(pixel index, weight) per ROI, as suite2p's extraction with allow_overlap=False."""
    out = []
    for s in stat:
        keep = ~s["overlap"]
        lam = s["lam"][keep]
        out.append((s["ypix"][keep] * Lx + s["xpix"][keep], lam / lam.sum()))
    return out


def extract(movie, W):
    X = movie.reshape(len(movie), -1)
    return np.stack([X[:, idx] @ w for idx, w in W]).astype(np.float32)


def run(name):
    ses = yaml_field(name, "session_path")
    tif = yaml_field(name, "tiff_name")
    plane = f"{ses}/suite2p/plane0/"
    ops = np.load(plane + "ops.npy", allow_pickle=True).item()
    assert not ops.get("bidiphase"), "bidirectional phase correction not handled"
    files = [os.path.basename(f.replace("\\", "/")) for f in ops["filelist"]]
    k = files.index(tif)
    off = int(np.sum(ops["frames_per_file"][:k]))
    n = int(ops["frames_per_file"][k])
    sl = slice(off, off + n)
    raw = tifffile.imread(f"{ses}/{tif}")
    assert raw.dtype == np.uint16 and len(raw) == n
    W = weights(np.load(plane + "stat.npy", allow_pickle=True), ops["Lx"])
    F_s2p = np.load(plane + "F.npy", mmap_mode="r")[:, sl].astype(np.float32)
    repro = extract(register((raw // 2).astype(np.float32), ops, sl).astype(np.int16)
                    .astype(np.float32), W)
    err = float(np.abs(repro - F_s2p).max())
    del repro
    F_full = extract(register(raw.astype(np.float32), ops, sl), W)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", F_full=F_full, F_s2p=F_s2p, max_repro_err=err)
    v, c = np.unique(raw[:200], return_counts=True)
    print(f"  {name}: {F_full.shape[0]} ROIs x {n} frames; suite2p reproduced to {err:.2g}; "
          f"raw pixels at {v[c.argmax()]}: {c.max() / c.sum():.3f}", flush=True)


def main():
    for name in sys.argv[1:] or NAMES:
        run(name)


if __name__ == "__main__":
    main()
