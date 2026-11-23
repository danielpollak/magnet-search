"""Measure lateral drift in the fish whose motion suite2p corrected, with the same method
used for the five photon-starved zebrafish (docs/custom_registration/measure.py), and put
suite2p's own rigid shifts next to it.

    python docs/drift_all_fish/measure_all.py                 # magneto2 env, all five runs
    python docs/drift_all_fish/measure_all.py --session 2022_02_23

Method (as measure.py): average the raw frames in bins of 10, cross-correlate each bin with a
two-pass session reference (smoothing 1.5 px, background 20 px, Hann taper, peak within
+-40 px, parabolic sub-pixel fit). Odd and even frames of each bin are measured separately;
half their difference is the bin's precision. Bins never cross a recording, nor a 60-frame
chunk in the stitched trial tiffs (2022-10-02 fish1, medaka). The 2022 Q1 recordings are
continuous, so their only boundary is the recording itself.

Reads the local tiff copies in C:/Users/dan/suite2p_v1_production/<run>/tiffs (no NAS read)
and suite2p's ops.npy for its rigid shifts (yoff/xoff), averaged over the same bins.

Outputs:
    docs/drift_all_fish/results_bins_<run>.csv      one row per bin
    C:/Users/dan/suite2p_v1_custom_reg/drift_all/<run>.npz
                                                    reference image, the per-recording first-
                                                    minute / settled means, for the figures
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import shift as nd_shift

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "custom_registration"))
sys.path.insert(0, str(_HERE.parents[1]))
import measure as M                                            # noqa: E402  prep, xcorr_shift, ...

ROOT = "C:/Users/dan/suite2p_v1_production"
OUT_DATA = Path("C:/Users/dan/suite2p_v1_custom_reg/drift_all")
BIN = M.BIN

# run -> (tiffs to measure, or None for all of suite2p's; chunk length in frames, or None)
SESSIONS = {
    "2022_02_21": (None, None),
    "2022_02_23": (None, None),
    "2022_03_01": (None, None),
    # magneto_2/3 and no-magneto_2/3 are byte-identical copies of _1 (CLAUDE.md)
    "2022_10_02-fish1_cytoGCaMP": (["magneto_1.tif", "no-magneto_1.tif"], 60),
    "fish3_8dpf_concat_x64": (None, 60),
}


def load_bins(path, n):
    """Mean, odd-frame mean and even-frame mean of each BIN-frame bin of one tiff."""
    mov = tifffile.imread(path)
    assert len(mov) == n, f"{path}: {len(mov)} frames, suite2p used {n}"
    nb = n // BIN
    b = mov[:nb * BIN].reshape(nb, BIN, *mov.shape[1:])
    out = (b.mean(1, dtype=np.float64).astype(np.float32),
           b[:, 0::2].mean(1, dtype=np.float64).astype(np.float32),
           b[:, 1::2].mean(1, dtype=np.float64).astype(np.float32))
    del mov
    return out


def run(session):
    from pipeline import ophys_extraction as ox
    names, chunk = SESSIONS[session]
    plane0 = f"{ROOT}/{session}/output/suite2p/plane0"
    ops = ox.load_ops(plane0)
    files = [f for f in ox.trial_files(ops) if names is None or f[0] in names]
    tiff_dir = f"{ROOT}/{session}/tiffs"
    print(f"== {session}: {[f[0] for f in files]}", flush=True)
    with Pool(len(files)) as pool:
        loaded = pool.starmap(load_bins, [(os.path.join(tiff_dir, f), n) for f, _, n in files])
    B = np.concatenate([x[0] for x in loaded])
    Bodd = np.concatenate([x[1] for x in loaded])
    Beven = np.concatenate([x[2] for x in loaded])
    del loaded

    # bins, in the order of `files`; frame = first frame in suite2p's concatenated movie
    rows = []
    for i, (f, start, n) in enumerate(files):
        for j in range(n // BIN):
            c = None if chunk is None else (j * BIN) // chunk
            rows.append(dict(file=f, file_index=i, bin_in_file=j, frame_in_file=j * BIN,
                             frame=start + j * BIN, chunk=c))
    meta = pd.DataFrame(rows)

    Ly, Lx = B.shape[1:]
    win = np.outer(np.hanning(Ly), np.hanning(Lx))
    d1 = M.measure(B.mean(0), B, win)
    ref = np.mean([nd_shift(b, -d, order=1, mode="nearest") for b, d in zip(B, d1)], axis=0)
    d = M.measure(ref, B, win)
    d_odd, d_even = M.measure(ref, Bodd, win), M.measure(ref, Beven, win)
    prec = np.abs(d_odd - d_even) / 2

    pref = M.prep(ref, win)
    c_before, c_after = [], []
    for b, dd in zip(B, d):
        bs = nd_shift(b, -dd, order=1, mode="nearest")
        c_before.append(M.corr(M.prep(b, win), pref))
        c_after.append(M.corr(M.prep(bs, win), pref))

    # suite2p's rigid shifts over the same bins
    yoff, xoff = np.asarray(ops["yoff"], float), np.asarray(ops["xoff"], float)
    s2p = np.array([[yoff[f:f + BIN].mean(), xoff[f:f + BIN].mean(),
                     yoff[f:f + BIN].std(), xoff[f:f + BIN].std(),
                     np.hypot(yoff[f:f + BIN] - np.median(yoff), xoff[f:f + BIN] - np.median(xoff)).max()]
                    for f in meta.frame.values])

    out = meta.assign(dy=d[:, 0], dx=d[:, 1], disp=np.hypot(*d.T),
                      prec_dy=prec[:, 0], prec_dx=prec[:, 1],
                      corr_before=c_before, corr_after=c_after,
                      s2p_yoff=s2p[:, 0], s2p_xoff=s2p[:, 1],
                      s2p_yoff_sd=s2p[:, 2], s2p_xoff_sd=s2p[:, 3], s2p_max_frame_disp=s2p[:, 4])
    out.to_csv(_HERE / f"results_bins_{session}.csv", index=False)

    # images for the figures: per recording, first minute and the rest, raw and corrected
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    imgs = dict(ref=ref)
    for i, (f, _, n) in enumerate(files):
        sel = out.file_index.values == i
        first = sel & (out.frame_in_file.values < 60)
        later = sel & (out.frame_in_file.values >= 600)
        if not later.any():
            later = sel & (out.frame_in_file.values >= 60)
        for tag, s in (("first", first), ("later", later)):
            imgs[f"{i}_{tag}_raw"] = B[s].mean(0)
            imgs[f"{i}_{tag}_corr"] = np.mean(
                [nd_shift(b, -dd, order=1, mode="nearest") for b, dd in zip(B[s], d[s])], 0)
    np.savez_compressed(OUT_DATA / f"{session}.npz", **imgs)
    print(f"   {len(out)} bins; displacement max {out.disp.max():.2f} px; precision median "
          f"{np.median(prec):.3f} px; corr with reference median {np.median(c_before):.3f} -> "
          f"{np.median(c_after):.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", choices=list(SESSIONS), action="append")
    a = ap.parse_args()
    for s in a.session or list(SESSIONS):
        run(s)
