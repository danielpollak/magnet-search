"""Re-extract 20220915_fish1's ROI traces with the measured per-frame displacement added to
suite2p's own (constant) registration.

    python docs/custom_registration/extract.py          # suite2p1.0 env, GPU, ~2 min

Production (pipeline/ophys_extraction.py) registers each raw frame exactly as suite2p did: an
integer roll by suite2p's rigid shift (yoff, xoff), then suite2p's nonrigid warp with its block
shifts (yoff1, xoff1), bilinear, in float. In this session suite2p's shifts are the same on every
frame, so the movie is not motion corrected. Here the same warp is used with the block shifts
replaced by yoff1 + dy(t), xoff1 + dx(t), where (dy, dx) is the displacement measured by
measure.py. The ROIs (stat.npy) and their weights are production's, unchanged.

Control: with dy = dx = 0 this code must reproduce production's F_full.npy (asserted). That checks
the whole path, frame order, ROI weights and warp, end to end.

Output (local disk, not the repo):
    C:/Users/dan/suite2p_v1_custom_reg/2022_09_15-fish1_cytoGCaMP/F_custom.npy
        (n_rois, n_frames) float32, same layout as F_full.npy
"""
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
from pipeline import ophys_extraction as ox  # noqa: E402

SESSION = "2022_09_15-fish1_cytoGCaMP"
PLANE0 = f"C:/Users/dan/suite2p_v1_production/{SESSION}/output/suite2p/plane0"
OUT_DATA = Path(f"C:/Users/dan/suite2p_v1_custom_reg/{SESSION}")
BATCH = 300


def main():
    from suite2p.registration import nonrigid
    dev = torch.device("cuda")
    ops = ox.load_ops(PLANE0)
    stat = np.load(os.path.join(PLANE0, "stat.npy"), allow_pickle=True)
    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
    W = ox.roi_weights(stat, Ly, Lx)
    reg = ops["registration"]
    blocks = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=reg["block_size"])
    sh = np.load(OUT_DATA / "shifts.npz")
    dy, dx = sh["dy"], sh["dx"]
    tiff_dir = os.path.dirname(str(ops["file_list"][0]))

    def warp(frames, first, add):
        n = len(frames)
        yoff, xoff = ops["yoff"][first:first + n], ops["xoff"][first:first + n]
        out = torch.stack([torch.roll(f, shifts=(-int(a), -int(b)), dims=(0, 1))
                           for f, a, b in zip(frames, yoff, xoff)], dim=0)
        y1 = np.ascontiguousarray(ops["yoff1"][first:first + n], np.float32)
        x1 = np.ascontiguousarray(ops["xoff1"][first:first + n], np.float32)
        if add:
            y1 = y1 + dy[first:first + n, None].astype(np.float32)
            x1 = x1 + dx[first:first + n, None].astype(np.float32)
        y1, x1 = torch.from_numpy(y1).to(dev), torch.from_numpy(x1).to(dev)
        return ox._transform_float(out, blocks[2], blocks[1], blocks[0], y1, x1)

    flat = lambda m: m.reshape(len(m), -1).T
    F0, F1 = [], []
    img = {k: np.zeros((Ly, Lx)) for k in ("first_min_s2p", "first_min_custom", "settled")}
    for name, start, n in ox.trial_files(ops):
        with tifffile.TiffFile(os.path.join(tiff_dir, name)) as tf:
            for i in range(0, n, BATCH):
                j = min(i + BATCH, n)
                raw = np.stack([tf.pages[k].asarray() for k in range(i, j)]).astype(np.float32)
                fr = torch.from_numpy(raw).to(dev)
                r0 = warp(fr, start + i, False).cpu().numpy()
                r1 = warp(fr, start + i, True).cpu().numpy()
                F0.append(np.asarray(W @ flat(r0), np.float32))
                F1.append(np.asarray(W @ flat(r1), np.float32))
                if start == 0:      # magneto_0: its first minute, and minutes 10-20 (settled)
                    fi = np.arange(i, j)
                    img["first_min_s2p"] += r0[fi < 60].sum(0)
                    img["first_min_custom"] += r1[fi < 60].sum(0)
                    img["settled"] += r0[fi >= 600].sum(0)
        print(f"  {name}: {n} frames", flush=True)
    F0, F1 = np.concatenate(F0, 1), np.concatenate(F1, 1)
    F_full = np.load(os.path.join(PLANE0, "F_full.npy"))
    err = float(np.abs(F0 - F_full).max())
    assert err < 1e-3, f"zero-displacement path does not reproduce F_full.npy (max err {err:.3g})"
    print(f"  control: zero displacement reproduces production F_full.npy (max err {err:.2g})")
    np.save(OUT_DATA / "F_custom.npy", F1)
    np.savez_compressed(OUT_DATA / "registered_means.npz", first_min_s2p=img["first_min_s2p"] / 60,
                        first_min_custom=img["first_min_custom"] / 60,
                        settled=img["settled"] / (1260 - 600), meanImg=ops["meanImg"])
    print(f"  wrote {OUT_DATA / 'F_custom.npy'} {F1.shape}")


if __name__ == "__main__":
    main()
