"""Video of a small crop of 20220915_fish1 magneto_0 around one ROI, with its mask outlined, as
suite2p registered it (left) and with the custom drift correction (right), above the ROI's trace.

    python docs/custom_registration/roi_video.py [--roi 58] [--trial magneto_0]   # suite2p1.0 env, GPU

Frames are registered exactly as production extracts them (pipeline/ophys_extraction.py: suite2p's
integer roll, then suite2p's nonrigid warp, bilinear, in float); the custom version adds the
drift measured by measure.py. Each video frame is the mean of BIN imaging frames, minus the
camera offset of 100, so a pixel's value is its mean photon count. The outline marks the ROI's
mask; the pixels production actually uses (the ones not shared with another ROI) are filled.

Check: the ROI's trace recomputed from these frames must equal production's F_full.npy row.

Output: roi<roi>_<trial>.mp4 next to this script (H.264, a 48 x 48 px crop, small).
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import tifffile
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
from pipeline import ophys_extraction as ox  # noqa: E402

SESSION = "2022_09_15-fish1_cytoGCaMP"
PLANE0 = f"C:/Users/dan/suite2p_v1_production/{SESSION}/output/suite2p/plane0"
DATA = Path(f"C:/Users/dan/suite2p_v1_custom_reg/{SESSION}")
FFMPEG = "C:/Users/dan/anaconda3/envs/asoid/Library/bin/ffmpeg.exe"
BIN = 5            # imaging frames per video frame
FPS = 15
OFFSET = 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", type=int, default=58)
    ap.add_argument("--trial", default="magneto_0")
    ap.add_argument("--half", type=int, default=24, help="crop half-width, px")
    ap.add_argument("--others", action="store_true",
                    help="also outline every other production ROI in the crop, in red")
    a = ap.parse_args()
    HALF = a.half
    from suite2p.registration import nonrigid
    dev = torch.device("cuda")
    ops = ox.load_ops(PLANE0)
    stat = np.load(os.path.join(PLANE0, "stat.npy"), allow_pickle=True)
    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
    blocks = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=ops["registration"]["block_size"])
    sh = np.load(DATA / "shifts.npz")
    tiff = f"2022_09_15_fish1_{a.trial}.tif"
    start, n = next((s, m) for f, s, m in ox.trial_files(ops) if f == tiff)
    s = stat[a.roi]
    cy, cx = int(np.median(s["ypix"])), int(np.median(s["xpix"]))
    y0, x0 = max(cy - HALF, 0), max(cx - HALF, 0)
    crop = (slice(y0, y0 + 2 * HALF), slice(x0, x0 + 2 * HALF))

    def warp(frames, first, add):
        k = len(frames)
        out = torch.stack([torch.roll(f, shifts=(-int(p), -int(q)), dims=(0, 1)) for f, p, q in
                           zip(frames, ops["yoff"][first:first + k], ops["xoff"][first:first + k])])
        y1 = np.ascontiguousarray(ops["yoff1"][first:first + k], np.float32)
        x1 = np.ascontiguousarray(ops["xoff1"][first:first + k], np.float32)
        if add:
            y1 = y1 + sh["dy"][first:first + k, None].astype(np.float32)
            x1 = x1 + sh["dx"][first:first + k, None].astype(np.float32)
        return ox._transform_float(out, blocks[2], blocks[1], blocks[0],
                                   torch.from_numpy(y1).to(dev), torch.from_numpy(x1).to(dev)).cpu().numpy()

    W = ox.roi_weights(stat, Ly, Lx)[a.roi]
    mov = tifffile.imread(os.path.join(os.path.dirname(str(ops["file_list"][0])), tiff)).astype(np.float32)
    assert len(mov) == n
    raw_min = float(mov.min())
    s2p, cus, tr_s2p, tr_cus = [], [], [], []
    for i in range(0, n, 300):
        fr = torch.from_numpy(mov[i:i + 300]).to(dev)
        r0, r1 = warp(fr, start + i, False), warp(fr, start + i, True)
        tr_s2p.append(np.asarray(W @ r0.reshape(len(r0), -1).T).ravel())
        tr_cus.append(np.asarray(W @ r1.reshape(len(r1), -1).T).ravel())
        s2p.append(r0[:, crop[0], crop[1]])
        cus.append(r1[:, crop[0], crop[1]])
    s2p, cus = np.concatenate(s2p), np.concatenate(cus)
    tr_s2p, tr_cus = np.concatenate(tr_s2p), np.concatenate(tr_cus)
    F_full = np.load(os.path.join(PLANE0, "F_full.npy"), mmap_mode="r")[a.roi, start:start + n]
    err = float(np.abs(tr_s2p - F_full).max())
    assert err < 1e-3, f"recomputed trace differs from F_full.npy by {err}"
    print(f"  check: ROI {a.roi}'s trace from these frames equals F_full.npy (max err {err:.2g})")
    print(f"  raw tiff minimum {raw_min:.0f}; ROI {a.roi}: {len(s['ypix'])} pixels, "
          f"{int((~s['overlap']).sum())} used; frames at exactly {OFFSET:.0f}: "
          f"{np.mean(F_full == OFFSET):.1%}")

    nb = n // BIN
    binned = lambda m: m[:nb * BIN].reshape(nb, BIN, *m.shape[1:]).mean(1) - OFFSET
    v0, v1 = binned(s2p), binned(cus)
    vmax = float(np.percentile(np.concatenate([v0, v1]), 99.5))
    def masks(st):
        m = np.zeros((2 * HALF, 2 * HALF), bool)
        u = np.zeros_like(m)
        for yy, xx, ov in zip(st["ypix"], st["xpix"], st["overlap"]):
            if y0 <= yy < y0 + 2 * HALF and x0 <= xx < x0 + 2 * HALF:
                m[yy - y0, xx - x0] = True
                u[yy - y0, xx - x0] = not ov
        return m, u
    mask, used = masks(s)
    others = []
    if a.others:
        # production's ROI set for this trial (P(iscell), npix, outline, flatline removal), as
        # analyze.py recorded it
        import csv      # no pandas in the suite2p1.0 env
        with open(_HERE / "results_rois.csv") as fh:
            keep = [int(row["roi"]) for row in csv.DictReader(fh)
                    if row["experiment"] == f"engert_20220915_fish1_{a.trial}"
                    and row["trace_set"] == "suite2p"]
        keep = set(keep)
        for k in range(len(stat)):
            if k != a.roi:
                m, _ = masks(stat[k])
                if m.any():
                    others.append((m, k in keep))
        print(f"  other ROIs in the crop: {sum(p for _, p in others)} in production's set, "
              f"{sum(not p for _, p in others)} that suite2p found but production drops")
    t = np.arange(n) / 60
    disp = np.hypot(sh["dy"][start:start + n], sh["dx"][start:start + n])

    fig = plt.figure(figsize=(7.2 * max(1.0, HALF / 24) ** 0.6, 6.2 * max(1.0, HALF / 24) ** 0.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.1, 0.8], hspace=0.45)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axt, axd = fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])
    ims = []
    for ax, v, ttl in ((axs[0], v0, "suite2p's registration (production)"),
                       (axs[1], v1, "custom drift correction")):
        ims.append(ax.imshow(v[0], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest"))
        for m, prod in others:
            ax.contour(m, levels=[0.5], colors="#ff4d4d" if prod else "#ffb3b3",
                       linewidths=0.7 if prod else 0.5, linestyles="-" if prod else ":")
        ax.contour(mask, levels=[0.5], colors="#5ad1ff", linewidths=1.0)
        ax.contour(used, levels=[0.5], colors="#5ad1ff", linewidths=0.6, linestyles=":")
        ax.set_title(ttl, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(ims[1], ax=axs, fraction=0.025, pad=0.02, label=f"mean photons / frame ({BIN}-frame mean)")
    axt.plot(t, F_full - OFFSET, color="#8a8a8a", lw=0.6, label="suite2p (production trace)")
    axt.plot(t, tr_cus - OFFSET, color="#2a78d6", lw=0.6, alpha=0.8, label="custom")
    axt.set_ylabel(f"ROI {a.roi}\nF - {OFFSET:.0f}", fontsize=7)
    axt.legend(fontsize=6, frameon=False, loc="upper left")
    axd.plot(t, disp, color="#e8590c", lw=0.8)
    axd.set_ylabel("drift (px)", fontsize=7)
    axd.set_xlabel("time in the trial tiff (min)", fontsize=7)
    cur = [ax.axvline(0, color="k", lw=0.8) for ax in (axt, axd)]
    for ax in (axt, axd):
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, t[-1])
    title = fig.suptitle("", fontsize=9)

    def update(k):
        ims[0].set_data(v0[k])
        ims[1].set_data(v1[k])
        tk = (k * BIN + BIN / 2) / 60
        for c in cur:
            c.set_xdata([tk, tk])
        title.set_text(f"20220915_fish1 {a.trial}: ROI {a.roi} teal (dotted = pixels used)"
                       + (", other ROIs red (dotted: dropped by production)" if others else "") + "   "
                       f"t = {tk:5.2f} min")
        return ims + cur + [title]

    plt.rcParams["animation.ffmpeg_path"] = FFMPEG
    out = _HERE / (f"roi{a.roi}_{a.trial}" + (f"_crop{2 * HALF}" if HALF != 24 else "")
                   + ("_all_rois" if a.others else "") + ".mp4")
    animation.FuncAnimation(fig, update, frames=nb, blit=False).save(
        out, writer=animation.FFMpegWriter(fps=FPS, bitrate=3000), dpi=110)
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {nb} frames at {FPS} fps)")


if __name__ == "__main__":
    main()
