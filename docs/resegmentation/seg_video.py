"""Video of one re-segmented ROI: production's segmentation (left) vs an x64 drift-corrected
re-segmentation (right; default the whole-pixel one), on the same crop of the same trial, with the
traces below.

    python docs/resegmentation/seg_video.py --session 2022_09_15-fish1_cytoGCaMP \
        --trial magneto_0 --roi 2056          # suite2p1.0 env, GPU
    python docs/resegmentation/seg_video.py --examples   # every row of results_examples.csv

Left: the frames exactly as production registered them (pipeline/ophys_extraction.py: suite2p's
integer roll and nonrigid warp, bilinear, in float), with production's ROIs. Right: the frames
suite2p read in the re-segmentation (its drift-corrected tiff / 64), with its ROIs. The two crops
show the same tissue: production's coordinates are the raw frame's moved by suite2p's constant
shift (common.prod_offset, rounded). Each video frame is the mean of BIN imaging frames minus the
offset of 100, i.e. mean photons per pixel per frame.

Outlines: red solid = ROIs production's selection keeps (P(iscell), npix, fish outline), light
red dotted = the others; teal = the highlighted re-segmented ROI (right) and the production ROIs
that lie inside it, at least half their pixels (left).

Traces: the re-segmented ROI's (F.npy / 32) and each teal production ROI's (F_full.npy), minus
the offset; below, the drift magnitude that the correction removed.

Checks (asserted): a teal production ROI's trace recomputed from the left frames equals
F_full.npy; the re-segmented ROI's trace recomputed from the right frames equals its F.npy / 32
to within suite2p's halving (1/64 photon).

Output: vid_<session short>_<trial>_roi<roi>[_<tag>].mp4 next to this script (no suffix for the
bilinear reseg_corrected).
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import common as cm  # noqa: E402
ox = cm.ox

FFMPEG = "C:/Users/dan/anaconda3/envs/asoid/Library/bin/ffmpeg.exe"
BIN, FPS = 5, 15
C_TEAL, C_KEEP, C_DROP = "#5ad1ff", "#ff4d4d", "#ffb3b3"
PROD_SHADES = ["#4a4a4a", "#8a8a8a", "#b0b0b0", "#c9a227", "#6b8e23", "#a0522d"]


def selected(session):
    """{trace_set: set of ROI indices production's selection keeps} from analyze.py's table."""
    out = {t: set() for t in cm.TAGS}
    with open(_HERE / "results_sessions_rois.csv") as fh:
        for r in csv.DictReader(fh):
            if r["session"] == session and r["selected"] == "True":
                out[r["trace_set"]].add(int(r["roi"]))
    return out


def make(session, trial, roi, half, tag="reseg_corrected_int"):
    from suite2p.registration import nonrigid
    dev = torch.device("cuda")
    sel = selected(session)
    p_prod, p_new = cm.plane0("production", session), cm.plane0(tag, session)
    ops = ox.load_ops(str(p_prod))
    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
    blocks = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=ops["registration"]["block_size"])
    stat_p = np.load(p_prod / "stat.npy", allow_pickle=True)
    stat_n = np.load(p_new / "stat.npy", allow_pickle=True)
    ic_n = np.load(p_new / "iscell.npy")[:, 1]
    off = cm.offset("production", session)
    tiff = next(f for f, _, _ in ox.trial_files(ops) if f.endswith(f"_{trial}.tif"))
    start, n = cm.trial_slice("production", session, tiff)
    start_n, n_n = cm.trial_slice(tag, session, tiff)
    assert n == n_n

    s = stat_n[roi]
    cy, cx = int(np.median(s["ypix"])), int(np.median(s["xpix"]))
    y0, x0 = min(max(cy - half, 0), Ly - 2 * half), min(max(cx - half, 0), Lx - 2 * half)
    crop_n = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
    py0, px0 = y0 + off[0], x0 + off[1]
    assert 0 <= py0 <= Ly - 2 * half and 0 <= px0 <= Lx - 2 * half, "crop leaves production's frame"
    crop_p = (slice(py0, py0 + 2 * half), slice(px0, px0 + 2 * half))

    # production ROIs inside the highlighted ROI (at least half their pixels)
    inside_new = set(zip(s["ypix"], s["xpix"]))
    teal_p = [j for j, t in enumerate(stat_p)
              if np.mean([(y - off[0], x - off[1]) in inside_new for y, x in zip(t["ypix"], t["xpix"])]) >= 0.5]

    # left: production's registered frames
    raw = tifffile.imread(os.path.join(os.path.dirname(str(ops["file_list"][0])), tiff)).astype(np.float32)
    assert len(raw) == n
    Wp = ox.roi_weights(stat_p, Ly, Lx)
    left, tr_p = [], []
    for i in range(0, n, 300):
        fr = torch.from_numpy(raw[i:i + 300]).to(dev)
        k = len(fr)
        out = torch.stack([torch.roll(f, shifts=(-int(a), -int(b)), dims=(0, 1)) for f, a, b in
                           zip(fr, ops["yoff"][start + i:start + i + k], ops["xoff"][start + i:start + i + k])])
        y1 = torch.from_numpy(np.ascontiguousarray(ops["yoff1"][start + i:start + i + k], np.float32)).to(dev)
        x1 = torch.from_numpy(np.ascontiguousarray(ops["xoff1"][start + i:start + i + k], np.float32)).to(dev)
        r = ox._transform_float(out, blocks[2], blocks[1], blocks[0], y1, x1).cpu().numpy()
        if teal_p:
            tr_p.append(np.asarray(Wp[teal_p[0]] @ r.reshape(k, -1).T).ravel())
        left.append(r[:, crop_p[0], crop_p[1]])
    left = np.concatenate(left)
    F_full = np.load(p_prod / "F_full.npy", mmap_mode="r")
    if teal_p:
        err = float(np.abs(np.concatenate(tr_p) - F_full[teal_p[0], start:start + n]).max())
        assert err < 1e-3, f"production frames do not reproduce F_full.npy (max err {err:.3g})"
        print(f"  check: production ROI {teal_p[0]}'s trace from the left frames equals F_full.npy (max err {err:.2g})")

    # right: the corrected tiff suite2p read
    new_tiff = Path(os.path.dirname(str(ox.load_ops(str(p_new))["file_list"][0]))) / tiff
    mov = tifffile.imread(new_tiff).astype(np.float32) / cm.SCALE
    Wn = ox.roi_weights(stat_n, Ly, Lx)
    tr_n = np.asarray(Wn[roi] @ mov.reshape(n, -1).T).ravel()
    F_n = np.asarray(np.load(p_new / "F.npy", mmap_mode="r")[roi, start_n:start_n + n]) / (cm.SCALE / 2)
    err = float(np.abs(tr_n - F_n).max())
    assert err < 1.5 / cm.SCALE, f"re-segmented frames do not reproduce F.npy / 32 (max err {err:.3g})"
    print(f"  check: ROI {roi}'s trace from the right frames equals F.npy / 32 (max err {err:.2g})")
    right = mov[:, crop_n[0], crop_n[1]]
    del mov, raw

    def masks(st, oy, ox_):
        m = np.zeros((2 * half, 2 * half), bool)
        yy, xx = st["ypix"] - oy, st["xpix"] - ox_
        ok = (yy >= 0) & (yy < 2 * half) & (xx >= 0) & (xx < 2 * half)
        m[yy[ok], xx[ok]] = True
        return m

    def outlines(stat, keep, oy, ox_, teal):
        res = []
        for k, st in enumerate(stat):
            if abs(np.median(st["ypix"]) - oy - half) > half + 25 or abs(np.median(st["xpix"]) - ox_ - half) > half + 25:
                continue
            m = masks(st, oy, ox_)
            if m.any():
                res.append((m, k in keep, k in teal))
        return res

    ol_p = outlines(stat_p, sel["production"], py0, px0, set(teal_p))
    ol_n = outlines(stat_n, sel[tag], y0, x0, {roi})
    nb = n // BIN
    binned = lambda m: m[:nb * BIN].reshape(nb, BIN, *m.shape[1:]).mean(1) - cm.OFFSET
    v0, v1 = binned(left), binned(right)
    vmax = float(np.percentile(np.concatenate([v0, v1]), 99.5))
    sh = np.load(cm.ROOT / session / "shifts.npz")
    disp = np.hypot(sh["dy"][start:start + n], sh["dx"][start:start + n])
    t = np.arange(n) / 60

    fig = plt.figure(figsize=(9.6, 8.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.3, 0.7], hspace=0.42)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axt, axd = fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])
    ims = []
    n_kp = sum(k for _, k, _ in ol_p)
    n_kn = sum(k for _, k, _ in ol_n)
    for ax, v, ol, ttl in ((axs[0], v0, ol_p, f"production: {n_kp} kept ROIs, {len(ol_p) - n_kp} others"),
                           (axs[1], v1, ol_n, f"{cm.LABELS[tag]}: {n_kn} kept ROIs, {len(ol_n) - n_kn} others")):
        ims.append(ax.imshow(v[0], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest"))
        for m, keep, teal in sorted(ol, key=lambda z: z[2]):
            if teal:
                ax.contour(m, levels=[0.5], colors=C_TEAL, linewidths=1.3)
            else:
                ax.contour(m, levels=[0.5], colors=C_KEEP if keep else C_DROP,
                           linewidths=0.7 if keep else 0.45, linestyles="-" if keep else ":")
        ax.set_title(ttl, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(ims[1], ax=axs, fraction=0.025, pad=0.02, label=f"photons / pixel / frame ({BIN}-frame mean)")
    gap = max([float(np.percentile(F_n - cm.OFFSET, 99.5))] +
              [float(np.percentile(np.asarray(F_full[j, start:start + n]) - cm.OFFSET, 99.5))
               for j in teal_p[:len(PROD_SHADES)]]) * 1.1
    axt.plot(t, F_n - cm.OFFSET, color=C_TEAL, lw=0.6,
             label=f"re-segmented ROI {roi}: {int(s['npix'])} px, P(iscell) {ic_n[roi]:.2f}")
    for i, j in enumerate(teal_p[:len(PROD_SHADES)]):
        kept = "kept" if j in sel["production"] else "dropped"
        axt.plot(t, np.asarray(F_full[j, start:start + n]) - cm.OFFSET - gap * (i + 1),
                 color=PROD_SHADES[i], lw=0.5, label=f"production ROI {j}: {int(stat_p[j]['npix'])} px, {kept}")
    if len(teal_p) > len(PROD_SHADES):
        axt.text(0.99, 0.02, f"+{len(teal_p) - len(PROD_SHADES)} more production ROIs inside",
                 transform=axt.transAxes, ha="right", fontsize=6)
    axt.set_ylabel("photons / pixel / frame\n(production ROIs offset)", fontsize=7)
    axt.set_yticks([])
    axt.legend(fontsize=6, frameon=False, loc="upper left", ncol=2)
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
        title.set_text(f"{cm.short(session)} {trial}: teal = re-segmented ROI {roi} and the "
                       f"{len(teal_p)} production ROIs inside it    t = {tk:5.2f} min")
        return ims + cur + [title]

    plt.rcParams["animation.ffmpeg_path"] = FFMPEG
    out = _HERE / (f"vid_{cm.short(session)}_{trial}_roi{roi}"
                   + ("" if tag == "reseg_corrected" else f"_{tag.replace('reseg_', '')}") + ".mp4")
    animation.FuncAnimation(fig, update, frames=nb, blit=False).save(
        out, writer=animation.FFMpegWriter(fps=FPS, bitrate=3000), dpi=110)
    plt.close(fig)
    print(f"  wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB, {nb} frames; "
          f"{len(teal_p)} production ROIs inside ROI {roi})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session")
    ap.add_argument("--trial")
    ap.add_argument("--roi", type=int, help="ROI index in the re-segmentation --tag")
    ap.add_argument("--tag", default="reseg_corrected_int", help="which re-segmentation (common.TAGS)")
    ap.add_argument("--half", type=int, default=48, help="crop half-width, px")
    ap.add_argument("--examples", action="store_true", help="every row of results_examples.csv")
    a = ap.parse_args()
    if a.examples:
        with open(_HERE / "results_examples.csv") as fh:
            for r in csv.DictReader(fh):
                make(r["session"], r["trial"], int(r["roi"]), a.half, r["tag"])
    else:
        make(a.session, a.trial, a.roi, a.half, a.tag)


if __name__ == "__main__":
    main()
