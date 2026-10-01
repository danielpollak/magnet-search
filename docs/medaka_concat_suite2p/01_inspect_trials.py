"""Inventory the six medaka fish3_8dpf trial directories on the NAS (READ-ONLY)
and estimate rigid inter-trial shifts from the ORIGINAL per-trial suite2p
mean images.

ENV:  magneto2 (numpy/scipy/tifffile/matplotlib)
NAS:  required (reads raw tiff headers + first/last page, and each trial's
      original suite2p/plane0/{ops,stat,iscell,F}.npy). ~1 min.

Outputs (next to this script):
  trial_inventory.csv         per-trial frames / dims / ROI counts / registration offsets
  intertrial_shifts_nas.csv   phase-correlation shift of every trial's meanImg vs every other
  fig_meanimg_nas.png         the six original mean images + overlay vs trial 0
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from common import (HERE, TRIALS, SHORT, nas_tiff, nas_plane0, load_plane0,
                    phase_corr_shift, ISCELL_THRES, NPIX_THRES)
from magpyneto2.engert_helpers import remove_flatlines


def norm(im):
    lo, hi = np.percentile(im, [1, 99.5])
    return np.clip((im - lo) / (hi - lo), 0, 1)


def main():
    rows, means = [], {}
    for t in TRIALS:
        with tifffile.TiffFile(nas_tiff(t)) as tf:
            npages = len(tf.pages)
            p0 = tf.pages[0].asarray(); pl = tf.pages[-1].asarray()
            ij = tf.imagej_metadata or {}
        d = load_plane0(nas_plane0(t))
        ops, stat, isc, F = d["ops"], d["stat"], d["iscell"], np.asarray(d["F"])
        npix = np.array([s["npix"] for s in stat])
        m = (isc[:, 1] > ISCELL_THRES) & (npix > NPIX_THRES)
        _, _, _, keep = remove_flatlines(F[m])
        means[t] = ops["meanImg"]
        rows.append({
            "trial": t, "tiff_pages": npages, "tiff_shape": f"{p0.shape}", "tiff_dtype": str(p0.dtype),
            "tiff_first_page_mean": float(p0.mean()), "tiff_last_page_mean": float(pl.mean()),
            "imagej_version": ij.get("ImageJ", ""),
            "s2p_version": ops["suite2p_version"], "s2p_nframes": ops["nframes"],
            "Ly": ops["Ly"], "Lx": ops["Lx"], "fs": ops["fs"], "tau": ops["tau"],
            "n_roi": len(stat), "n_iscell_npix": int(m.sum()), "n_included": len(keep),
            "xoff_absmax": int(np.abs(ops["xoff"]).max()), "yoff_absmax": int(np.abs(ops["yoff"]).max()),
            "xoff1_absmax": float(np.abs(ops["xoff1"]).max()), "yoff1_absmax": float(np.abs(ops["yoff1"]).max()),
            "corrXY_mean": float(np.mean(ops["corrXY"])),
        })
    with open(HERE / "trial_inventory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)

    sh = []
    for a in TRIALS:
        for b in TRIALS:
            dy, dx, pk = phase_corr_shift(means[a], means[b])
            r = np.corrcoef(means[a].ravel(), means[b].ravel())[0, 1]
            sh.append({"ref": a, "moving": b, "dy": round(dy, 2), "dx": round(dx, 2),
                       "peak": round(pk, 4), "meanimg_pearson_r": round(r, 4)})
    with open(HERE / "intertrial_shifts_nas.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sh[0])); w.writeheader(); w.writerows(sh)
    for s in sh:
        if s["ref"] == TRIALS[0]:
            print(s)

    fig, axs = plt.subplots(2, 6, figsize=(18, 6.5))
    ref = norm(means[TRIALS[0]])
    for i, t in enumerate(TRIALS):
        axs[0, i].imshow(norm(means[t]), cmap="gray"); axs[0, i].set_title(SHORT[t], fontsize=9)
        rgb = np.dstack([ref, norm(means[t]), ref * 0])
        axs[1, i].imshow(rgb); axs[1, i].set_title(f"red={SHORT[TRIALS[0]]}, green={SHORT[t]}", fontsize=7)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Original per-trial suite2p meanImg (NAS, suite2p 0.10.1); bottom: overlay vs magneto_0")
    fig.tight_layout(); fig.savefig(HERE / "fig_meanimg_nas.png", dpi=110)


if __name__ == "__main__":
    main()
