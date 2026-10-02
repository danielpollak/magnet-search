"""One-time extraction of suite2p summary images for every field of view, cached locally.

    python docs/nfc_finite_sample_bias/fov_images.py     # ~2 min, reads ops.npy from the NAS

The NWB files keep only the ROI masks and Ly/Lx, not suite2p's summary images. This copies
meanImg, meanImgE, max_proj and Vcorr out of each segmentation's ops.npy (100+ MB each on
the NAS) into fov_images.npz (gitignored, ~20 MB). max_proj and Vcorr only cover suite2p's
motion-corrected crop (yrange/xrange); they are padded back to the full frame with NaN.
Keys are "<segmentation index>/<image name>"; "segmentations" lists the session paths in
index order.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import slow_variation as sv  # noqa: E402
from slow_variation import schema  # noqa: E402
from coverage_vs_roi_quality import recordings, segmentation_key  # noqa: E402

CACHE = _HERE / "fov_images.npz"
IMAGES = ["meanImg", "meanImgE", "max_proj", "Vcorr"]


def session_path(name):
    return schema.load_experiment(str(sv._REPO / "experiments" / f"{name}.yml")).session_path


def extract():
    seen, out = {}, {}
    for name, _ in recordings():
        key = segmentation_key(name)
        if key in seen:
            continue
        seen[key] = len(seen)
        ops_path = os.path.join(session_path(name), "suite2p", "plane0", "ops.npy")
        print(f"  {ops_path}", flush=True)
        ops = np.load(ops_path, allow_pickle=True).item()
        Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
        (y0, y1), (x0, x1) = ops["yrange"], ops["xrange"]
        for im in IMAGES:
            a = np.asarray(ops[im], dtype=np.float32)
            if a.shape != (Ly, Lx):              # cropped to the motion-corrected range
                full = np.full((Ly, Lx), np.nan, np.float32)
                full[y0:y1, x0:x1] = a
                a = full
            out[f"{seen[key]}/{im}"] = a
    np.savez_compressed(CACHE, segmentations=np.array(list(seen)), **out)
    print("wrote", CACHE)


def load():
    """{segmentation key: {image name: array}}."""
    z = np.load(CACHE)
    return {seg: {im: z[f"{i}/{im}"] for im in IMAGES}
            for i, seg in enumerate(z["segmentations"])}


if __name__ == "__main__":
    extract()
