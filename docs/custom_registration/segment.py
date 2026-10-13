"""Run suite2p's cell detection on 20220915_fish1 with the movie already motion corrected, so the
ROIs are drawn on a movie where the cells stay put.

    python docs/custom_registration/segment.py --variant corrected   # suite2p1.0 env, GPU, ~2 min
    python docs/custom_registration/segment.py --variant raw         # the control

Both variants give suite2p tiffs multiplied by SCALE (as the medaka run does), with its own
registration switched off, and every other setting copied from production's run of this session
(settings_used.json). They differ in one thing only:

    corrected   each raw frame shifted back by measure.py's displacement (dy, dx): bilinear,
                pixels shifted in from outside the frame set to the offset (100)
    corrected_int  the same, rounded to whole pixels: no interpolation, so each pixel keeps its
                photon counts (bilinear mixing of neighbouring pixels changes the noise from frame
                to frame with the fractional shift, which suite2p detects as ROIs on the dark
                background; docs/resegmentation.md)
    raw         the raw frames, not moved

So corrected vs raw is the effect of the motion correction on the segmentation, and raw vs
production is the effect of the scaling (production gave suite2p the unscaled tiffs, which its
uint16 halving reduces to a few grey levels) together with switching off suite2p's registration
(which in this session applied the same shift to every frame anyway).

With registration off and the tiffs scaled, suite2p's uint16 halving and int16 binary are exact
for the raw variant (values are multiples of SCALE, well below 32767 after halving) and lose at
most 1/SCALE photon for the corrected variant (rounded to 1/SCALE photon, then halved), so its
F.npy / (SCALE / 2) is the ROI trace in raw units (production's F_full.npy counterpart).

Output (local disk only, never the NAS):
    C:/Users/dan/suite2p_v1_custom_reg/2022_09_15-fish1_cytoGCaMP/s2p_<variant>_x<SCALE>/
        tiffs/                    the tiffs suite2p read
        output/suite2p/plane0/    suite2p's output
        output/settings_used.json
"""
import argparse
import datetime
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "docs" / "suite2p_v1"))
from pipeline import ophys_extraction as ox  # noqa: E402
from run_suite2p_v1 import assert_local, assert_all_local, _py  # noqa: E402

SESSION = "2022_09_15-fish1_cytoGCaMP"
PROD = Path(f"C:/Users/dan/suite2p_v1_production/{SESSION}")
PLANE0 = PROD / "output" / "suite2p" / "plane0"
ROOT = Path("C:/Users/dan/suite2p_v1_custom_reg")
OUT_DATA = ROOT / SESSION
SCALE = 64
OFFSET = 100.0


def configure(session):
    """Point the module at another session (default: 20220915_fish1, this report's)."""
    global SESSION, PROD, PLANE0, OUT_DATA
    SESSION = session
    PROD = Path(f"C:/Users/dan/suite2p_v1_production/{SESSION}")
    PLANE0 = PROD / "output" / "suite2p" / "plane0"
    OUT_DATA = ROOT / SESSION


def write_tiffs(variant, tiff_dir):
    from scipy.ndimage import shift as nd_shift
    ops = ox.load_ops(str(PLANE0))
    sh = np.load(OUT_DATA / "shifts.npz")
    tiff_dir.mkdir(parents=True, exist_ok=True)
    for name, start, n in ox.trial_files(ops):
        dst = tiff_dir / name
        assert_local(dst, ROOT)
        if dst.exists():
            print(f"  {name}: exists, kept")
            continue
        raw = tifffile.imread(PROD / "tiffs" / name)
        assert len(raw) == n
        if variant == "raw":
            out = raw.astype(np.uint32) * SCALE
        elif variant == "corrected_int":
            # whole-pixel shift: no interpolation, so every pixel keeps its photon counts
            out = np.full(raw.shape, OFFSET * SCALE, np.float64)
            for k in range(n):
                dy, dx = int(round(sh["dy"][start + k])), int(round(sh["dx"][start + k]))
                # out(p) = raw(p + d)
                ys, yd = (slice(dy, None), slice(0, raw.shape[1] - dy)) if dy >= 0 else (slice(0, dy), slice(-dy, None))
                xs, xd = (slice(dx, None), slice(0, raw.shape[2] - dx)) if dx >= 0 else (slice(0, dx), slice(-dx, None))
                out[k][yd, xd] = raw[k][ys, xs].astype(np.float64) * SCALE
        else:
            out = np.empty(raw.shape, np.float64)
            for k in range(n):
                d = (sh["dy"][start + k], sh["dx"][start + k])
                # out(p) = raw(p + d): the content moves back onto the reference
                out[k] = nd_shift(raw[k].astype(np.float64), (-d[0], -d[1]), order=1,
                                  mode="constant", cval=OFFSET)
            out = np.round(out * SCALE)
        assert out.max() <= 65535
        tmp = tiff_dir / (name + ".part")
        tifffile.imwrite(tmp, out.astype(np.uint16))
        os.replace(tmp, dst)
        print(f"  {name}: {n} frames written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["corrected", "corrected_int", "raw"], required=True)
    ap.add_argument("--force", action="store_true", help="delete an existing suite2p output first")
    ap.add_argument("--session", default=SESSION, help="production suite2p run folder name")
    ap.add_argument("--tiffs-only", action="store_true", help="write the tiffs, don't run suite2p")
    ap.add_argument("--max-rois", type=int,
                    help="override detection.sparsery_settings.max_ROIs (production: 5000, which every "
                         "run reaches); output in s2p_<variant>_x64_max<N>/, reusing the variant's tiffs")
    a = ap.parse_args()
    configure(a.session)

    run_dir = OUT_DATA / (f"s2p_{a.variant}_x{SCALE}" + (f"_max{a.max_rois}" if a.max_rois else ""))
    tiff_dir, out_dir = OUT_DATA / f"s2p_{a.variant}_x{SCALE}" / "tiffs", run_dir / "output"
    for p in (run_dir, tiff_dir, out_dir):
        assert_local(p, ROOT)
    prod = json.load(open(PROD / "output" / "settings_used.json"))
    settings, db = prod["settings"], prod["db"]
    settings["run"]["do_registration"] = 0
    if a.max_rois:
        settings["detection"]["sparsery_settings"]["max_ROIs"] = a.max_rois
    db.update(data_path=[str(tiff_dir)], save_path0=str(out_dir), fast_disk=str(out_dir))
    assert_all_local(db, ROOT)
    if (out_dir / "suite2p").exists():
        if not a.force:
            sys.exit(f"{out_dir / 'suite2p'} exists; use --force to rerun")
        shutil.rmtree(out_dir / "suite2p")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    write_tiffs(a.variant, tiff_dir)
    t_tiff = time.time() - t0
    if a.tiffs_only:
        return

    import torch
    import suite2p
    from suite2p.run_s2p import logger_setup
    info = dict(variant=a.variant, scale=SCALE, production_run=str(PROD), date=str(datetime.datetime.now()),
                changed_from_production={"run.do_registration": 0, "db.data_path": str(tiff_dir),
                                         **({"detection.sparsery_settings.max_ROIs": a.max_rois}
                                            if a.max_rois else {})},
                db=_py(db), settings=_py(settings), status="started")
    with open(out_dir / "settings_used.json", "w") as fh:
        json.dump(info, fh, indent=1, default=str)
    logger_setup(str(out_dir / "suite2p_log"))
    t1 = time.time()
    suite2p.run_s2p(db=db, settings=settings)
    info.update(status="done", seconds_tiffs=t_tiff, seconds_suite2p=time.time() - t1,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    with open(out_dir / "settings_used.json", "w") as fh:
        json.dump(info, fh, indent=1, default=str)
    print(f"done: tiffs {t_tiff:.0f} s, suite2p {info['seconds_suite2p']:.0f} s -> {out_dir / 'suite2p/plane0'}")


if __name__ == "__main__":
    main()
