"""Run suite2p on the medaka fish3_8dpf trials: once over all six trials
concatenated, and (as a same-version control) once per trial separately.

ENV:  C:/Users/dan/anaconda3/envs/suite2p/python.exe   (suite2p 0.14.4, CPU torch)
NAS:  needed only for step 1 (READ-ONLY copy of the six raw tiffs, ~3 GB,
      a few minutes over the LAN) and for reading the original ops.npy.
      Nothing is written to the NAS.
TIME: measured on this machine (CPU): tiff copy ~30 s; concatenated run ~2.2 min;
      six separate runs ~35 s each; concat_aligned ~2.2 min + ~1 min tiff writing.
DISK: ~6 GB tiff copies (raw + aligned) + ~9 GB data.bin under SCRATCH (see common.py).

Usage (concat_aligned = registration-corrected variant, see run_concat_aligned):
    python 02_run_suite2p.py            # copy tiffs, concat, separate, concat_aligned
    python 02_run_suite2p.py concat     # only the concatenated run
    python 02_run_suite2p.py separate   # only the per-trial control runs
    python 02_run_suite2p.py concat_aligned  # pre-aligned concatenated run (~3 min)

ops: the original per-trial runs (suite2p 0.10.1, 2023-03-02) used the
suite2p default ops in every field that also exists in 0.14.4 (verified by
diffing ops.npy against suite2p.default_ops(): only path fields,
`bidi_corrected` [an output, not an input], `classifier_path` [0 vs '',
both meaning "unset"] and `suite2p_version` differ). So we start from
0.14.4 default_ops() and pin the scientifically relevant fields explicitly
to the original values (fs=10, tau=1, diameter=0, nonrigid, block_size=128,
sparse_mode, ...). Deliberate changes, recorded in suite2p_ops_used.csv:
  - use_builtin_classifier=True: the original run's classifier is
    unknowable (classifier_path unset + use_builtin_classifier False means
    "the user classifier on the processing machine if one existed, else
    builtin"). This machine's ~/.suite2p/classifiers/classifier_user.npy
    is from 2025 and unrelated, so we force the builtin one for every run
    here (concat AND separate control), keeping the comparison fair.
  - delete_bin=False / paths: keep the registered binary locally so
    03_compare_runs.py can compute per-trial mean images from it.
"""
import csv
import shutil
import sys
import time

import numpy as np
import suite2p

from common import (TRIALS, TIFF_DIR, CONCAT_DIR, SEPARATE_DIR, HERE,
                    nas_tiff, nas_plane0)

PINNED = ["fs", "tau", "diameter", "nchannels", "nplanes", "functional_chan",
          "do_registration", "nonrigid", "block_size", "maxregshift",
          "maxregshiftNR", "smooth_sigma", "smooth_sigma_time", "snr_thresh",
          "two_step_registration", "nimg_init", "batch_size", "th_badframes",
          "norm_frames", "do_bidiphase", "bidiphase", "1Preg", "spatial_hp_reg",
          "pre_smooth", "spatial_taper", "subpixel", "sparse_mode",
          "spatial_scale", "threshold_scaling", "max_overlap", "high_pass",
          "connected", "max_iterations", "nbinned", "spatial_hp_detect",
          "denoise", "anatomical_only", "neucoeff", "inner_neuropil_radius",
          "min_neuropil_pixels", "allow_overlap", "lam_percentile",
          "preclassify", "soma_crop", "baseline", "win_baseline",
          "sig_baseline", "prctile_baseline", "spikedetect"]


def base_ops():
    orig = np.load(f"{nas_plane0(TRIALS[0])}/ops.npy", allow_pickle=True).item()
    ops = suite2p.default_ops()
    rows = []
    for k in PINNED:
        if k in orig:
            ops[k] = orig[k]
            rows.append((k, repr(orig[k]), "copied from original ops.npy"))
    ops["use_builtin_classifier"] = True
    rows.append(("use_builtin_classifier", "True",
                 "CHANGED: original classifier unknowable; force builtin for all local runs"))
    ops["delete_bin"] = False
    ops["keep_movie_raw"] = False
    rows.append(("delete_bin", "False", "keep registered data.bin locally (scratch)"))
    rows.append(("suite2p_version", suite2p.version, f"original: {orig['suite2p_version']}"))
    with open(HERE / "suite2p_ops_used.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "value", "note"])
        w.writerows(rows)
    return ops


def copy_tiffs():
    TIFF_DIR.mkdir(parents=True, exist_ok=True)
    for t in TRIALS:
        dst = TIFF_DIR / f"{t}.tif"
        if dst.exists():
            continue
        t0 = time.time()
        shutil.copyfile(nas_tiff(t), dst)          # read-only on the NAS side
        print(f"copied {t}.tif ({time.time() - t0:.0f}s)")


def run_concat(ops):
    CONCAT_DIR.mkdir(parents=True, exist_ok=True)
    db = {"data_path": [str(TIFF_DIR)],
          "tiff_list": [f"{t}.tif" for t in TRIALS],   # fixed order
          "save_path0": str(CONCAT_DIR), "fast_disk": str(CONCAT_DIR)}
    t0 = time.time()
    out = suite2p.run_s2p(ops=dict(ops), db=db)
    print(f"concat run done in {(time.time() - t0) / 60:.1f} min")
    fpf = np.load(CONCAT_DIR / "suite2p" / "plane0" / "ops.npy", allow_pickle=True).item()["frames_per_file"]
    starts = np.concatenate([[0], np.cumsum(fpf)[:-1]])
    with open(HERE / "concat_frame_ranges.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["order", "trial", "tiff", "frame_start", "frame_stop", "n_frames"])
        for i, (t, s, n) in enumerate(zip(TRIALS, starts, fpf)):
            w.writerow([i, t, f"{t}.tif", int(s), int(s + n), int(n)])
    return out


def run_separate(ops):
    for t in TRIALS:
        d = SEPARATE_DIR / t
        if (d / "suite2p" / "plane0" / "iscell.npy").exists():
            print(f"separate {t}: exists, skipping")
            continue
        d.mkdir(parents=True, exist_ok=True)
        db = {"data_path": [str(TIFF_DIR)], "tiff_list": [f"{t}.tif"],
              "save_path0": str(d), "fast_disk": str(d)}
        t0 = time.time()
        suite2p.run_s2p(ops=dict(ops), db=db)
        print(f"separate {t} done in {(time.time() - t0) / 60:.1f} min")


def run_concat_aligned(ops):
    """Variant: suite2p's own registration turned out to be a no-op on this
    photon-starved data (refImg is ~flat, every frame's rigid offset is 0 --
    see the write-up), so the ~2 px inter-block shift between the magneto_*
    and no_magneto_* trials is NOT corrected by the plain concatenated run.
    Here we estimate each trial's rigid shift vs magneto_0 from its raw-tiff
    mean image (phase correlation), apply the ROUNDED integer shift to every
    frame (edges filled with the trial's median value; integer to keep the
    photon-count statistics untouched), write the shifted copies to SCRATCH,
    and run the same suite2p ops over them. Shifts -> concat_aligned_shifts.csv."""
    import tifffile
    from common import phase_corr_shift
    adir = TIFF_DIR.parent / "tiffs_aligned"
    adir.mkdir(parents=True, exist_ok=True)
    movs = {t: tifffile.imread(TIFF_DIR / f"{t}.tif") for t in TRIALS}
    means = {t: movs[t].mean(0) for t in TRIALS}
    nas_mean0 = np.load(f"{nas_plane0(TRIALS[0])}/ops.npy", allow_pickle=True).item()["meanImg"]
    rows = []
    for t in TRIALS:
        dy, dx, pk = phase_corr_shift(means[TRIALS[0]], means[t])
        iy, ix = int(round(dy)), int(round(dx))
        m = movs[t]
        out = np.full_like(m, int(np.median(m[::50])))
        ys, yd = (slice(0, m.shape[1] - iy), slice(iy, None)) if iy >= 0 else (slice(-iy, None), slice(0, m.shape[1] + iy))
        xs, xd = (slice(0, m.shape[2] - ix), slice(ix, None)) if ix >= 0 else (slice(-ix, None), slice(0, m.shape[2] + ix))
        out[:, yd, xd] = m[:, ys, xs]
        tifffile.imwrite(adir / f"{t}.tif", out)
        rdy, rdx, _ = phase_corr_shift(nas_mean0, means[t]) if t == TRIALS[0] else (np.nan, np.nan, 0)
        rows.append([t, round(dy, 2), round(dx, 2), round(pk, 4), iy, ix,
                     "" if np.isnan(rdy) else f"{rdy:.2f},{rdx:.2f}"])
        print(t, rows[-1])
    with open(HERE / "concat_aligned_shifts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trial", "dy_vs_magneto_0", "dx_vs_magneto_0", "pc_peak", "applied_dy", "applied_dx",
                    "raw_mean_vs_NAS_meanImg_shift(magneto_0 only)"])
        w.writerows(rows)
    del movs
    d = TIFF_DIR.parent / "concat_aligned"
    d.mkdir(parents=True, exist_ok=True)
    db = {"data_path": [str(adir)], "tiff_list": [f"{t}.tif" for t in TRIALS],
          "save_path0": str(d), "fast_disk": str(d)}
    t0 = time.time()
    suite2p.run_s2p(ops=dict(ops), db=db)
    print(f"concat_aligned run done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    ops = base_ops()
    copy_tiffs()
    if what in ("all", "concat"):
        run_concat(ops)
    if what in ("all", "separate"):
        run_separate(ops)
    if what in ("all", "concat_aligned"):
        run_concat_aligned(ops)
