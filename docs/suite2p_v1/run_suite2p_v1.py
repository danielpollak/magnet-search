"""Re-run one GCaMP imaging session with suite2p v1 (PyTorch/GPU), matched to the original run.

Run with the `suite2p1.0` conda env (suite2p 1.1.0 + CUDA torch):

    C:/Users/dan/anaconda3/envs/suite2p1.0/python.exe docs/suite2p_v1/run_suite2p_v1.py \
        --session 2022_10_01-fish2_cytoGCaMP

    # list the 15 known sessions, or only build/print the settings without running
    ... run_suite2p_v1.py --list
    ... run_suite2p_v1.py --session 2022_02_21 --dry-run

What it does, for one session (a directory under the Engert NAS root that holds the tiffs and
the ORIGINAL suite2p/plane0/ output):

1. Reads the original suite2p/plane0/ops.npy from the NAS (read-only).
2. Copies the tiffs listed in that ops' `filelist` (basenames; the stored full paths are stale,
   the files sit in the session directory itself), in the same order, to
   <local-root>/<run-name>/tiffs/. A file already present with the same size is not re-copied.
3. Maps the original ops onto suite2p v1's nested `settings` dict + `db` dict (see
   build_settings(); every mapping, and every old key without a v1 equivalent, is recorded in
   <run-name>/output/settings_used.json under "mapping").
4. Refuses to run unless data_path, save_path0, fast_disk (and every other path it writes to)
   are local: not starting with // or \\\\, not containing "datanas", not on a network drive,
   and inside <local-root>.
   With --scale K, suite2p reads x K copies of the tiffs (<run-name>/tiffs_xK/) instead: in
   the photon-starved medaka movies (pixel values 99-306) suite2p's reference image has only
   ~6 grey levels after its uint16 halving, norm_frames clips everything to about one value,
   and registration returns the same shift for every frame. 2022_10_02-fish1's tiffs were
   saved x256 and register normally; x128 does the same for medaka (2026-10-07).
   With --add-session (repeatable), the tiffs of further sessions are appended, in order, so
   suite2p segments them as one movie and every trial shares one set of ROIs. Settings still
   come from --session's ops. Used for the six medaka trials (2026-10-07), whose separate
   segmentations gave no ROI identity across trials (docs/medaka_concat_suite2p.md).
5. Runs suite2p.run_s2p and records runtime, GPU, versions and per-file frame counts in
   settings_used.json. The original NAS plane0 files' sizes and mtimes are recorded before and
   after the run and must be unchanged.

Nothing is ever written to the NAS.
"""
import argparse
import ctypes
import datetime
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

NAS_ROOT = "//datanas/family/data_aggregated/Engert"
LOCAL_ROOT = "C:/Users/dan/suite2p_v1_runs"
MEDAKA = "Medaka Experiments/fish3_8dpf_gcamp_good"

# session (relative to NAS_ROOT) -> suite2p version of the original run, for reference only
SESSIONS = {
    "2022_02_21": "0.14.4",
    "2022_02_23": "0.14.4",
    "2022_03_01": "0.14.4",
    "2022_09_14-fish2_cytoGCaMP": "0.10.1",
    "2022_09_15-fish1_cytoGCaMP": "0.10.1",
    "2022_10_01-fish1_cytoGCamp": "0.10.1",
    "2022_10_01-fish2_cytoGCaMP": "0.10.1",
    "2022_10_02-fish1_cytoGCaMP": "0.10.1",
    "2022_10_02-fish2_cytoGCaMP": "0.10.1",
    **{f"{MEDAKA}/fish3_8dpf_{t}": "0.10.1"
       for t in ["magneto_0", "magneto_1", "magneto_2",
                 "no_magneto_0", "no_magneto_1", "no_magneto_2"]},
}

SAFETY_MARGIN_GB = 20.0


# ----------------------------------------------------------------------------- safety
def _is_network_drive(path):
    """True if `path` sits on a Windows network (mapped) drive."""
    if os.name != "nt":
        return False
    drive = os.path.splitdrive(os.path.abspath(path))[0]
    if not drive or drive.startswith(("\\\\", "//")):
        return True
    DRIVE_REMOTE = 4
    return ctypes.windll.kernel32.GetDriveTypeW(drive + "\\") == DRIVE_REMOTE


def assert_local(path, local_root):
    """Raise unless `path` is a local path inside `local_root`."""
    p = str(path)
    a = os.path.abspath(p)
    for s in (p, a, os.path.realpath(p)):
        if s.startswith("//") or s.startswith("\\\\"):
            raise RuntimeError(f"UNSAFE PATH (UNC): {p}")
        if "datanas" in s.lower():
            raise RuntimeError(f"UNSAFE PATH (contains 'datanas'): {p}")
    if _is_network_drive(a):
        raise RuntimeError(f"UNSAFE PATH (network drive): {p}")
    root = os.path.normcase(os.path.abspath(local_root))
    if os.path.commonpath([root, os.path.normcase(a)]) != root:
        raise RuntimeError(f"UNSAFE PATH (outside local root {local_root}): {p}")


def assert_all_local(db, local_root):
    keys = ["data_path", "save_path0", "fast_disk"]
    for k in keys:
        v = db.get(k)
        if v is None:
            raise RuntimeError(f"db['{k}'] must be set explicitly (a None value defaults elsewhere)")
        for p in (v if isinstance(v, (list, tuple)) else [v]):
            assert_local(p, local_root)
    for f in db.get("file_list") or []:
        assert_local(os.path.join(db["data_path"][0], f), local_root)


def nas_fingerprint(plane0):
    """{filename: (size, mtime_ns)} of the original output files, to prove they are untouched."""
    return {f.name: (f.stat().st_size, f.stat().st_mtime_ns)
            for f in sorted(Path(plane0).iterdir()) if f.is_file()}


# ----------------------------------------------------------------------------- settings
def _py(v):
    """numpy scalars/arrays/tuples -> JSON-able python."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, dict):
        return {k: _py(x) for k, x in v.items()}
    if isinstance(v, (datetime.datetime, datetime.date, Path)):
        return str(v)
    return v


def build_settings(old, default_settings, default_db):
    """Map original ops -> (settings, db_partial, mapping). `mapping` is a list of
    {old_key, old_value, v1_key, v1_value, note} records, one per old setting considered."""
    s = default_settings()
    db = default_db()
    mapping = []

    def put(old_key, v1_key, conv=lambda x: x, note=""):
        if old_key not in old:
            mapping.append(dict(old_key=old_key, old_value="<missing in old ops>",
                                v1_key=v1_key, v1_value=_py(_get(v1_key)),
                                note=("kept v1 default. " + note).strip()))
            return
        val = conv(old[old_key])
        _set(v1_key, val)
        mapping.append(dict(old_key=old_key, old_value=_py(old[old_key]), v1_key=v1_key,
                            v1_value=_py(val), note=note))

    def _target(v1_key):
        root, *parts = v1_key.split(".")
        d = db if root == "db" else s
        if root != "db":
            parts = [root] + parts
        return d, parts

    def _get(v1_key):
        d, parts = _target(v1_key)
        for p in parts[:-1]:
            d = d[p]
        return d[parts[-1]]

    def _set(v1_key, val):
        d, parts = _target(v1_key)
        for p in parts[:-1]:
            d = d[p]
        if parts[-1] not in d:
            raise KeyError(f"{v1_key} is not a suite2p v1 setting")
        d[parts[-1]] = val

    def none(old_key, note):
        mapping.append(dict(old_key=old_key, old_value=_py(old.get(old_key, "<missing in old ops>")),
                            v1_key=None, v1_value=None, note=note))

    b = lambda x: bool(x)
    i = lambda x: int(x)
    f = lambda x: float(x)

    # --- general
    put("fs", "fs", f, "kept as in the original run (10 Hz, the suite2p default) even though "
        "these movies are ~1 frame/s: fs sets the detection bin size (round(tau*fs) frames), the "
        "baseline window (win_baseline*fs frames) and deconvolution, so changing it would change "
        "the result for reasons unrelated to the suite2p version")
    put("tau", "tau", f)
    s["torch_device"] = "cuda"
    mapping.append(dict(old_key=None, old_value=None, v1_key="torch_device", v1_value="cuda",
                        note="v1 only (PyTorch); overridable with --torch-device"))
    put("diameter", "diameter",
        lambda x: [float(x), float(x)] if np.ndim(x) == 0 and float(x) > 0 else
        ([float(v) for v in x] if np.ndim(x) == 1 and min(x) > 0 else s["diameter"]),
        "only used by sourcery/cellpose; old 0 (= unset) keeps v1's default [12, 12], unused by sparsery")

    # --- db (input/IO)
    put("nplanes", "db.nplanes", i)
    put("nchannels", "db.nchannels", i)
    put("functional_chan", "db.functional_chan", i)
    put("keep_movie_raw", "db.keep_movie_raw", b)
    put("force_sktiff", "db.force_sktiff", b)
    put("batch_size", "db.batch_size", i, "old batch_size was shared by tiff->binary and registration")
    put("look_one_level_down", "db.look_one_level_down", lambda x: False,
        "always False here: an explicit file_list is used")
    put("ignore_flyback", "db.ignore_flyback", lambda x: (list(x) or None))
    db["input_format"] = "tif"
    none("filelist", "replaced by db.file_list = basenames of the old filelist, in order, "
         "relative to the LOCAL tiff copy directory")
    none("data_path", "replaced by the local tiff copy directory")
    none("save_path0", "replaced by the local output directory")
    none("fast_disk", "replaced by the local output directory")

    # --- run switches
    put("do_registration", "run.do_registration", i)
    put("roidetect", "run.do_detection", b)
    put("spikedetect", "run.do_deconvolution", b)
    put("multiplane_parallel", "run.multiplane_parallel", b)

    # --- io
    put("combined", "io.combined", b)
    put("save_mat", "io.save_mat", b)
    put("save_NWB", "io.save_NWB", b)
    put("delete_bin", "io.delete_bin", lambda x: False,
        "forced False: data.bin is kept to check the uint16 halving empirically")
    put("move_bin", "io.move_bin", lambda x: False, "forced False (fast_disk == save path anyway)")

    # --- registration
    put("align_by_chan", "registration.align_by_chan2", lambda x: int(x) == 2)
    put("nimg_init", "registration.nimg_init", i)
    put("maxregshift", "registration.maxregshift", f)
    put("do_bidiphase", "registration.do_bidiphase", b)
    put("bidiphase", "registration.bidiphase", f)
    put("batch_size", "registration.batch_size", i)
    put("nonrigid", "registration.nonrigid", b)
    put("maxregshiftNR", "registration.maxregshiftNR", i)
    put("block_size", "registration.block_size", lambda x: tuple(int(v) for v in x))
    put("smooth_sigma_time", "registration.smooth_sigma_time", f)
    put("smooth_sigma", "registration.smooth_sigma", f)
    put("spatial_taper", "registration.spatial_taper", f,
        "same meaning (edge taper in pixels); v1 default is 3.45, original runs used 40")
    put("th_badframes", "registration.th_badframes", f)
    put("norm_frames", "registration.norm_frames", b)
    put("snr_thresh", "registration.snr_thresh", f)
    put("subpixel", "registration.subpixel", i)
    put("two_step_registration", "registration.two_step_registration", b)
    put("reg_tif", "registration.reg_tif", b)
    put("reg_tif_chan2", "registration.reg_tif_chan2", b)
    for k in ["1Preg", "spatial_hp_reg", "pre_smooth", "spatial_hp"]:
        none(k, "no v1 equivalent (1P-registration high-pass/smoothing removed in v1); "
             "1Preg was False in the original runs, so these had no effect there either")
    none("bidi_corrected", "output flag, not a setting (bidiphase was 0)")
    none("frames_include", "no v1 equivalent; all frames are used (original value -1 = all)")
    none("aspect", "no v1 setting (GUI display only; 1.0 in the original runs)")

    # --- detection
    if old.get("anatomical_only", 0):
        alg = "cellpose"
    else:
        alg = "sparsery" if old.get("sparse_mode", True) else "sourcery"
    s["detection"]["algorithm"] = alg
    mapping.append(dict(old_key="sparse_mode / anatomical_only",
                        old_value=[_py(old.get("sparse_mode")), _py(old.get("anatomical_only"))],
                        v1_key="detection.algorithm", v1_value=alg, note=""))
    put("denoise", "detection.denoise", b)
    put("nbinned", "detection.nbins", i)
    mapping.append(dict(old_key="(bin_size formula)", old_value="max(1, nframes//nbinned, round(tau*fs))",
                        v1_key="detection.bin_size", v1_value=None,
                        note="None in v1 = the same formula as the original runs"))
    put("high_pass", "detection.highpass_time", i)
    put("threshold_scaling", "detection.threshold_scaling", f)
    put("max_overlap", "detection.max_overlap", f)
    put("soma_crop", "detection.soma_crop", b)
    put("chan2_thres", "detection.chan2_threshold", f)
    put("spatial_hp_detect", "detection.sparsery_settings.highpass_neuropil", i)
    put("spatial_scale", "detection.sparsery_settings.spatial_scale", i)
    put("max_iterations", "detection.sparsery_settings.max_ROIs", lambda x: 250 * int(x),
        "old sparsery ran for 250*max_iterations iterations (detect.py), i.e. at most that many ROIs")
    put("max_iterations", "detection.sourcery_settings.max_iterations", i, "sourcery only (unused)")
    put("connected", "detection.sourcery_settings.connected", b, "sourcery only (unused)")
    put("flow_threshold", "detection.cellpose_settings.flow_threshold", f, "cellpose only (unused)")
    put("cellprob_threshold", "detection.cellpose_settings.cellprob_threshold", f,
        "cellpose only (unused)")
    for k in ["pretrained_model", "spatial_hp_cp"]:
        none(k, "cellpose-only; v1 uses cellpose 4 ('cpsam'); unused because detection is sparsery")
    for k, note in [("detection.npix_norm_min", "v1 only; 0 = no lower npix_norm cut (none in the original runs)"),
                    ("detection.npix_norm_max", "v1 only; 100 = effectively no upper npix_norm cut"),
                    ("detection.sparsery_settings.active_percentile", "v1 only; 0 = original thresholding"),
                    ("detection.sourcery_settings.smooth_masks", "v1 only, sourcery (unused)"),
                    ("extraction.snr_threshold", "v1 only; 0 = no SNR-based ROI removal"),
                    ("extraction.circular_neuropil", "v1 only; False = square neuropil masks as before"),
                    ("registration.upsample_meanImg", "v1 only; None = no upsampling"),
                    ("run.do_regmetrics", "v1 default kept (registration metrics, as 0.14.4 did)")]:
        mapping.append(dict(old_key=None, old_value=None, v1_key=k, v1_value=_py(_get(k)), note=note))

    # --- classification
    s["classification"]["classifier_path"] = None
    s["classification"]["use_builtin_classifier"] = True
    mapping.append(dict(
        old_key="classifier_path / use_builtin_classifier",
        old_value=[_py(old.get("classifier_path")), _py(old.get("use_builtin_classifier"))],
        v1_key="classification.use_builtin_classifier", v1_value=True,
        note="original runs used the user classifier of the processing machine, which is not "
             "recorded; on this machine ~/.suite2p/classifiers/classifier_user.npy is byte-identical "
             "to the built-in classifier.npy (of v1.1.0 and 0.14.4), so the built-in one is used "
             "explicitly for reproducibility"))
    put("preclassify", "classification.preclassify", f)

    # --- extraction
    put("neuropil_extract", "extraction.neuropil_extract", b)
    put("neucoeff", "extraction.neuropil_coefficient", f)
    put("inner_neuropil_radius", "extraction.inner_neuropil_radius", i)
    put("min_neuropil_pixels", "extraction.min_neuropil_pixels", i)
    put("lam_percentile", "extraction.lam_percentile", f)
    put("allow_overlap", "extraction.allow_overlap", b)

    # --- deconvolution preprocessing
    put("baseline", "dcnv_preprocess.baseline", str)
    put("win_baseline", "dcnv_preprocess.win_baseline", f)
    put("sig_baseline", "dcnv_preprocess.sig_baseline", f)
    put("prctile_baseline", "dcnv_preprocess.prctile_baseline", f)

    return s, db, mapping


# ----------------------------------------------------------------------------- copy
def copy_tiffs(src_dir, names, dst_dir, local_root):
    dst_dir.mkdir(parents=True, exist_ok=True)
    assert_local(dst_dir, local_root)
    for n in names:
        src, dst = Path(src_dir) / n, dst_dir / n
        assert_local(dst, local_root)
        size = src.stat().st_size
        if dst.exists() and dst.stat().st_size == size:
            print(f"  have  {n} ({size / 1e9:.2f} GB)")
            continue
        tmp = dst.with_name(dst.name + ".part")
        t = time.time()
        shutil.copyfile(src, tmp)          # read from NAS, write locally
        os.replace(tmp, dst)
        print(f"  copied {n} ({size / 1e9:.2f} GB, {time.time() - t:.0f} s)")


def scale_tiffs(src_dir, names, dst_dir, k, local_root):
    """Write uint16 copies of the tiffs multiplied by k; refuse if any pixel would overflow."""
    import tifffile
    dst_dir.mkdir(parents=True, exist_ok=True)
    assert_local(dst_dir, local_root)
    for n in names:
        dst = dst_dir / n
        assert_local(dst, local_root)
        with tifffile.TiffFile(Path(src_dir) / n) as tf:
            a = np.stack([p.asarray() for p in tf.pages])
        if a.dtype != np.uint16 or int(a.max()) * k > 65535:
            raise ValueError(f"{n}: {a.dtype}, max {a.max()}: x{k} would not fit in uint16")
        tifffile.imwrite(dst, (a.astype(np.uint32) * k).astype(np.uint16))
        print(f"  scaled {n} x{k} (max {a.max()} -> {int(a.max()) * k})")


def tiff_nframes(path):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0].shape
    return int(s[0]) if len(s) == 3 else 1


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="session directory relative to --nas-root (see --list)")
    ap.add_argument("--list", action="store_true", help="list known sessions and exit")
    ap.add_argument("--nas-root", default=NAS_ROOT)
    ap.add_argument("--local-root", default=LOCAL_ROOT)
    ap.add_argument("--run-name", help="local folder name (default: last component of --session)")
    ap.add_argument("--torch-device", default="cuda")
    ap.add_argument("--reg-batch-size", type=int,
                    help="override registration.batch_size (only if the GPU runs out of memory)")
    ap.add_argument("--force", action="store_true",
                    help="delete an existing local output/ folder and rerun from scratch")
    ap.add_argument("--dry-run", action="store_true", help="build and print settings; no copy, no run")
    ap.add_argument("--add-session", action="append", default=[],
                    help="append this session's tiffs (relative to --nas-root; repeatable), "
                         "segmenting all sessions together; see step 4 above")
    ap.add_argument("--scale", type=int, default=1,
                    help="run suite2p on uint16 copies of the tiffs multiplied by this factor "
                         "(written to <run-name>/tiffs_x<scale>/); see step 4 above")
    ap.add_argument("--delete-tiffs", action="store_true",
                    help="delete the local tiff copies after a successful run")
    a = ap.parse_args()

    if a.list:
        for k, v in SESSIONS.items():
            print(f"{v:8s} {k}")
        return
    if not a.session:
        ap.error("--session is required")

    # never accept a NAS local root
    assert_local(a.local_root, a.local_root)

    session_dir = Path(a.nas_root) / a.session
    plane0 = session_dir / "suite2p" / "plane0"
    run_name = a.run_name or Path(a.session).name
    run_dir = Path(a.local_root) / run_name
    tiff_dir, out_dir = run_dir / "tiffs", run_dir / "output"
    s2p_tiff_dir = run_dir / f"tiffs_x{a.scale}" if a.scale != 1 else tiff_dir
    for p in (run_dir, tiff_dir, s2p_tiff_dir, out_dir):
        assert_local(p, a.local_root)

    old = np.load(plane0 / "ops.npy", allow_pickle=True).item()
    names = [os.path.basename(str(p).replace("\\", "/")) for p in old["filelist"]]
    fpf_old = [int(x) for x in old["frames_per_file"]]
    source = {n: session_dir for n in names}          # tiff name -> NAS directory holding it
    plane0s = [plane0]
    for extra in a.add_session:
        xdir = Path(a.nas_root) / extra
        xold = np.load(xdir / "suite2p" / "plane0" / "ops.npy", allow_pickle=True).item()
        xnames = [os.path.basename(str(p).replace("\\", "/")) for p in xold["filelist"]]
        if (xold["Ly"], xold["Lx"]) != (old["Ly"], old["Lx"]) or set(xnames) & set(source):
            raise ValueError(f"{extra}: different frame size or a repeated tiff name")
        names += xnames
        fpf_old += [int(x) for x in xold["frames_per_file"]]
        source.update({n: xdir for n in xnames})
        plane0s.append(xdir / "suite2p" / "plane0")
    print(f"session {a.session}: original suite2p {old.get('suite2p_version')}, "
          f"{len(names)} tiffs, frames_per_file {fpf_old}, Ly x Lx {old['Ly']} x {old['Lx']}")

    import suite2p
    from suite2p import default_settings, default_db
    settings, db, mapping = build_settings(old, default_settings, default_db)
    settings["torch_device"] = a.torch_device
    if a.reg_batch_size:
        settings["registration"]["batch_size"] = a.reg_batch_size
        mapping.append(dict(old_key="batch_size", old_value=_py(old.get("batch_size")),
                            v1_key="registration.batch_size", v1_value=a.reg_batch_size,
                            note="overridden on the command line (GPU memory)"))
    if a.scale != 1:
        mapping.append(dict(old_key="(tiffs)", old_value="as acquired", v1_key="db.data_path",
                            v1_value=f"tiffs_x{a.scale}",
                            note=f"suite2p reads the tiffs multiplied by {a.scale} (--scale)"))
    db.update(data_path=[str(s2p_tiff_dir)], save_path0=str(out_dir), fast_disk=str(out_dir),
              file_list=names, save_folder="suite2p")
    assert_all_local(db, a.local_root)

    if a.dry_run:
        print(json.dumps({"db": _py(db), "settings": _py(settings)}, indent=1))
        for m in mapping:
            print(m)
        return

    # disk space: tiffs still to copy + a data.bin of the same size + margin
    sizes = {n: (source[n] / n).stat().st_size for n in names}
    run_dir.mkdir(parents=True, exist_ok=True)
    have = sum((tiff_dir / n).stat().st_size for n in names
               if (tiff_dir / n).exists() and (tiff_dir / n).stat().st_size == sizes[n])
    need = ((sum(sizes.values()) - have) + sum(sizes.values()) * (2 if a.scale != 1 else 1)
            + SAFETY_MARGIN_GB * 1e9)
    free = shutil.disk_usage(run_dir).free
    print(f"disk: {free / 1e9:.0f} GB free, need ~{need / 1e9:.1f} GB "
          f"(tiffs {sum(sizes.values()) / 1e9:.1f} GB + binary + {SAFETY_MARGIN_GB:.0f} GB margin)")
    if free < need:
        sys.exit("not enough free disk space")

    if (out_dir / "suite2p").exists():
        if not a.force:
            sys.exit(f"{out_dir / 'suite2p'} exists; use --force to rerun from scratch")
        assert_local(out_dir / "suite2p", a.local_root)
        shutil.rmtree(out_dir / "suite2p")
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_before = [nas_fingerprint(p) for p in plane0s]
    t0 = time.time()
    for src in dict.fromkeys(source.values()):
        copy_tiffs(src, [n for n in names if source[n] == src], tiff_dir, a.local_root)
    if a.scale != 1:
        scale_tiffs(tiff_dir, names, s2p_tiff_dir, a.scale, a.local_root)
    t_copy = time.time() - t0
    fpf_new = [tiff_nframes(s2p_tiff_dir / n) for n in names]
    if fpf_new != fpf_old:
        print(f"WARNING: tiff frame counts {fpf_new} differ from the old ops' frames_per_file {fpf_old}")

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()} ({gpu}); "
          f"device '{settings['torch_device']}'")

    # record the settings before running, so a crashed run still documents what it tried
    with open(out_dir / "settings_used.json", "w") as fh:
        json.dump(dict(session=a.session, status="started", db=_py(db), settings=_py(settings),
                       mapping=mapping), fh, indent=1, default=str)

    # suite2p's own log (registration/detection details) -> <output>/suite2p_log/run.log
    from suite2p.run_s2p import logger_setup
    log_dir = out_dir / "suite2p_log"
    assert_local(log_dir, a.local_root)
    logger_setup(str(log_dir))

    t1 = time.time()
    suite2p.run_s2p(db=db, settings=settings)
    t_run = time.time() - t1

    fp_after = [nas_fingerprint(p) for p in plane0s]
    if fp_after != fp_before:
        raise RuntimeError("ORIGINAL NAS suite2p/plane0 FILES CHANGED DURING THE RUN")

    p0 = out_dir / "suite2p" / "plane0"
    new_ops = np.load(p0 / "ops.npy", allow_pickle=True).item()
    info = dict(
        session=a.session, added_sessions=a.add_session, run_name=run_name,
        nas_session_dir=str(session_dir),
        original_suite2p_version=_py(old.get("suite2p_version")),
        suite2p_version=_py(getattr(suite2p, "version", None) or
                            __import__("importlib.metadata").metadata.version("suite2p")),
        torch_version=torch.__version__, torch_device=settings["torch_device"], gpu=gpu,
        python=sys.version, date=str(datetime.datetime.now()),
        tiffs=names, tiff_scale=a.scale, frames_per_file_old=fpf_old, frames_per_file_tiff=fpf_new,
        nframes_v1=_py(new_ops.get("nframes")),
        tiff_bytes=sum(sizes.values()),
        output_bytes=sum(f.stat().st_size for f in p0.iterdir() if f.is_file()),
        seconds_copy=t_copy, seconds_suite2p=t_run, plane_times=_py(new_ops.get("plane_times")),
        nas_plane0_unchanged=True,
        db=_py(db), settings=_py(settings), mapping=mapping,
    )
    with open(out_dir / "settings_used.json", "w") as fh:
        json.dump(info, fh, indent=1, default=str)
    print(f"done: copy {t_copy:.0f} s, suite2p {t_run:.0f} s -> {p0}")

    if a.delete_tiffs:
        for n in names:
            (tiff_dir / n).unlink()


if __name__ == "__main__":
    main()
