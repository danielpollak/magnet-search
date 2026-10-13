"""Shared helpers for the re-segmentation report (docs/resegmentation.md): which sessions and
experiments, where each segmentation lives, production's ROI selection applied to any of them,
and the offset between production's registered coordinates and the raw frame.

Four segmentations per session ("trace sets"):
    production        the previous production suite2p run (unscaled tiffs, suite2p's registration,
                      which in these sessions applies one constant shift) and its full-resolution
                      traces F_full.npy (pipeline/ophys_extraction.py)
    reseg_raw         docs/custom_registration/segment.py --variant raw: tiffs x64, suite2p's
                      registration off
    reseg_corrected   segment.py --variant corrected: the same on frames moved back by the drift
                      that docs/custom_registration/measure.py measured (bilinear)
    reseg_corrected_int  segment.py --variant corrected_int: the drift rounded to whole pixels
                      (no interpolation); the current production run since 2026-10-12
segment.py's runs read tiffs x64 and suite2p halves uint16 input, so their F.npy / 32 is in raw
photon units, like F_full.npy.
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
from pipeline import ophys_extraction as ox  # noqa: E402

PROD_ROOT = Path("C:/Users/dan/suite2p_v1_production")
ROOT = Path("C:/Users/dan/suite2p_v1_custom_reg")
# The photon-starved zebrafish sessions (.claude/CLAUDE.md "Registration (2026-10-07)"): every
# 0.1 / 0.3 Hz zebrafish session except 20221002_fish1, whose tiffs were saved x256.
SESSIONS = ["2022_09_14-fish2_cytoGCaMP", "2022_09_15-fish1_cytoGCaMP",
            "2022_10_01-fish1_cytoGCamp", "2022_10_01-fish2_cytoGCaMP",
            "2022_10_02-fish2_cytoGCaMP"]
TAGS = ["production", "reseg_raw", "reseg_corrected", "reseg_corrected_int"]
LABELS = {"production": "production", "reseg_raw": "x64, raw frames",
          "reseg_corrected": "x64, drift corrected (bilinear)",
          "reseg_corrected_int": "x64, drift corrected (whole pixels)"}
COLORS = {"production": "#8a8a8a", "reseg_raw": "#e8590c", "reseg_corrected": "#7a3fb0",
          "reseg_corrected_int": "#2a78d6"}
OFFSET = 100.0
SCALE = 64


def short(session):
    """2022_10_01-fish2_cytoGCaMP -> 20221001_fish2"""
    d, f = session.split("-")[0], session.split("-")[1].split("_")[0]
    return d.replace("_", "") + "_" + f


def experiments():
    """{session: [experiment names]}, from the YAMLs whose suite2p_path is that session's
    production run (so the set follows the YAMLs, not a hand-made list)."""
    from pipeline import schema      # needs yaml, absent from the suite2p1.0 env
    out = {s: [] for s in SESSIONS}
    for y in sorted((_REPO / "experiments").glob("engert_*.yml")):
        cfg = schema.load_experiment(str(y))
        for s in SESSIONS:
            p = str(cfg.suite2p_path).replace("\\", "/")
            if f"/{s}/" in p or f"/{s}_x64_drift/" in p:
                out[s].append(cfg.name)
    return out


def session_of(name):
    for s, names in experiments().items():
        if name in names:
            return s
    raise KeyError(name)


def plane0(tag, session):
    if tag == "production":
        return PROD_ROOT / session / "output/suite2p/plane0"
    if tag == "reseg_corrected_int":
        # promoted to production on 2026-10-12 (docs/resegmentation/run_production.sh): the same
        # ROIs and P(iscell) as segment.py's run, F.npy to within 0.001 (GPU rounding); that run
        # was then deleted
        return PROD_ROOT / f"{session}_x64_drift" / "output/suite2p/plane0"
    return ROOT / session / f"s2p_{tag.replace('reseg_', '')}_x{SCALE}/output/suite2p/plane0"


def traces_file(tag, session):
    """(path, divisor to raw photon units)"""
    p = plane0(tag, session)
    if tag == "production":
        return p / "F_full.npy", 1.0
    if tag == "reseg_corrected_int":
        # the production run: its full-resolution traces, as the NWB files hold them (x64 units).
        # F.npy / 32 agrees to float32 rounding, which at an offset of 6400 moves NFC by ~1e-3
        return p / "F_full.npy", float(SCALE)
    return p / "F.npy", SCALE / 2


def prod_offset(session):
    """Subtract from production's ROI pixels to get raw-frame coordinates: production's
    registered frame q shows raw pixel q + (yoff + yoff1, xoff + xoff1), which must be one value
    on every frame and block (asserted)."""
    o = ox.load_ops(str(plane0("production", session)))
    for k in ("yoff", "xoff", "yoff1", "xoff1"):
        assert np.ptp(o[k]) == 0, f"{session}: production's {k} is not constant"
    s = np.array([o["yoff"].flat[0] + o["yoff1"].flat[0], o["xoff"].flat[0] + o["xoff1"].flat[0]])
    return -np.round(s).astype(int), s


def current_tag(name):
    """The segmentation the experiment's YAML (suite2p_path) uses in production now, and the
    factor from its NWB traces to raw photon units (the x64 runs' F_full is in x64 units)."""
    from pipeline import schema
    cfg = schema.load_experiment(str(_REPO / "experiments" / f"{name}.yml"))
    p = Path(str(cfg.suite2p_path))
    for tag in TAGS:
        if p == plane0(tag, session_of(name)):
            return tag, (SCALE if "_x64" in str(p) else 1.0)
    raise KeyError(f"{name}: {p} is none of the segmentations")


def offset(tag, session):
    return prod_offset(session)[0] if tag == "production" else np.zeros(2, int)


def selection(tag, session, cfg, outline_only=False):
    """Production's session-level ROI selection on one segmentation: P(iscell) and npix above
    the YAML's thresholds (strict >), inside the fish outline (the YAML's parameters, on that
    run's own mean image). Flatline removal is per trial and comes later."""
    from pipeline import body_outline
    p0 = plane0(tag, session)
    stat = np.load(p0 / "stat.npy", allow_pickle=True)
    p = np.load(p0 / "iscell.npy")[:, 1]
    npix = np.array([s["npix"] for s in stat])
    mean = ox.load_ops(str(p0))["meanImg"]
    body = body_outline.body(mean, body_outline.params_for(cfg.body_outline))
    share = body_outline.inside_share([(s["ypix"], s["xpix"]) for s in stat], body)
    if outline_only:
        return share >= body_outline.INSIDE_SHARE
    return (p > cfg.iscell_threshold) & (npix > cfg.npix_threshold) & (share >= body_outline.INSIDE_SHARE)


def trial_slice(tag, session, tiff_name):
    (f, start, n), = ox.trial_files(ox.load_ops(str(plane0(tag, session))), tiff_name)
    return start, n


def load(tag, name, outline=True, series="full"):
    """slow_variation.load's (cfg, F, ROI indices) for any of the three segmentations: production's
    selection, then flatline removal on the trial's traces, then the first N frames that
    fit_Fourier analyses. check_loader (analyze.py) asserts that on production's run this equals
    what the pipeline wrote to the NWB files."""
    assert outline and series == "full"
    from magpyneto2.engert_helpers import remove_flatlines
    from pipeline import schema
    cfg = schema.load_experiment(str(_REPO / "experiments" / f"{name}.yml"))
    session = session_of(name)
    mask = selection(tag, session, cfg)
    start, n = trial_slice(tag, session, cfg.tiff_name)
    path, div = traces_file(tag, session)
    F = np.asarray(np.load(path, mmap_mode="r")[mask, start:start + n], float) / div
    F, _, _, keep = remove_flatlines(F)
    roi = np.where(mask)[0][keep]
    N = min(int(120 * (F.shape[1] // 60)), F.shape[1])
    return cfg, F[:, :N], roi
