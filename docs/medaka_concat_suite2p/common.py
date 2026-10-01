"""Shared paths/constants for the medaka concatenated-suite2p investigation.

Importable from BOTH the `suite2p` conda env (02_run_suite2p.py) and the
`magneto2` env (03/04 scripts) -- numpy + stdlib only.

NAS is READ-ONLY for this investigation: nothing here (or in any sibling
script) writes under NAS_BASE. Every output goes to HERE (small: CSVs,
figures) or SCRATCH (large: tiff copies, suite2p data.bin / F.npy).
"""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
import sys as _sys
if str(REPO) not in _sys.path:          # so `magpyneto2` imports from this checkout
    _sys.path.insert(0, str(REPO))
SCRATCH = Path(os.environ.get("MEDAKA_S2P_SCRATCH", "C:/Users/dan/medaka_suite2p_scratch"))

NAS_BASE = "//datanas/family/data_aggregated/Engert/Medaka Experiments/fish3_8dpf_gcamp_good"

# Fixed concatenation order (= acquisition-condition order used by the YAMLs;
# recorded in concat_frame_ranges.csv by 02_run_suite2p.py).
TRIALS = [
    "fish3_8dpf_magneto_0", "fish3_8dpf_magneto_1", "fish3_8dpf_magneto_2",
    "fish3_8dpf_no_magneto_0", "fish3_8dpf_no_magneto_1", "fish3_8dpf_no_magneto_2",
]
SHORT = {t: t.replace("fish3_8dpf_", "") for t in TRIALS}

# Production inclusion rule + Fourier params (experiments/medaka_*.yml,
# pipeline/analysis_stages/medaka.py) -- identical across all six YAMLs.
ISCELL_THRES = 0.7
NPIX_THRES = 10
F_MAG = 0.1
Q_FRAC_MAG = 0.15
F_VIS = 1 / 60
Q_FRAC_VIS = 0.50
T_SAMPLE = 1.0

TIFF_DIR = SCRATCH / "tiffs"              # local read-only copies of the 6 raw tiffs
CONCAT_DIR = SCRATCH / "concat"           # suite2p over all 6 trials
SEPARATE_DIR = SCRATCH / "separate"       # suite2p re-run per trial, same version/ops (control)
CONCAT_ALIGNED_DIR = SCRATCH / "concat_aligned"  # suite2p over all 6 trials after rigid pre-alignment


def nas_plane0(trial):
    return f"{NAS_BASE}/{trial}/suite2p/plane0"


def nas_tiff(trial):
    return f"{NAS_BASE}/{trial}/{trial}.tif"


def load_plane0(plane0, load_F=True):
    """Return dict(stat, iscell, ops, F) from a suite2p plane0 directory."""
    plane0 = str(plane0)
    out = {
        "stat": np.load(os.path.join(plane0, "stat.npy"), allow_pickle=True),
        "iscell": np.load(os.path.join(plane0, "iscell.npy"), allow_pickle=True),
        "ops": np.load(os.path.join(plane0, "ops.npy"), allow_pickle=True).item(),
    }
    if load_F:
        out["F"] = np.load(os.path.join(plane0, "F.npy"), mmap_mode="r")
    return out


def run_plane0(kind, trial=None):
    """Local suite2p output dir. kind in {'concat', 'separate', 'nas'}."""
    if kind == "concat":
        return CONCAT_DIR / "suite2p" / "plane0"
    if kind == "separate":
        return SEPARATE_DIR / trial / "suite2p" / "plane0"
    if kind == "nas":
        return nas_plane0(trial)
    raise ValueError(kind)


def phase_corr_shift(ref, img, taper=0.1, max_shift=20):
    """Rigid (dy, dx) such that img shifted by (dy, dx) best aligns with ref,
    i.e. a point at (y, x) in `img` sits at (y + dy, x + dx) in `ref`.
    Whitened phase correlation with a Tukey-ish edge taper and 3-point
    parabolic sub-pixel refinement. Returns (dy, dx, peak_height)."""
    ref = np.asarray(ref, float); img = np.asarray(img, float)
    Ly, Lx = ref.shape
    def _tw(n):
        w = np.ones(n); k = int(taper * n)
        if k > 0:
            r = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
            w[:k] = r; w[-k:] = r[::-1]
        return w
    win = np.outer(_tw(Ly), _tw(Lx))
    a = (ref - ref.mean()) * win
    b = (img - img.mean()) * win
    R = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    R /= np.abs(R) + 1e-12
    cc = np.real(np.fft.ifft2(R))
    # restrict the peak search to |shift| <= max_shift (wrap-around indexing)
    yy = np.minimum(np.arange(Ly), Ly - np.arange(Ly))
    xx = np.minimum(np.arange(Lx), Lx - np.arange(Lx))
    ok = (yy[:, None] <= max_shift) & (xx[None, :] <= max_shift)
    iy, ix = np.unravel_index(np.argmax(np.where(ok, cc, -np.inf)), cc.shape)
    def _sub(c, i, n, axis):
        im, ip = (i - 1) % n, (i + 1) % n
        v0 = cc[i, ix] if axis == 0 else cc[iy, i]
        vm = cc[im, ix] if axis == 0 else cc[iy, im]
        vp = cc[ip, ix] if axis == 0 else cc[iy, ip]
        den = vm - 2 * v0 + vp
        return 0.0 if den == 0 else 0.5 * (vm - vp) / den
    dy = iy + _sub(cc, iy, Ly, 0); dx = ix + _sub(cc, ix, Lx, 1)
    if dy > Ly / 2: dy -= Ly
    if dx > Lx / 2: dx -= Lx
    return float(dy), float(dx), float(cc[iy, ix])


def footprint_mask(st, Ly, Lx, soma_only=False):
    m = np.zeros((Ly, Lx), bool)
    if soma_only and "soma_crop" in st:
        sc = st["soma_crop"]
        m[st["ypix"][sc], st["xpix"][sc]] = True
    else:
        m[st["ypix"], st["xpix"]] = True
    return m


def pix_index(st, Lx, dy=0, dx=0, Ly=None):
    """Linear pixel indices of an ROI footprint, optionally shifted by (dy, dx)
    and clipped to the frame."""
    y = np.asarray(st["ypix"]) + int(dy)
    x = np.asarray(st["xpix"]) + int(dx)
    if Ly is not None:
        ok = (y >= 0) & (y < Ly) & (x >= 0) & (x < Lx)
        y, x = y[ok], x[ok]
    return y * Lx + x
