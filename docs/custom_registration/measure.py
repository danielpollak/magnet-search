"""Measure the fish's motion in 20220915_fish1 from the raw tiffs, frame by frame, where suite2p's
registration returned one constant shift for every frame.

    python docs/custom_registration/measure.py          # magneto2 env, ~5 min, CPU
    python docs/custom_registration/measure.py --session 2022_10_01-fish2_cytoGCaMP --csv <path>

Method: average the raw frames in bins of BIN frames (the movies are photon-starved, ~96% of
pixels at the offset 100, so single frames are mostly empty), and cross-correlate each bin with a
reference image of the same session (rigid translation, subpixel by a parabolic fit to the
peak). Two passes: the first reference is the plain mean of every frame; the second is the mean
of the bins after shifting each by its first-pass displacement. Bins never straddle a 60-frame
chunk of the stitched trial tiffs (each trial tiff is 60 s chunks from different times of the
session), and the per-frame displacement is interpolated between bin centres within a chunk only.

Precision: each bin's odd and even frames are measured separately; half their difference is the
measurement error of the whole bin.

Check that the correction works: each bin, shifted back by its displacement, is measured again
against the reference (residual), and its correlation with the reference is compared before and
after.

Outputs:
    results_bins.csv                    one row per bin: displacement, precision, residual, corr
    C:/Users/dan/suite2p_v1_custom_reg/2022_09_15-fish1_cytoGCaMP/shifts.npz
                                        per-frame displacement (dy, dx) in raw pixels, plus the
                                        reference and early/settled mean images for the figures
"""
import os
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter, median_filter, shift as nd_shift

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))

SESSION = "2022_09_15-fish1_cytoGCaMP"
PLANE0 = f"C:/Users/dan/suite2p_v1_production/{SESSION}/output/suite2p/plane0"
OUT_DATA = Path(f"C:/Users/dan/suite2p_v1_custom_reg/{SESSION}")
BIN = 10            # frames per bin
CHUNK = 60          # frames per stitched chunk of a trial tiff
MAXS = 40           # px search range
SIGMA, BG_SIGMA = 1.5, 20.0


def prep(img, win):
    x = gaussian_filter(img.astype(np.float64), SIGMA)
    x = x - gaussian_filter(x, BG_SIGMA)          # keep anatomy, drop slow background
    x = (x - x.mean()) / (x.std() + 1e-12)
    return x * win


def xcorr_shift(ref_f, img):
    """Displacement (dy, dx) of img's content relative to the reference (img(p + d) ~ ref(p)),
    with ref_f = conj(FFT(ref)); and the normalised peak height."""
    c = np.fft.fftshift(np.real(np.fft.ifft2(ref_f * np.fft.fft2(img))))
    cy, cx = np.array(c.shape) // 2
    sub = c[cy - MAXS:cy + MAXS + 1, cx - MAXS:cx + MAXS + 1]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)

    def para(a, b, d):
        den = a - 2 * b + d
        return 0.0 if den == 0 else 0.5 * (a - d) / den
    dy = para(*sub[iy - 1:iy + 2, ix]) if 0 < iy < sub.shape[0] - 1 else 0.0
    dx = para(*sub[iy, ix - 1:ix + 2]) if 0 < ix < sub.shape[1] - 1 else 0.0
    return iy - MAXS + dy, ix - MAXS + dx


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def configure(session):
    """Point the module at another session (default: 20220915_fish1, this report's)."""
    global SESSION, PLANE0, OUT_DATA
    SESSION = session
    PLANE0 = f"C:/Users/dan/suite2p_v1_production/{SESSION}/output/suite2p/plane0"
    OUT_DATA = Path(f"C:/Users/dan/suite2p_v1_custom_reg/{SESSION}")


def trial_files():
    from pipeline import ophys_extraction as ox
    ops = ox.load_ops(PLANE0)
    tiff_dir = os.path.dirname(str(ops["file_list"][0]))
    return [(os.path.join(tiff_dir, n), start, n_fr) for n, start, n_fr in ox.trial_files(ops)]


def load_bins(path, n):
    """Mean, odd-frame mean and even-frame mean of each BIN-frame bin of one tiff."""
    mov = tifffile.imread(path).astype(np.float32)
    assert len(mov) == n, f"{path}: {len(mov)} frames, suite2p used {n}"
    nb = n // BIN
    b = mov[:nb * BIN].reshape(nb, BIN, *mov.shape[1:])
    return b.mean(1), b[:, 0::2].mean(1), b[:, 1::2].mean(1)


def measure(ref, bins, win):
    ref_f = np.conj(np.fft.fft2(prep(ref, win)))
    return np.array([xcorr_shift(ref_f, prep(b, win)) for b in bins])


def main(csv_path=None):
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    files = trial_files()
    with Pool(len(files)) as pool:
        loaded = pool.starmap(load_bins, [(p, n) for p, _, n in files])
    B = np.concatenate([x[0] for x in loaded])
    Bodd = np.concatenate([x[1] for x in loaded])
    Beven = np.concatenate([x[2] for x in loaded])
    meta = pd.DataFrame([dict(file=os.path.basename(p), file_index=i, bin_in_file=j,
                              frame=start + j * BIN)
                         for i, (p, start, n) in enumerate(files) for j in range(n // BIN)])
    Ly, Lx = B.shape[1:]
    win = np.outer(np.hanning(Ly), np.hanning(Lx))

    # pass 1: reference = mean of everything; pass 2: mean of the pass-1-aligned bins
    d1 = measure(B.mean(0), B, win)
    ref = np.mean([nd_shift(b, -d, order=1, mode="nearest") for b, d in zip(B, d1)], axis=0)
    d = measure(ref, B, win)
    d_odd, d_even = measure(ref, Bodd, win), measure(ref, Beven, win)
    prec = np.abs(d_odd - d_even) / 2

    # Residual and correlation after shifting each bin back by its displacement
    pref = prep(ref, win)
    ref_f = np.conj(np.fft.fft2(pref))
    resid, c_before, c_after = [], [], []
    for b, dd in zip(B, d):
        bs = nd_shift(b, -dd, order=1, mode="nearest")
        resid.append(xcorr_shift(ref_f, prep(bs, win)))
        c_before.append(corr(prep(b, win), pref))
        c_after.append(corr(prep(bs, win), pref))
    resid = np.array(resid)

    rows = meta.assign(dy=d[:, 0], dx=d[:, 1], disp=np.hypot(*d.T),
                       prec_dy=prec[:, 0], prec_dx=prec[:, 1],
                       resid_dy=resid[:, 0], resid_dx=resid[:, 1],
                       corr_before=c_before, corr_after=c_after)
    rows.to_csv(csv_path or _HERE / "results_bins.csv", index=False)

    # Per-frame displacement: bin values median-filtered over 3 bins within a chunk,
    # interpolated linearly between bin centres within the chunk, held at its edges.
    n_total = sum(n for _, _, n in files)
    dy_f, dx_f = np.zeros(n_total), np.zeros(n_total)
    for i, (p, start, n) in enumerate(files):
        sel = rows.file_index.values == i
        for c0 in range(0, n, CHUNK):
            inchunk = sel & (rows.bin_in_file.values * BIN >= c0) & (rows.bin_in_file.values * BIN < c0 + CHUNK)
            if not inchunk.any():
                continue
            centres = rows.frame.values[inchunk] + (BIN - 1) / 2
            frames = np.arange(start + c0, min(start + c0 + CHUNK, start + n))
            for arr, col in ((dy_f, "dy"), (dx_f, "dx")):
                v = median_filter(rows[col].values[inchunk], size=3, mode="nearest")
                arr[frames] = np.interp(frames, centres, v)
    early = rows.frame.values < 60          # the first minute of magneto_0
    settled = (rows.file_index.values == 0) & (rows.frame.values >= 600)
    np.savez_compressed(OUT_DATA / "shifts.npz", dy=dy_f, dx=dx_f, ref=ref,
                        mean_first_minute=B[early].mean(0),
                        mean_first_minute_corrected=np.mean(
                            [nd_shift(b, -dd, order=1, mode="nearest") for b, dd in zip(B[early], d[early])], 0),
                        mean_settled=B[settled].mean(0))
    print(f"  {len(rows)} bins; displacement max {rows.disp.max():.1f} px; "
          f"precision median {np.median(prec):.2f} px; |residual| 95th pct "
          f"{np.percentile(np.hypot(*resid.T), 95):.2f} px")
    print(f"  correlation with the reference: median {np.median(c_before):.3f} before, "
          f"{np.median(c_after):.3f} after; first minute of magneto_0 "
          f"{np.median(np.array(c_before)[early]):.3f} -> {np.median(np.array(c_after)[early]):.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=SESSION, help="production suite2p run folder name")
    ap.add_argument("--csv", help="per-bin results (default: results_bins.csv next to this script)")
    a = ap.parse_args()
    configure(a.session)
    main(a.csv)
