"""Where in the anatomy z-stack does each minute of a photon-starved zebrafish recording sit?

    python docs/zplane/zpos.py --workers 24          # magneto2 env; reads the NAS (read-only)
    python docs/zplane/zpos.py --match-only          # reuse the cached mean images

Three of the five photon-starved zebrafish sessions have an anatomy z-stack on the NAS, taken
with the same rig and zoom as the functional recording: one 60-frame tiff per plane, the plane's
stage z in field 5 of the filename (<chunk>.<cond>.0.<rep>.<z>.1000.<zoom>.<zoom>.59.0.tif).
The functional recordings are 61 chunks of 60 frames per coil condition, in the same naming, so
the chunk index (field 1) gives the acquisition order.

Method (the photon-starved frames are too empty to place one at a time):
  - every stack plane and every functional chunk is averaged over its 60 frames (also over its
    odd and its even frames, for the precision check); repeats of a plane (field 4) are averaged;
  - images are smoothed and high-passed as in docs/custom_registration/measure.py (prep), and
    each chunk is cross-correlated with each plane; the score is the normalised peak, i.e. the
    correlation at the best lateral shift (within +-MAXS px), as suite2p's register_to_zstack
    does per frame (suite2p 1.1.0's compute_zpos itself is a stub that returns None);
  - the chunk's z is the parabola vertex through the best plane and its two neighbours.

Checks:
  - stack self-test: each plane's odd-frame mean must match its own even-frame mean best;
  - precision: z from a chunk's odd frames vs its even frames;
  - the stage z of the recording (filename) should lie near the matched z, if the stack's
    coordinates and the recording's are the same.

Outputs (docs/zplane/): results_zpos.csv (one row per chunk), results_stack_selftest.csv,
results_corr.npz (chunk x plane correlation), fig_zpos.png. Cached mean images:
C:/Users/dan/zplane_cache/<session>.npz (local disk).
"""
import re
import sys
import argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tifffile

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "custom_registration"))
import measure  # noqa: E402  (prep, xcorr_shift: same preprocessing as the drift measurement)

NAS = "//datanas/family/data_aggregated/Engert"
CACHE = Path("C:/Users/dan/zplane_cache")
SESSIONS = {
    # session: (functional condition folders, stack folders)
    "2022_10_01-fish1_cytoGCamp": (["magneto", "no-magneto"], ["anatomy stack no magneto"]),
    "2022_10_01-fish2_cytoGCaMP": (["magneto", "no-magneto"], ["anatomy stack no magneto"]),
    "2022_10_02-fish2_cytoGCaMP": (["magneto", "no_magneto"],
                                   ["anatomy_stack_magneto", "anatomy_stack-no_magneto"]),
}
NAME = re.compile(r"^(\d+)\.(\d+)\.0\.(\d+)\.(\d+)\.1000\.([\d.]+?)\.[\d.]+\.59\.0\.tif$")


def parse(path):
    m = NAME.match(Path(path).name)
    return None if m is None else dict(chunk=int(m[1]), cond=int(m[2]), rep=int(m[3]),
                                       z=int(m[4]), zoom=float(m[5]))


def tiffs(folder):
    """All acquisition tiffs under folder (the stacks keep each plane in its own subfolder)."""
    out = []
    for p in Path(folder).rglob("*.tif"):
        if "suite2p" in p.parts or "concatenated" in p.parts:
            continue
        info = parse(p)
        if info is not None:
            out.append((str(p), info))
    return sorted(out, key=lambda t: t[1]["chunk"])


def means(path):
    a = tifffile.imread(path).astype(np.float32)
    return a.mean(0), a[0::2].mean(0), a[1::2].mean(0), Path(path).stat().st_mtime


def load_session(session, workers, dirs=None):
    """Mean images of every chunk (and stack plane) of a session, cached on local disk. dirs:
    (functional folders, stack folders) under NAS/session; default SESSIONS[session]."""
    cache = CACHE / f"{Path(session).name}.npz"
    if cache.exists():
        return dict(np.load(cache, allow_pickle=True))
    func_dirs, stack_dirs = dirs if dirs is not None else SESSIONS[session]
    rows, paths = [], []
    for kind, dirs_ in (("func", func_dirs), ("stack", stack_dirs)):
        for d in dirs_:
            for p, info in tiffs(f"{NAS}/{session}/{d}"):
                rows.append(dict(kind=kind, folder=d, path=p, **info))
                paths.append(p)
    with Pool(workers) as pool:
        out = pool.map(means, paths)
    meta = pd.DataFrame(rows)
    meta["mtime"] = [o[3] for o in out]
    CACHE.mkdir(exist_ok=True)
    d = dict(full=np.stack([o[0] for o in out]), odd=np.stack([o[1] for o in out]),
             even=np.stack([o[2] for o in out]), meta=meta.to_records(index=False))
    np.savez(cache, **d)
    return d


def score(ref_imgs, imgs, win):
    """Normalised cross-correlation peak (within +-MAXS px) of every img against every ref."""
    R = [measure.prep(r, win) for r in ref_imgs]
    Rf = [np.conj(np.fft.fft2(r)) for r in R]
    Rn = [np.linalg.norm(r) for r in R]
    S = np.zeros((len(imgs), len(ref_imgs)))
    M = measure.MAXS
    for i, im in enumerate(imgs):
        x = measure.prep(im, win)
        X, xn = np.fft.fft2(x), np.linalg.norm(x)
        for j in range(len(R)):
            c = np.fft.fftshift(np.real(np.fft.ifft2(Rf[j] * X)))
            cy, cx = np.array(c.shape) // 2
            S[i, j] = c[cy - M:cy + M + 1, cx - M:cx + M + 1].max() / (xn * Rn[j])
    return S


def vertex(zs, s):
    """z of the parabola through the best plane and its neighbours; edge -> the edge plane."""
    j = int(np.argmax(s))
    if j == 0 or j == len(s) - 1:
        return float(zs[j]), True
    a, b, c = s[j - 1:j + 2]
    den = a - 2 * b + c
    off = 0.0 if den == 0 else 0.5 * (a - c) / den
    return float(zs[j] + off * (zs[j + 1] - zs[j - 1]) / 2), False


def match(session, d):
    meta = pd.DataFrame(d["meta"])
    full, odd, even = d["full"], d["odd"], d["even"]
    win = np.outer(np.hanning(full.shape[1]), np.hanning(full.shape[2]))
    st = meta[meta.kind == "stack"]
    # one stack per session: if there are several (20221002_fish2), each is matched separately
    rows, selftest, corr = [], [], {}
    for folder, s in st.groupby("folder"):
        zs = np.sort(s.z.unique())
        planes = np.stack([full[s.index[s.z == z]].mean(0) for z in zs])
        p_odd = np.stack([odd[s.index[s.z == z]].mean(0) for z in zs])
        p_even = np.stack([even[s.index[s.z == z]].mean(0) for z in zs])
        # self-test: odd half of each plane against the even halves of all planes
        S = score(p_even, p_odd, win)
        for j, z in enumerate(zs):
            zh, edge = vertex(zs, S[j])
            selftest.append(dict(session=session, stack_dir=folder, z=z, best=zs[np.argmax(S[j])],
                                 z_est=zh, own_corr=S[j, j], next_best=np.sort(S[j])[-2]))
        fn = meta[meta.kind == "func"]
        Sf = score(planes, full[fn.index], win)
        So = score(planes, odd[fn.index], win)
        Se = score(planes, even[fn.index], win)
        corr[f"{session}|{folder}"] = dict(zs=zs, S=Sf, chunk=fn.chunk.values, cond_folder=fn.folder.values)
        for k, (i, r) in enumerate(fn.iterrows()):
            z, edge = vertex(zs, Sf[k])
            zo, _ = vertex(zs, So[k])
            ze, _ = vertex(zs, Se[k])
            rows.append(dict(session=session, stack_dir=folder, folder=r.folder, chunk=r.chunk,
                             cond=r.cond, stage_z=r.z, mtime=r.mtime, z_est=z, at_edge=edge,
                             z_odd=zo, z_even=ze, peak_corr=Sf[k].max()))
    return pd.DataFrame(rows), pd.DataFrame(selftest), corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--match-only", action="store_true")
    ap.add_argument("--sessions", nargs="*", default=list(SESSIONS))
    a = ap.parse_args()
    Z, T, C = [], [], {}
    for s in a.sessions:
        d = load_session(s, a.workers)
        z, t, c = match(s, d)
        Z.append(z); T.append(t); C.update(c)
        print(s, "chunks", len(z) // max(1, z.stack_dir.nunique()), "planes", len(t))
    Z, T = pd.concat(Z), pd.concat(T)
    Z.to_csv(_HERE / "results_zpos.csv", index=False)
    T.to_csv(_HERE / "results_stack_selftest.csv", index=False)
    np.savez(_HERE / "results_corr.npz", **{k.replace("|", "__"): np.array(v, dtype=object)
                                             for k, v in C.items()})
    print(T.assign(ok=T.best == T.z).groupby(["session", "stack_dir"]).ok.mean())
    print(Z.groupby(["session", "stack_dir", "folder"]).agg(
        stage_z=("stage_z", "first"), z_med=("z_est", "median"), z_min=("z_est", "min"),
        z_max=("z_est", "max"), edge=("at_edge", "mean"), peak=("peak_corr", "median")).to_string())
    Z["half_diff"] = Z.z_odd - Z.z_even
    print(Z.groupby(["session", "stack_dir"]).half_diff.agg(lambda v: np.std(v) / 2).rename("z precision (sd/2)"))


if __name__ == "__main__":
    main()
