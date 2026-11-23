"""Indirect test for a changing focal plane, without a z-stack: does the image content drift
with acquisition time, after lateral alignment?

    python docs/zplane/similarity.py --workers 6      # magneto2 env; reads the NAS (read-only)

Every 60-frame chunk of a session's two coil conditions (61 + 61, in acquisition order by file
time) is averaged; chunk i's odd frames are cross-correlated with chunk j's even frames (same
preprocessing as zpos.py; normalised peak within +-MAXS px, so lateral motion is factored out).
Odd vs even keeps the photon noise of the two images independent, so the diagonal (i = j) is
the similarity of an image with itself at this photon count: the ceiling.

If the content is stable, the similarity stays at the ceiling at every time lag. A change of
focal plane (or of the tissue) makes it fall with lag. Bleaching alone does not: the score is a
correlation, blind to a change in overall brightness.

On 20221001_fish1, which has a stack, the decay can be checked against zpos.py's z.

Outputs: results_similarity.csv (one row per pair of chunks), results_similarity_summary.csv,
fig_similarity.png.
"""
import sys
import argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "custom_registration"))
import zpos  # noqa: E402
import measure  # noqa: E402

SESSIONS = {
    "2022_09_14-fish2_cytoGCaMP": ["magneto", "no-magneto"],
    "2022_09_15-fish1_cytoGCaMP": ["magneto", "no-magneto"],
    "2022_10_01-fish1_cytoGCamp": ["magneto", "no-magneto"],
    "2022_10_01-fish2_cytoGCaMP": ["magneto", "no-magneto"],
    "2022_10_02-fish2_cytoGCaMP": ["magneto", "no_magneto"],
    "Medaka Experiments/fish3_8dpf_gcamp_good": ["magneto", "nomagneto"],
}


def pairwise(odd, even):
    """S[i, j]: normalised xcorr peak of odd[i] against even[j]."""
    win = np.outer(np.hanning(odd.shape[1]), np.hanning(odd.shape[2]))
    A = [measure.prep(x, win) for x in odd]
    B = [measure.prep(x, win) for x in even]
    An, Bn = np.array([np.linalg.norm(a) for a in A]), np.array([np.linalg.norm(b) for b in B])
    FA = np.stack([np.fft.fft2(a) for a in A])
    FB = np.conj(np.stack([np.fft.fft2(b) for b in B]))
    M = measure.MAXS
    S = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        c = np.fft.fftshift(np.real(np.fft.ifft2(FA[i][None] * FB)), axes=(-2, -1))
        cy, cx = np.array(c.shape[1:]) // 2
        S[i] = c[:, cy - M:cy + M + 1, cx - M:cx + M + 1].max((-2, -1)) / (An[i] * Bn)
    return S


def run(session):
    d = zpos.load_session(session, 4, dirs=(SESSIONS[session], []))
    meta = pd.DataFrame(d["meta"])
    fn = meta[meta.kind == "func"].sort_values("mtime")
    S = pairwise(d["odd"][fn.index], d["even"][fn.index])
    t = (fn.mtime.values - fn.mtime.values.min()) / 60.0
    rows = []
    for a in range(len(fn)):
        for b in range(len(fn)):
            rows.append(dict(session=Path(session).name, i=a, j=b, folder_i=fn.folder.values[a],
                             folder_j=fn.folder.values[b], chunk_i=fn.chunk.values[a],
                             chunk_j=fn.chunk.values[b], t_i=t[a], t_j=t[b], S=S[a, b]))
    return pd.DataFrame(rows)


def summarize(P):
    P = P.assign(lag=(P.t_j - P.t_i).abs(), same=P.folder_i == P.folder_j)
    out = []
    for s, g in P.groupby("session"):
        ceil = g[g.i == g.j].S.median()
        w = g[g.same & (g.i != g.j)]
        r = dict(session=s, ceiling=ceil,
                 lag_le3=w[w.lag <= 3].S.median(),
                 lag_25_35=w[(w.lag >= 25) & (w.lag <= 35)].S.median(),
                 lag_ge50=w[w.lag >= 50].S.median(),
                 across_conditions=g[~g.same].S.median())
        out.append(r)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not a.summary_only:
        # loading is parallel inside zpos.load_session only on a cache miss; fill caches first
        for s in SESSIONS:
            zpos.load_session(s, 24, dirs=(SESSIONS[s], []))
        with Pool(a.workers) as pool:
            P = pd.concat(pool.map(run, list(SESSIONS)))
        P.to_csv(_HERE / "results_similarity.csv", index=False)
    P = pd.read_csv(_HERE / "results_similarity.csv")
    T = summarize(P)
    T.to_csv(_HERE / "results_similarity_summary.csv", index=False)
    print(T.round(3).to_string(index=False))
    C = calibrate(P)
    C.to_csv(_HERE / "results_similarity_vs_dz.csv", index=False)
    print(C.round(3).to_string(index=False))


def calibrate(P, session="2022_10_01-fish1_cytoGCamp", folder="magneto"):
    """20221001_fish1, coil on: similarity of two chunks against the difference of their z from
    the stack (zpos.py; usable chunks only, as in roi_z.py)."""
    Z = pd.read_csv(_HERE / "results_zpos.csv")
    z = Z[(Z.session == session) & ~Z.at_edge & (Z.peak_corr >= 0.35) & (Z.folder == folder)]
    z = z.set_index("chunk").z_est
    g = P[(P.session == session) & (P.folder_i == folder) & (P.folder_j == folder) & (P.i != P.j)]
    g = g.assign(dz=(g.chunk_i.map(z) - g.chunk_j.map(z)).abs()).dropna(subset=["dz"])
    rows = []
    for a, b in [(0, 1), (1, 2), (2, 4), (4, 6), (6, 8), (8, 12)]:
        w = g[(g.dz >= a) & (g.dz < b)]
        rows.append(dict(dz=f"{a}-{b}", pairs=len(w), similarity=w.S.median(),
                         lag_min=(w.t_i - w.t_j).abs().median()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
