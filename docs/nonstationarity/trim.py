"""Nonstationarity of the 0.3 Hz zebrafish with the first or the last minutes of each trace removed
(docs/nonstationarity.md, Table 3).

    python docs/nonstationarity/trim.py --workers 24     # ~15 min

Runs nonstationarity.py's own run() on the twelve 0.3 Hz recordings (20220914_fish2,
20220915_fish1) with sv.load wrapped to drop TRIM_S seconds from the start or from the end of
every trace. Removing the end is the control for the shorter trace: a shorter trace has less power
to detect anything, so only the difference between removing the start and removing the end says
where the nonstationarity sits.

Output (next to this script): results_trim.csv, one row per recording x removed part, with the
share of ROIs with block-shuffle p < 0.05 for every metric and K (synthetic AR(1) rows excluded).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import nonstationarity as ns  # noqa: E402

FISH = ("20220914_fish2", "20220915_fish1")
TRIM_S = (180, 300)        # 3 and 5 minutes
OUT_CSV = _HERE / "results_trim.csv"
_load = ns.sv.load


def job(args):
    name, side, trim_s = args

    def load(n, **kw):
        cfg, F, roi = _load(n, **kw)
        k = int(round(trim_s / cfg.sample_period))
        if side == "start":
            return cfg, F[:, k:], roi
        return cfg, F[:, :F.shape[1] - k], roi
    ns.sv.load = load
    d, _ = ns.run(name)
    d = d[~d.synthetic]
    return dict(experiment=name, removed=side, removed_min=trim_s / 60, n_frames=int(d.N.iloc[0]),
                **{f"{k}_{K}": float((d[f"p_block_{k}_{K}"] < ns.ALPHA).mean())
                   for k, _ in ns.METRICS for K in ns.KS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    names = [n for n in ns.recordings() if any(f in n for f in FISH)]
    jobs = [(n, "none", 0) for n in names]
    jobs += [(n, side, t) for n in names for side in ("start", "end") for t in TRIM_S]
    with ProcessPoolExecutor(a.workers) as pool:
        t = pd.DataFrame(list(pool.map(job, jobs)))
    t.to_csv(OUT_CSV, index=False)
    t["fish"] = t.experiment.str.extract(f"({'|'.join(FISH)})", expand=False)
    t["coil"] = np.where(t.experiment.str.contains("no_magneto"), "off", "on")
    pd.set_option("display.width", 250)
    print("median over the three trials, share with block p < 0.05:")
    print(t.groupby(["fish", "coil", "removed", "removed_min"])
          [["n_frames", "rms_10", "band_10", "cv2_10", "rms_20", "band_20"]]
          .median().round(2).to_string())


if __name__ == "__main__":
    main()
