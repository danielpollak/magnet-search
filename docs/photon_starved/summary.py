"""The numbers in docs/photon_starved_imaging.md: ROI size and how often a trace sits on its floor,
before and after the fixes, for the photon-starved zebrafish and medaka.

    python docs/photon_starved/summary.py          # magneto2 env, ~2 min (reads the NAS, read-only)

Segmentations ("runs"):
  zebrafish (the five sessions of docs/resegmentation.md)
    before   the previous production run: suite2p v1 on the tiffs as acquired, suite2p's
             registration (one constant shift), C:/Users/dan/suite2p_v1_production/<session>/
    after    the current production run: tiffs x64, drift corrected in whole pixels, suite2p's
             registration off, <session>_x64_drift/
  medaka (one fish, six trials)
    before   the original suite2p 0.10.1 runs on the NAS, one per trial, tiffs as acquired (the
             suite2p v1 runs on the as-acquired tiffs were deleted on 2026-10-12)
    after    the current production run: the six trials as one movie, tiffs x64, suite2p's own
             registration (it works on x64 medaka), fish3_8dpf_concat_x64/

Per ROI:
  npix, P(iscell); "selected" = P(iscell) > 0.5 and npix >= 10 (the analysis thresholds; the fish
  outline and flatline removal are left out here so every run is treated alike);
  floor share = share of the session's frames on which the trace equals its own minimum, for
    suite2p's own trace (F.npy: halved and, after a warp, truncated to integers) and for the
    full-resolution trace (F_full.npy: no halving, no truncation; there the floor is a frame
    in which none of the ROI's pixels caught a photon).
Truncation is what separates suite2p's trace from the full-resolution trace of the same ROIs;
an empty frame is what remains in the full-resolution trace, and is rarer the more pixels an
ROI has.

Nonstationarity (share of ROIs with block-shuffle p < 0.05 at K = 10, docs/nonstationarity.md),
median over recordings:
  zebrafish before / after   from docs/resegmentation/results_experiments.csv (previous
                             production vs the run now in production)
  medaka after               computed here with docs/nonstationarity/nonstationarity.py's run()
                             on the six medaka recordings (their production NWB files)

Interpretation (the doc keeps these apart):
  truncation   suite2p's trace on its floor while the full-resolution trace of the same ROI is
               not: removed for the analysed traces by the full-resolution extraction
               (2026-10-05, docs/cv2.md), and from suite2p's own traces by the x64 tiffs
  empty frames frames with no photon in the ROI; a warp's interpolation hides them (a frame
               rarely lands exactly on the minimum), so their share is comparable only between
               unwarped traces: here the zebrafish "after" runs

Outputs: results_rois.csv (per ROI), results_summary.csv (per run), results_nonstationarity.csv,
results_fig2.csv (Fig 2's flagged recordings and zebrafish neurons, before and after the switch).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
PROD = Path("C:/Users/dan/suite2p_v1_production")
NAS_MEDAKA = Path("//datanas/family/data_aggregated/Engert/Medaka Experiments/fish3_8dpf_gcamp_good")
ZF = ["2022_09_14-fish2_cytoGCaMP", "2022_09_15-fish1_cytoGCaMP", "2022_10_01-fish1_cytoGCamp",
      "2022_10_01-fish2_cytoGCaMP", "2022_10_02-fish2_cytoGCaMP"]
MEDAKA_TRIALS = [f"fish3_8dpf_{c}_{t}" for c in ("magneto", "no_magneto") for t in (0, 1, 2)]


def floor_share(F):
    F = np.asarray(F, np.float32)
    return (F == F.min(axis=1, keepdims=True)).mean(axis=1)


def run_rows(plane0, species, session, when, full=True):
    """One row per ROI of one suite2p run."""
    stat = np.load(plane0 / "stat.npy", allow_pickle=True)
    p = np.load(plane0 / "iscell.npy")[:, 1]
    npix = np.array([s["npix"] for s in stat])
    d = pd.DataFrame(dict(species=species, session=session, when=when, roi=np.arange(len(stat)),
                          npix=npix, p_iscell=p, selected=(p > 0.5) & (npix >= 10)))
    d["floor_suite2p"] = floor_share(np.load(plane0 / "F.npy"))
    d["floor_full"] = floor_share(np.load(plane0 / "F_full.npy")) if full else np.nan
    return d


def ns_medaka(name):
    sys.path.insert(0, str(_HERE.parent / "nonstationarity"))
    import nonstationarity as ns
    d, _ = ns.run(name)
    d = d[~d.synthetic]
    return dict(species="medaka", when="after", experiment=name, rois=len(d),
                rms_10=(d.p_block_rms_10 < 0.05).mean(), band_10=(d.p_block_band_10 < 0.05).mean(),
                cv2_10=(d.p_block_cv2_10 < 0.05).mean())


def nonstationarity():
    E = pd.read_csv(_HERE.parent / "resegmentation" / "results_experiments.csv")
    E = E[E.trace_set.isin(["production", "reseg_corrected_int"])]
    E = E.assign(species="zebrafish", when=np.where(E.trace_set == "production", "before", "after"))
    names = [f"medaka_fish3_8dpf_{c}_{t}" for c in ("magneto", "no_magneto") for t in (0, 1, 2)]
    with ProcessPoolExecutor(6) as pool:
        M = pd.DataFrame(list(pool.map(ns_medaka, names)))
    cols = ["species", "when", "experiment", "rois", "rms_10", "band_10", "cv2_10"]
    N = pd.concat([E.assign(experiment=E.experiment)[cols + ["batch", "coil"]], M[cols]], ignore_index=True)
    N.to_csv(_HERE / "results_nonstationarity.csv", index=False)
    out = N.groupby(["species", "when"], sort=False)[["rms_10", "band_10", "cv2_10"]].agg(["median", "min", "max"])
    print(out.round(3).to_string())
    return N


BEFORE_PARQUET = Path("C:/Users/dan/nwb_backup_pre_reseg_2026-10-12/all_fourier_df.parquet")


def fig2_counts():
    """Fig 2's flagged sessions and the zebrafish neuron count, from the production parquet and from
    the parquet before the zebrafish re-segmentation (its backup was deleted after this ran on
    2026-10-12; the "before" rows already in results_fig2.csv are then kept)."""
    repo = _HERE.parents[1]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "pipeline" / "manuscript"))
    import fig2
    from magpyneto2 import statistics as st
    rows = []
    for when, p in (("before", BEFORE_PARQUET),
                    ("after", Path(os.environ.get("NS_DATA_DIR", repo / "data")) / "manuscript" / "all_fourier_df.parquet")):
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        neg, pos, _ = st.get_poscontrols_negresults(df)
        for group, d in (("magnetic", neg), ("visual & auditory", pos)):
            flags = {k[2]: fig2.flag_sessions_with_excess_suspects(r, 1)[0]
                     for k, r in d.groupby(["species", "area", "rec"])}
            rows.append(dict(when=when, group=group, recordings=len(flags), flagged=sum(flags.values()),
                             zebrafish_flagged=" ".join(sorted(k for k, f in flags.items() if f and "engert" in k)),
                             zebrafish_neurons=d[d.species == "zebrafish"].groupby(["ID", "date", "id"]).ngroups))
    out = pd.DataFrame(rows)
    old = _HERE / "results_fig2.csv"
    if not BEFORE_PARQUET.exists() and old.exists():
        # the pre-switch parquet was deleted on 2026-10-12; keep the "before" rows computed from it
        prev = pd.read_csv(old)
        out = pd.concat([prev[prev.when == "before"], out], ignore_index=True)
    out.to_csv(old, index=False)
    print(out.drop(columns="zebrafish_flagged").to_string(index=False))


def main():
    rows = []
    for s in ZF:
        rows.append(run_rows(PROD / s / "output/suite2p/plane0", "zebrafish", s, "before"))
        rows.append(run_rows(PROD / f"{s}_x64_drift" / "output/suite2p/plane0", "zebrafish", s, "after"))
    for t in MEDAKA_TRIALS:      # no full-resolution traces were ever extracted for these
        rows.append(run_rows(NAS_MEDAKA / t / "suite2p/plane0", "medaka", t, "before", full=False))
    rows.append(run_rows(PROD / "fish3_8dpf_concat_x64/output/suite2p/plane0", "medaka",
                         "fish3_8dpf (6 trials)", "after"))
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(_HERE / "results_rois.csv", index=False)

    sel = d[d.selected]
    S = d.groupby(["species", "when"], sort=False).agg(
        runs=("session", "nunique"), rois=("roi", "size"),
        npix_median_all=("npix", "median"), share_under_10px=("npix", lambda v: np.mean(v < 10)))
    S["selected"] = sel.groupby(["species", "when"]).size()
    S["selected_per_run"] = S.selected / S.runs
    S["npix_median_selected"] = sel.groupby(["species", "when"]).npix.median()
    S["floor_suite2p_selected"] = sel.groupby(["species", "when"]).floor_suite2p.median()
    S["floor_full_selected"] = sel.groupby(["species", "when"]).floor_full.median()
    S = S.reset_index()
    S.to_csv(_HERE / "results_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))
    per = sel.groupby(["species", "session", "when"], sort=False).agg(
        selected=("roi", "size"), npix=("npix", "median"),
        floor_suite2p=("floor_suite2p", "median"), floor_full=("floor_full", "median"))
    print(per.round(3).to_string())
    nonstationarity()
    fig2_counts()


if __name__ == "__main__":
    main()
