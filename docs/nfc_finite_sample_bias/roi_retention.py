"""How many ROIs survive each inclusion step, and how many would at a lower P(iscell) threshold?

    python docs/nfc_finite_sample_bias/roi_retention.py     # ~3 min

Builds one table with every ROI of every imaging recording in the Fig 2C magnetic pool plus
the visual-only 2022 Q1 recordings (coverage_vs_roi_quality.recordings), and one column per
inclusion step, so retention under any combination of steps is a groupby away:

    p_iscell, npix          suite2p classifier probability and pixel count
    iscell_threshold,       the recording's current YAML thresholds (a ROI passes if strictly
      npix_threshold        above both, as in _load_from_nwb)
    inside_share, inside    share of the ROI's pixels inside the fish outline, and whether that
                            is >= 0.5 (pipeline/body_outline.py, the YAML's body_outline block)
    not_flat                survives remove_flatlines (a per-trace test, so it is computed for
                            every ROI, not only for those passing the other steps)
    coverage, active_frac,  activity_coverage.activity: share of 60 s windows with >= 3 frames
      clipped               above the trace's modal floor, share of such frames, and whether
                            the trace is floor-clipped
    analysed                the production population: thresholds, inside and not_flat; should
                            match the analysis stage's own count (checked below)

One row per ROI per recording: repeat trials of a fish share the ROIs (same segmentation) but
not the traces, so not_flat and coverage can differ between trials.

The summary counts ROIs per field of view, as the earlier outline tables did: a ROI counts at
a step if it passes that step in at least one of the field of view's recordings in the set.
Every column requires npix above the current npix threshold (10 everywhere) and the ROI to be
inside the outline; the columns then add, left to right, the P(iscell) threshold, flatline
removal and a coverage threshold (coverage >= COVERAGE_MIN). The total row sums the magnetic
sets; the visual-only set is left out because its ROIs are the 2022 Q1 ones again.

    python docs/nfc_finite_sample_bias/roi_retention.py --summary   # summary from the saved CSV

Outputs (next to this script, gitignored): results_roi_retention.csv,
results_roi_retention_summary.csv; the summary is also printed as a markdown table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from activity_coverage import activity  # noqa: E402
from coverage_vs_roi_quality import SETS, VISUAL, recordings, segmentation_key  # noqa: E402
import slow_variation as sv  # noqa: E402
from slow_variation import schema  # noqa: E402
from magpyneto2.engert_helpers import remove_flatlines  # noqa: E402
from pipeline import body_outline, nwb_io  # noqa: E402

OUT_CSV = _HERE / "results_roi_retention.csv"
OUT_SUMMARY = _HERE / "results_roi_retention_summary.csv"
ISCELL_FLOOR = 0.5         # lowest P(iscell) threshold considered
COVERAGE_MIN = 0.1         # active in at least 10% of the 60 s windows


def load(name, batch):
    cfg = schema.load_experiment(str(sv._REPO / "experiments" / f"{name}.yml"))
    io_r, nwbfile = nwb_io.read_nwbfile(cfg.nwb_path())
    F, roi_df = nwb_io.read_roi_data(nwbfile, "suite2p")   # the traces this report describes
    mean_img = nwb_io.read_mean_image(nwbfile)
    fourier = nwb_io.read_fourier_results_as_full_fourier_df(nwbfile)
    n_analysed = int(np.isclose(fourier["freq"], cfg.analysis.f).sum())
    io_r.close()
    inside_share = body_outline.inside_share(
        [body_outline.pixel_mask_yx(m) for m in roi_df["pixel_mask"]],
        body_outline.body(mean_img, body_outline.params_for(cfg.body_outline)))
    not_flat = np.zeros(len(F), bool)
    not_flat[remove_flatlines(F)[3]] = True
    N = min(int(120 * (F.shape[1] // 60)), F.shape[1])   # same frames as coverage_vs_roi_quality
    d = pd.DataFrame(dict(experiment=name, batch=batch, segmentation=segmentation_key(name),
                          roi=np.arange(len(F)), p_iscell=roi_df["p_iscell"].values,
                          npix=roi_df["npix"].values, iscell_threshold=cfg.iscell_threshold,
                          npix_threshold=cfg.npix_threshold, inside_share=inside_share,
                          inside=inside_share >= body_outline.INSIDE_SHARE, not_flat=not_flat,
                          **activity(F[:, :N].astype(float), cfg.sample_period)))
    d["analysed"] = ((d.p_iscell > d.iscell_threshold) & (d.npix > d.npix_threshold)
                     & d.inside & d.not_flat)
    return d, n_analysed


def summary(df):
    base = (df.npix > df.npix_threshold) & df.inside
    steps = {
        "current P(iscell)": base & (df.p_iscell > df.iscell_threshold),
        "current P(iscell) + flatline (analysed now)": base & (df.p_iscell > df.iscell_threshold) & df.not_flat,
        f"P(iscell) > {ISCELL_FLOOR}": base & (df.p_iscell > ISCELL_FLOOR),
        f"P(iscell) > {ISCELL_FLOOR} + flatline": base & (df.p_iscell > ISCELL_FLOOR) & df.not_flat,
        f"P(iscell) > {ISCELL_FLOOR} + flatline + coverage >= {COVERAGE_MIN:g}":
            base & (df.p_iscell > ISCELL_FLOOR) & df.not_flat & (df.coverage >= COVERAGE_MIN),
    }
    flags = df[["batch", "segmentation", "roi"]].assign(**steps)
    per_fov = flags.groupby(["batch", "segmentation", "roi"], sort=False)[list(steps)].any()
    out = per_fov.groupby("batch", sort=False).sum()
    out = out.reindex([b for b in SETS if b in out.index])
    out.loc["total (magnetic sets)"] = out.drop(index=VISUAL, errors="ignore").sum()
    return out


def print_summary(df):
    s = summary(df)
    s.to_csv(OUT_SUMMARY)
    print("\n| set of recordings | " + " | ".join(s.columns) + " |")
    print("|---" * (len(s.columns) + 1) + "|")
    for b, r in s.iterrows():
        print(f"| {b} | " + " | ".join(f"{v:,}" for v in r.values) + " |")


def main():
    if "--summary" in sys.argv:
        print_summary(pd.read_csv(OUT_CSV))
        return
    rows, check = [], []
    for name, batch in recordings():
        d, n_analysed = load(name, batch)
        rows.append(d)
        check.append((name, int(d["analysed"].sum()), n_analysed))
        print(f"  {name}: {len(d)} ROIs, {int(d['analysed'].sum())} analysed "
              f"(analysis stage: {n_analysed})", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    bad = [c for c in check if c[1] != c[2]]
    print("analysed count matches the analysis stage for every recording" if not bad
          else f"MISMATCH with the analysis stage: {bad}")
    print_summary(df)
    print("\nwrote", OUT_CSV.name, OUT_SUMMARY.name)


if __name__ == "__main__":
    main()
