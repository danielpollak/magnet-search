"""Pick the example ROIs for seg_video.py, by a fixed rule (written to results_examples.csv).

    python docs/resegmentation/examples.py          # magneto2, after analyze.py

One example per session (20221001_fish1 aside, to keep it to four videos):
  * 20220915_fish1 magneto_0: the selected re-segmented ROI within 15 px of production ROI 58
    (the cell of docs/custom_registration.md section 4) whose trace correlates best with ROI 58's;
  * the others, trial magneto_1 (coil on, grating): among the whole-pixel drift-corrected run's analysed
    ROIs that respond to the grating (p < 0.01 at 1/60 Hz) and hold at least two of production's
    kept ROIs, the one collecting the most photons per frame, whose 96 px crop fits inside
    production's registered frame.
"""
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
TAG = "reseg_corrected_int"           # the recommended re-segmentation (docs/resegmentation.md)
HALF, LY, OFF = 48, 477, (53, -20)    # crop half-width; frame size; production offset (common.py)


def fits(y, x):
    y0, x0 = int(y) - HALF, int(x) - HALF
    return 0 <= y0 + OFF[0] <= LY - 2 * HALF and 0 <= x0 + OFF[1] <= LY - 2 * HALF and y0 >= 0 and x0 >= 0


def main():
    d = pd.read_csv(_HERE / "results_rois.csv")
    R = pd.read_csv(_HERE / "results_sessions_rois.csv")
    Fr = pd.read_csv(_HERE / "results_fragments.csv")
    fr = Fr[(Fr.kind == "reseg_roi") & (Fr.trace_set == TAG)]
    rows = []
    # 20220915_fish1: of the whole-pixel drift-corrected run's selected ROIs within 15 px of production ROI 58,
    # the one whose magneto_0 trace correlates best with ROI 58's
    import sys
    sys.path.insert(0, str(_HERE))
    import common as cm
    s = "2022_09_15-fish1_cytoGCaMP"
    p58 = np.load(cm.plane0("production", s) / "stat.npy", allow_pickle=True)[58]
    off = cm.offset("production", s)
    cy, cx = np.median(p58["ypix"]) - off[0], np.median(p58["xpix"]) - off[1]
    r = R[(R.session == s) & (R.trace_set == TAG) & R.selected]
    r = r[np.hypot(r.med_y - cy, r.med_x - cx) < 15]
    start, n = cm.trial_slice("production", s, "2022_09_15_fish1_magneto_0.tif")
    f58 = np.load(cm.plane0("production", s) / "F_full.npy", mmap_mode="r")[58, start:start + n]
    s0, _ = cm.trial_slice(TAG, s, "2022_09_15_fish1_magneto_0.tif")
    F = np.load(cm.plane0(TAG, s) / "F.npy", mmap_mode="r")
    c = [np.corrcoef(F[k, s0:s0 + n], f58)[0, 1] for k in r.roi]
    k58 = int(r.roi.values[int(np.argmax(c))])
    rows.append(dict(session=s, trial="magneto_0", roi=k58, tag=TAG,
                     rule=f"selected ROI near production ROI 58 whose trace matches it best (r = {max(c):.2f})"))
    for s in ("2022_09_14-fish2_cytoGCaMP", "2022_10_01-fish2_cytoGCaMP", "2022_10_02-fish2_cytoGCaMP"):
        name = f"engert_{cm.short(s)}_magneto_1"
        g = d[(d.experiment == name) & (d.trace_set == TAG) & (d.p_vis < 0.01)]
        g = g.merge(fr[fr.session == s][["roi", "n_prod", "n_prod_selected"]], on="roi")
        g = g.merge(R[(R.session == s) & (R.trace_set == TAG)][["roi", "med_y", "med_x"]], on="roi")
        g = g[(g.n_prod_selected >= 2) & np.array([fits(y, x) for y, x in zip(g.med_y, g.med_x)])]
        g = g.assign(photons=g.brightness * g.npix)      # roughly the photons it collects per frame
        r = g.sort_values("photons").iloc[-1]
        rows.append(dict(session=s, trial="magneto_1", roi=int(r.roi), tag=TAG,
                         rule=f"grating responder holding {int(r.n_prod_selected)} kept production ROIs "
                              f"({int(r.n_prod)} in all), most photons per frame"))
    out = pd.DataFrame(rows)
    out.to_csv(_HERE / "results_examples.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
