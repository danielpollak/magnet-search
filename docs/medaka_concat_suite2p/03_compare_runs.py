"""Task A comparison: separate per-trial suite2p segmentations vs one
segmentation over all six trials concatenated.

Runs compared (see 02_run_suite2p.py):
  separate_nas    original per-trial runs on the NAS (suite2p 0.10.1) -- what production uses
  separate_local  same per-trial runs redone locally with suite2p 0.14.4 + builtin classifier (control
                  for version/classifier differences vs the concatenated runs)
  concat          suite2p 0.14.4 over the 6 trials concatenated, original ops (registration is a no-op)
  concat_aligned  same, after rigid integer pre-alignment of the no_magneto block (-2 px in y)

ENV:  magneto2
NAS:  required (READ-ONLY) for separate_nas; the rest is local SCRATCH (incl. the tiff copies).
TIME: ~3-5 min (mostly reading F / data.bin and the Fourier fits).

Outputs (next to this script):
  roi_counts.csv               total / iscell&npix / included (after remove_flatlines), per run x trial
  registration_meanimg_corr.csv  Pearson r between per-trial mean images (raw tiffs, aligned, data.bin)
  raw_pixel_value_distribution.csv  raw uint16 value histogram (photon-starved data, cf. suite2p //2)
  fourier_pvalue_summary.csv   per run x trial x freq: N, frac p<0.05, max |ECDF(p)-p|
  fig_roi_ecdfs.png            P(iscell) and npix ECDFs + joint P(iscell) x npix histograms
  fig_roi_footprints.png       included-ROI footprint maps (separate magneto_0 vs concat runs)
  fig_registration.png         per-trial mean images (concat runs), correlation matrices, offsets
  fig_pvalue_ecdf.png          ECDF(p) - p with binomial 95% band, per frequency, per run
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (HERE, TRIALS, SHORT, nas_plane0, load_plane0, run_plane0,
                    CONCAT_DIR, CONCAT_ALIGNED_DIR, ISCELL_THRES, NPIX_THRES,
                    F_MAG, Q_FRAC_MAG, F_VIS, Q_FRAC_VIS, T_SAMPLE, footprint_mask)
from magpyneto2.engert_helpers import fit_Fourier, remove_flatlines
from magpyneto2.statistics import corrected_pvalues

RUNS = ["separate_nas", "separate_local", "concat", "concat_aligned"]
COL = {"separate_nas": "k", "separate_local": "tab:gray", "concat": "tab:orange", "concat_aligned": "tab:blue"}
FR = pd.read_csv(HERE / "concat_frame_ranges.csv")


def concat_plane0(run):
    return (CONCAT_DIR if run == "concat" else CONCAT_ALIGNED_DIR) / "suite2p" / "plane0"


def per_trial_data(run):
    """Yield (trial, stat, iscell, F_trial[ndarray], ops) -- for concat runs the
    stat/iscell are shared and F is the trial's frame slice."""
    if run.startswith("separate"):
        for t in TRIALS:
            d = load_plane0(nas_plane0(t) if run == "separate_nas" else run_plane0("separate", t))
            yield t, d["stat"], d["iscell"], np.asarray(d["F"]), d["ops"]
    else:
        d = load_plane0(concat_plane0(run))
        for t, r in zip(TRIALS, FR.itertuples()):
            yield t, d["stat"], d["iscell"], np.asarray(d["F"][:, r.frame_start:r.frame_stop]), d["ops"]


def inclusion(stat, iscell, F):
    npix = np.array([s["npix"] for s in stat])
    m = (iscell[:, 1] > ISCELL_THRES) & (npix > NPIX_THRES)
    _, _, _, keep = remove_flatlines(F[m])
    inc = np.zeros(len(stat), bool); inc[np.where(m)[0][keep]] = True
    return m, inc


def fourier_p(F):
    out = {}
    for f, q in ((F_MAG, Q_FRAC_MAG), (F_VIS, Q_FRAC_VIS)):
        NFC, _, _, _, M, _ = fit_Fourier(F, T=T_SAMPLE, f=f, Q_frac=q)
        out[f] = corrected_pvalues(np.array(NFC), M)
    return out


def main():
    count_rows, prows, pvals, roi_props, inc_masks = [], [], {}, {}, {}
    for run in RUNS:
        inc_all = None
        for t, stat, isc, F, ops in per_trial_data(run):
            m, inc = inclusion(stat, isc, F)
            inc_all = inc.copy() if inc_all is None else (inc_all & inc) if run.startswith("concat") else None
            count_rows.append({"run": run, "trial": SHORT[t], "n_roi": len(stat),
                               "n_iscell_npix": int(m.sum()), "n_included": int(inc.sum()),
                               # ROIs whose centroid is within 10 px of the frame border: the
                               # concatenated runs pick up many streak-like ROIs along the
                               # registration-fill border (see fig_roi_footprints.png)
                               "n_included_within_10px_of_edge": int(
                                   (lambda med: ((med < 10) | (med > np.array([ops["Ly"], ops["Lx"]]) - 11)).any(1))(
                                       np.array([s["med"] for s in stat])[inc]).sum())})
            inc_masks[(run, t)] = inc
            if (run, "props") not in roi_props or run.startswith("separate"):
                roi_props.setdefault(run, []).append(
                    (isc[:, 1], np.array([s["npix"] for s in stat]), stat, ops))
            ps = fourier_p(F[inc])
            for f, p in ps.items():
                pvals.setdefault((run, f), []).append(p)
                N = len(p); x = np.sort(p); e = np.arange(1, N + 1) / N
                prows.append({"run": run, "trial": SHORT[t], "freq": round(f, 5), "N": N,
                              "frac_p_lt_0.05": round(float(np.mean(p < 0.05)), 4),
                              "max_abs_ecdf_minus_p": round(float(np.max(np.abs(e - x))), 4)})
            if run.startswith("concat"):
                roi_props[(run, "props")] = True
        if run.startswith("concat"):
            count_rows.append({"run": run, "trial": "included_in_all_6", "n_roi": len(stat),
                               "n_iscell_npix": int(m.sum()), "n_included": int(inc_all.sum())})
            union = np.any([inc_masks[(run, t)] for t in TRIALS], axis=0)
            count_rows.append({"run": run, "trial": "included_in_any", "n_roi": len(stat),
                               "n_iscell_npix": int(m.sum()), "n_included": int(union.sum())})
        else:
            tot = [r for r in count_rows if r["run"] == run]
            count_rows.append({"run": run, "trial": "sum_over_trials",
                               "n_roi": sum(r["n_roi"] for r in tot),
                               "n_iscell_npix": sum(r["n_iscell_npix"] for r in tot),
                               "n_included": sum(r["n_included"] for r in tot)})
    pd.DataFrame(count_rows).to_csv(HERE / "roi_counts.csv", index=False)
    pd.DataFrame(prows).to_csv(HERE / "fourier_pvalue_summary.csv", index=False)
    print(pd.DataFrame(count_rows).to_string())
    print(pd.DataFrame(prows).to_string())

    # ---- ROI ECDFs + joint histograms
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    for run in RUNS:
        props = roi_props[run]
        pis = np.concatenate([p[0] for p in props]); npx = np.concatenate([p[1] for p in props])
        lab = f"{run} ({'6 runs pooled' if run.startswith('separate') else 'one run'})"
        axs[0, 0].plot(np.sort(pis), np.linspace(0, 1, len(pis)), color=COL[run], label=lab)
        axs[0, 1].plot(np.sort(npx), np.linspace(0, 1, len(npx)), color=COL[run], label=lab)
    axs[0, 0].axvline(ISCELL_THRES, ls="--", c="r"); axs[0, 0].set_xlabel("P(iscell)"); axs[0, 0].set_ylabel("ECDF"); axs[0, 0].legend(fontsize=7)
    axs[0, 1].axvline(NPIX_THRES, ls="--", c="r"); axs[0, 1].set_xscale("log"); axs[0, 1].set_xlabel("npix")
    axs[0, 2].axis("off"); axs[0, 3].axis("off")
    axs[0, 2].text(0, 0.5, pd.DataFrame(count_rows).query("trial in ['sum_over_trials','included_in_all_6','included_in_any','magneto_0']")
                   .to_string(index=False), family="monospace", fontsize=7, va="center")
    for ax, run in zip(axs[1], RUNS):
        props = roi_props[run][0]  # magneto_0 for separate runs; the single run for concat
        h = ax.hist2d(props[0], np.log10(np.maximum(props[1], 1)), bins=[np.linspace(0, 1, 41), np.linspace(0, 3, 41)],
                      cmap="magma", norm=matplotlib.colors.LogNorm())
        ax.axvline(ISCELL_THRES, c="c", ls="--"); ax.axhline(np.log10(NPIX_THRES), c="c", ls="--")
        ax.set_title(f"{run}" + (" (magneto_0)" if run.startswith("separate") else " (all 6 trials)"), fontsize=9)
        ax.set_xlabel("P(iscell)"); ax.set_ylabel("log10 npix")
    fig.tight_layout(); fig.savefig(HERE / "fig_roi_ecdfs.png", dpi=110)

    # ---- footprint maps
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.3))
    panels = [("separate_nas", TRIALS[0]), ("separate_nas", TRIALS[3]), ("concat", TRIALS[0]), ("concat_aligned", TRIALS[0])]
    for ax, (run, t) in zip(axs, panels):
        props = roi_props[run][TRIALS.index(t) if run.startswith("separate") else 0]
        stat, ops = props[2], props[3]
        Ly, Lx = ops["Ly"], ops["Lx"]
        mimg = ops["meanImg"]; lo, hi = np.percentile(mimg, [1, 99.5])
        ax.imshow(np.clip((mimg - lo) / (hi - lo), 0, 1), cmap="gray")
        inc = inc_masks[(run, t)]
        if run.startswith("concat"):
            inc = np.any([inc_masks[(run, tt)] for tt in TRIALS], axis=0)
        lab = np.zeros((Ly, Lx, 4))
        rng = np.random.default_rng(0)
        for k in np.where(inc)[0]:
            lab[footprint_mask(stat[k], Ly, Lx)] = (*rng.uniform(0.2, 1, 3), 0.8)
        ax.imshow(lab); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{run} {'' if run.startswith('concat') else SHORT[t]}: {inc.sum()} included ROIs", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "fig_roi_footprints.png", dpi=110)

    # ---- registration / inter-trial similarity
    # Ground truth for inter-trial similarity is the RAW tiff mean image
    # (local copies). suite2p's own registration is degenerate on this data
    # (refImg ~flat; rigid offsets 0; a constant nonrigid block shift applied
    # to every frame), and its int16 binary loses information to the uint16
    # `//2` conversion (values 100 and 101 both become 50 -- this data is
    # photon-starved, with almost all pixel values in 100..103), so data.bin
    # means are shown only as a secondary row.
    import tifffile
    from common import TIFF_DIR
    raw = {t: tifffile.imread(TIFF_DIR / f"{t}.tif") for t in TRIALS}
    vals, cnts = np.unique(raw[TRIALS[0]][::10], return_counts=True)
    q_rows = [{"trial": SHORT[TRIALS[0]], "value": int(v), "frac_of_pixel_frames": round(c / cnts.sum(), 5)}
              for v, c in zip(vals, cnts) if c / cnts.sum() > 1e-4]
    pd.DataFrame(q_rows).to_csv(HERE / "raw_pixel_value_distribution.csv", index=False)
    print(pd.DataFrame(q_rows).to_string())
    sh = pd.read_csv(HERE / "concat_aligned_shifts.csv").set_index("trial")
    rows_def = {}
    rows_def["raw_tiff"] = [raw[t].mean(0, dtype=np.float64) for t in TRIALS]
    rows_def["raw_tiff_aligned(-2px no_magneto)"] = [
        np.roll(m, (int(sh.loc[t, "applied_dy"]), int(sh.loc[t, "applied_dx"])), axis=(0, 1))
        for t, m in zip(TRIALS, rows_def["raw_tiff"])]
    del raw
    for run in ("concat", "concat_aligned"):
        ops = load_plane0(concat_plane0(run), load_F=False)["ops"]
        reg = np.memmap(os.path.join(concat_plane0(run), "data.bin"), dtype=np.int16, mode="r",
                        shape=(int(FR.frame_stop.max()), ops["Ly"], ops["Lx"]))
        rows_def[f"{run} data.bin"] = [np.asarray(reg[r.frame_start:r.frame_stop], float).mean(0) for r in FR.itertuples()]
    reg_rows = []
    fig, axs = plt.subplots(4, 7, figsize=(22, 12.5), gridspec_kw={"width_ratios": [1] * 6 + [1.2]})
    cr = slice(20, -20)  # ignore edges (alignment fill / registration border)
    for row, (name, means) in enumerate(rows_def.items()):
        C = np.corrcoef(np.array([m[cr, cr].ravel() for m in means]))
        for i in range(6):
            for j in range(6):
                reg_rows.append({"source": name, "trial_i": SHORT[TRIALS[i]], "trial_j": SHORT[TRIALS[j]], "r": round(C[i, j], 4)})
        for k in range(6):
            lo, hi = np.percentile(means[k], [1, 99.5])
            axs[row, k].imshow(np.clip((means[k] - lo) / (hi - lo), 0, 1)[150:350, 150:350], cmap="gray")
            axs[row, k].set_title(f"{name}: {SHORT[TRIALS[k]]}", fontsize=7); axs[row, k].set_xticks([]); axs[row, k].set_yticks([])
        axs[row, 6].imshow(C, vmin=0.3, vmax=1, cmap="viridis")
        axs[row, 6].set_xticks(range(6)); axs[row, 6].set_xticklabels([SHORT[t] for t in TRIALS], rotation=90, fontsize=7)
        axs[row, 6].set_yticks(range(6)); axs[row, 6].set_yticklabels([SHORT[t] for t in TRIALS], fontsize=7)
        for i in range(6):
            for j in range(6):
                axs[row, 6].text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w")
    fig.suptitle("Per-trial mean images (central 200x200 crop) and mean-image Pearson r (20-px border excluded). "
                 "suite2p's own rigid offsets are 0 for every frame in every run (refImg ~flat).")
    fig.tight_layout(); fig.savefig(HERE / "fig_registration.png", dpi=95)
    pd.DataFrame(reg_rows).to_csv(HERE / "registration_meanimg_corr.csv", index=False)

    # ---- p-value ECDFs
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, f in zip(axs, (F_MAG, F_VIS)):
        for run in RUNS:
            p = np.sort(np.concatenate(pvals[(run, f)])); N = len(p)
            ax.plot(p, np.arange(1, N + 1) / N - p, color=COL[run], label=f"{run} (N={N} cell-trials)")
        N = len(np.concatenate(pvals[("separate_nas", f)]))
        x = np.linspace(0, 1, 201); band = 1.96 * np.sqrt(x * (1 - x) / N)
        ax.fill_between(x, -band, band, color="0.85", label=f"binomial 95% band (N={N})")
        ax.axhline(0, c="k", lw=0.5)
        ax.set_xlabel("p"); ax.set_ylabel("ECDF(p) - p")
        ax.set_title(f"{'magnetic' if f == F_MAG else 'visual'} f = {f:.4g} Hz, pooled over 6 trials")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(HERE / "fig_pvalue_ecdf.png", dpi=120)


if __name__ == "__main__":
    main()
