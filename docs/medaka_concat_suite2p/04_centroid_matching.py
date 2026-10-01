"""Task B: match ROIs across the EXISTING separate per-trial suite2p runs by
centroid (Hungarian assignment after rigid inter-trial alignment), and --
if the concatenated run from 02_run_suite2p.py exists -- validate those
matches against the shared (concatenated) segmentation.

ENV:  magneto2
NAS:  required (READ-ONLY: original per-trial suite2p/plane0/{ops,stat,iscell,F}.npy).
      The validation part also needs SCRATCH/concat (02_run_suite2p.py).
TIME: ~2-3 min (15 trial pairs x 2 Hungarian solves of ~4900x4900, plus
      footprint IoUs).

Outputs (next to this script):
  match_cutoffs.csv               per population: distance cutoff from real-vs-null histograms
  match_pairs_summary.csv         per trial pair x population: n matched, distance/IoU quantiles
  match_distance_null.csv         Hungarian match distances, real vs spatially-shifted null
  match_tracks_summary.csv        how many cells are tracked across k of 6 trials
  concat_validation_summary.csv   centroid matching vs concatenated segmentation (precision/recall)
  fig_match_distance_iou.png      distance / IoU distributions vs null, cutoff
  fig_match_tracks.png            tracked-across-k-trials counts
"""
import csv
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from common import (HERE, TRIALS, SHORT, nas_plane0, load_plane0, run_plane0,
                    phase_corr_shift, pix_index, CONCAT_ALIGNED_DIR, ISCELL_THRES, NPIX_THRES)
from magpyneto2.engert_helpers import remove_flatlines

SEARCH_PX = 15.0       # candidate radius; anything farther is never a match
NULL_OFFSET = (37, -41)  # spatial-shift null: displace one trial's centroids by this


def load_trials():
    out = {}
    for t in TRIALS:
        d = load_plane0(nas_plane0(t))
        stat, isc = d["stat"], d["iscell"]
        npix = np.array([s["npix"] for s in stat])
        m = (isc[:, 1] > ISCELL_THRES) & (npix > NPIX_THRES)
        _, _, _, keep = remove_flatlines(np.asarray(d["F"])[m])
        inc = np.zeros(len(stat), bool); inc[np.where(m)[0][keep]] = True
        out[t] = dict(stat=stat, med=np.array([s["med"] for s in stat], float),
                      included=inc, meanImg=d["ops"]["meanImg"],
                      Ly=d["ops"]["Ly"], Lx=d["ops"]["Lx"])
    return out


def hungarian(ca, cb, cutoff):
    """Min-total-distance one-to-one assignment, restricted to pairs closer
    than `cutoff` (costs beyond are clipped so the solver never prefers
    them; any such assignment is then dropped)."""
    if len(ca) == 0 or len(cb) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    d = np.sqrt(((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1)).astype(np.float32)
    big = cutoff * 2 + 1
    r, c = linear_sum_assignment(np.minimum(d, big))
    keep = d[r, c] <= cutoff
    return r[keep], c[keep], d[r, c][keep].astype(float)


def iou(sa, sb, Lx, Ly, dy, dx):
    a = set(pix_index(sa, Lx).tolist())
    b = set(pix_index(sb, Lx, dy=round(dy), dx=round(dx), Ly=Ly).tolist())
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def main():
    D = load_trials()
    Ly, Lx = D[TRIALS[0]]["Ly"], D[TRIALS[0]]["Lx"]
    # pairwise rigid shifts: point (y,x) in trial j -> (y+dy, x+dx) in trial i
    SH = {(i, j): phase_corr_shift(D[i]["meanImg"], D[j]["meanImg"])[:2]
          for i, j in itertools.permutations(TRIALS, 2)}

    # ---- 1. distance / IoU distributions per population, real vs spatial null
    def pop_idx(t, pop):
        return np.arange(len(D[t]["stat"])) if pop == "all" else np.where(D[t]["included"])[0]

    dist_rows, CUT, cut_info = [], {}, []
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for row_ax, pop in zip(axs, ("all", "included")):
        real_d, null_d, real_iou, null_iou = [], [], [], []
        for i, j in itertools.combinations(TRIALS, 2):
            dy, dx = SH[(i, j)]
            si, sj = pop_idx(i, pop), pop_idx(j, pop)
            ci = D[i]["med"][si]; cj = D[j]["med"][sj] + [dy, dx]
            r, c, d = hungarian(ci, cj, SEARCH_PX)
            ious = [iou(D[i]["stat"][si[a]], D[j]["stat"][sj[b]], Lx, Ly, dy, dx) for a, b in zip(r, c)]
            rn, cnn, dn = hungarian(ci, cj + NULL_OFFSET, SEARCH_PX)
            iousn = [iou(D[i]["stat"][si[a]], D[j]["stat"][sj[b]], Lx, Ly, dy + NULL_OFFSET[0], dx + NULL_OFFSET[1])
                     for a, b in zip(rn, cnn)]
            real_d.append(d); null_d.append(dn); real_iou.append(ious); null_iou.append(iousn)
            dist_rows += [(pop, i, j, "real", round(v, 3), round(w, 4)) for v, w in zip(d, ious)]
            dist_rows += [(pop, i, j, "null_shifted", round(v, 3), round(w, 4)) for v, w in zip(dn, iousn)]
        rd, nd = np.concatenate(real_d), np.concatenate(null_d)
        ri, ni = np.concatenate(real_iou), np.concatenate(null_iou)
        # cutoff: the LARGEST distance on a 0.5-px grid at which the cumulative
        # chance-match estimate null(<=d)/real(<=d) is still <= 0.2 (i.e. an
        # estimated FDR of <=20% among accepted matches). Cumulative rather than
        # per-bin because stat["med"] is integer-pixel, so distances are
        # quantized (0, 1, 1.41, 2, ...) and per-bin ratios are noisy.
        bins = np.arange(0, SEARCH_PX + 0.5, 0.5)
        grid = bins[1:]
        cfdr = np.array([(nd <= g).sum() / max((rd <= g).sum(), 1) for g in grid])
        okg = grid[cfdr <= 0.2]
        k = len(okg)
        CUT[pop] = float(okg.max()) if k else np.nan
        ratio = cfdr
        cdisp = CUT[pop] if k > 0 else 2.0
        fdr = (nd <= cdisp).sum() / max((rd <= cdisp).sum(), 1)
        info = {"population": pop, "n_real_matches_le15px": len(rd), "n_null_matches_le15px": len(nd),
                "cutoff_px": CUT[pop], "null_over_real_within_cutoff_or_2px": round(fdr, 4),
                "real_median_d": round(float(np.median(rd)), 3),
                "real_frac_le_cutoff": round(float(np.mean(rd <= cdisp)), 4),
                "real_median_iou_le_cutoff": round(float(np.median(ri[rd <= cdisp])), 4) if np.any(rd <= cdisp) else np.nan,
                "null_median_iou": round(float(np.median(ni)), 4) if len(ni) else np.nan,
                "cum_null_over_real_at_0.5..4px": " ".join(f"{x:.2f}" for x in ratio[:8])}
        cut_info.append(info); print(info)
        row_ax[0].hist(rd, bins, histtype="step", label=f"real (n={len(rd)})", color="k")
        row_ax[0].hist(nd, bins, histtype="step", label=f"spatial-shift null (n={len(nd)})", color="tab:red")
        row_ax[0].axvline(cdisp, ls="--", color="tab:blue",
                          label=f"cutoff {cdisp:.1f} px" + ("" if k > 0 else " (no d reaches FDR<=0.2; 2 px shown)"))
        row_ax[0].set_xlabel(f"Hungarian match distance (px), {pop} ROIs, 15 trial pairs"); row_ax[0].set_ylabel("count")
        row_ax[0].legend(fontsize=8)
        row_ax[1].scatter(rd, ri, s=2, alpha=0.3, color="k", rasterized=True, label="real")
        row_ax[1].scatter(nd, ni, s=2, alpha=0.3, color="tab:red", rasterized=True, label="null")
        row_ax[1].axvline(cdisp, ls="--", color="tab:blue"); row_ax[1].set_xlabel("distance (px)")
        row_ax[1].set_ylabel("footprint IoU"); row_ax[1].legend(fontsize=8, markerscale=5)
        ib = np.linspace(0, 1, 21)
        row_ax[2].hist(ri[rd <= cdisp], ib, histtype="step", color="k", label="real, d<=cutoff")
        row_ax[2].hist(ni, ib, histtype="step", color="tab:red", label="null, d<=15")
        row_ax[2].set_xlabel("footprint IoU"); row_ax[2].legend(fontsize=8)
    with open(HERE / "match_distance_null.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["population", "trial_i", "trial_j", "kind", "distance_px", "iou"]); w.writerows(dist_rows)
    pd.DataFrame(cut_info).to_csv(HERE / "match_cutoffs.csv", index=False)
    fig.suptitle("Centroid matching across separate per-trial suite2p runs (NAS originals); "
                 f"null = one trial's centroids displaced by {NULL_OFFSET} px")
    fig.tight_layout(); fig.savefig(HERE / "fig_match_distance_iou.png", dpi=120)
    if np.isnan(CUT["all"]):
        CUT["all"] = CUT["included"]   # no distance separates real from chance; reuse for the graph, flagged in write-up
    print("cutoffs used:", CUT)

    # ---- 2. pairwise matches at CUTOFF, for both populations
    pair_rows, edges = [], {"all": [], "included": []}
    for i, j in itertools.combinations(TRIALS, 2):
        dy, dx = SH[(i, j)]
        for pop in ("all", "included"):
            si = np.arange(len(D[i]["stat"])) if pop == "all" else np.where(D[i]["included"])[0]
            sj = np.arange(len(D[j]["stat"])) if pop == "all" else np.where(D[j]["included"])[0]
            r, c, d = hungarian(D[i]["med"][si], D[j]["med"][sj] + [dy, dx], CUT[pop])
            a, b = si[r], sj[c]
            ious = np.array([iou(D[i]["stat"][x], D[j]["stat"][y], Lx, Ly, dy, dx) for x, y in zip(a, b)])
            edges[pop] += [((i, x), (j, y)) for x, y in zip(a, b)]
            row = {"trial_i": SHORT[i], "trial_j": SHORT[j], "population": pop,
                   "n_i": len(si), "n_j": len(sj), "n_matched": len(a),
                   "frac_of_smaller": round(len(a) / max(min(len(si), len(sj)), 1), 4),
                   "dist_median": round(float(np.median(d)), 3) if len(d) else np.nan,
                   "dist_p90": round(float(np.percentile(d, 90)), 3) if len(d) else np.nan,
                   "iou_median": round(float(np.median(ious)), 3) if len(ious) else np.nan,
                   "iou_p10": round(float(np.percentile(ious, 10)), 3) if len(ious) else np.nan,
                   "shift_dy": round(dy, 2), "shift_dx": round(dx, 2)}
            if pop == "included":
                # does an included ROI's ALL-ROI partner exist / pass inclusion in j?
                ra, ca_, _ = hungarian(D[i]["med"], D[j]["med"] + [dy, dx], CUT["included"])
                part = dict(zip(ra, ca_))
                has = [x for x in si if x in part]
                row["inc_i_with_any_partner"] = len(has)
                row["inc_i_partner_included"] = int(sum(D[j]["included"][part[x]] for x in has))
            pair_rows.append(row)
    keys = list(dict.fromkeys(k for r in pair_rows for k in r))
    with open(HERE / "match_pairs_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(pair_rows)

    # ---- 3. multi-trial tracks: connected components of the pairwise-match graph
    track_rows = []
    track_counts = {}
    for pop in ("all", "included"):
        parent = {}
        def find(u):
            while parent.setdefault(u, u) != u:
                parent[u] = parent[parent[u]]; u = parent[u]
            return u
        for u, v in edges[pop]:
            parent[find(u)] = find(v)
        nodes = [(t, k) for t in TRIALS for k in (range(len(D[t]["stat"])) if pop == "all"
                                                 else np.where(D[t]["included"])[0])]
        comps = {}
        for n in nodes:
            comps.setdefault(find(n), []).append(n)
        eset = set(edges[pop]) | {(v, u) for u, v in edges[pop]}
        counts = {k: [0, 0] for k in range(1, 7)}  # k -> [clean, clean & fully-connected]
        n_conflict = 0
        for comp in comps.values():
            trials_in = [n[0] for n in comp]
            if len(set(trials_in)) != len(trials_in):
                n_conflict += 1; continue
            k = len(comp)
            counts[k][0] += 1
            if all((u, v) in eset for u, v in itertools.combinations(comp, 2)):
                counts[k][1] += 1
        track_counts[pop] = counts
        for k in range(1, 7):
            track_rows.append({"population": pop, "n_trials_spanned": k,
                               "n_tracks_one_roi_per_trial": counts[k][0],
                               "n_tracks_fully_pairwise_consistent": counts[k][1]})
        track_rows.append({"population": pop, "n_trials_spanned": "conflict(>1 ROI/trial)",
                           "n_tracks_one_roi_per_trial": n_conflict, "n_tracks_fully_pairwise_consistent": ""})
        print(pop, {k: v for k, v in counts.items()}, "conflicts", n_conflict)
    # within-block subsets (3 magneto trials / 3 no_magneto trials)
    for pop in ("all", "included"):
        for name, sub in (("magneto_only", TRIALS[:3]), ("no_magneto_only", TRIALS[3:])):
            es = [(u, v) for u, v in edges[pop] if u[0] in sub and v[0] in sub]
            eset = set(es) | {(v, u) for u, v in es}
            first = [(sub[0], k) for k in (range(len(D[sub[0]]["stat"])) if pop == "all"
                                           else np.where(D[sub[0]]["included"])[0])]
            nbr = {}
            for u, v in es:
                nbr.setdefault(u, []).append(v); nbr.setdefault(v, []).append(u)
            n3 = 0
            for u in first:
                vs = {w[0]: w for w in nbr.get(u, [])}
                if sub[1] in vs and sub[2] in vs and (vs[sub[1]], vs[sub[2]]) in eset:
                    n3 += 1
            track_rows.append({"population": pop, "n_trials_spanned": f"3-of-3 {name} (triangle)",
                               "n_tracks_one_roi_per_trial": n3, "n_tracks_fully_pairwise_consistent": n3})
            print(pop, name, "triangles", n3)
    with open(HERE / "match_tracks_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(track_rows[0])); w.writeheader(); w.writerows(track_rows)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, pop in zip(axs, ("all", "included")):
        ks = np.arange(1, 7)
        ax.bar(ks - 0.2, [track_counts[pop][k][0] for k in ks], 0.4, label="one ROI per trial")
        ax.bar(ks + 0.2, [track_counts[pop][k][1] for k in ks], 0.4, label="+ all pairwise edges")
        ax.set_xlabel("# trials the track spans"); ax.set_ylabel("# tracks"); ax.set_title(f"{pop} ROIs, cutoff {CUT[pop]:.1f} px")
        ax.set_yscale("log"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / "fig_match_tracks.png", dpi=120)

    # ---- 4. validation vs concatenated segmentation
    allrows = []
    for name, cdir in (("concat_aligned", CONCAT_ALIGNED_DIR / "suite2p" / "plane0"), ("concat", run_plane0("concat"))):
        if not os.path.exists(os.path.join(cdir, "stat.npy")):
            print(f"no {name} run found; skipping its validation"); continue
        for r in validate(D, edges, cdir, Ly, Lx):
            allrows.append({"concat_run": name, **r})
    if allrows:
        keys = list(dict.fromkeys(k for r in allrows for k in r))
        with open(HERE / "concat_validation_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(allrows)


def validate(D, edges, cdir, Ly, Lx):
    C = load_plane0(cdir, load_F=False)
    cstat = C["stat"]
    fr = pd.read_csv(HERE / "concat_frame_ranges.csv")
    reg = np.memmap(os.path.join(cdir, "data.bin"), dtype=np.int16, mode="r",
                    shape=(int(fr.frame_stop.max()), C["ops"]["Ly"], C["ops"]["Lx"]))
    # pixel -> concat ROI owner (concat ROIs don't overlap: allow_overlap=False
    # except where suite2p marks overlap; take max-lam owner)
    owner = -np.ones(Ly * Lx, int); best = np.zeros(Ly * Lx)
    for k, s in enumerate(cstat):
        idx = s["ypix"] * Lx + s["xpix"]; lam = s["lam"]
        upd = lam > best[idx]; owner[idx[upd]] = k; best[idx[upd]] = lam[upd]
    cnpix = np.array([s["npix"] for s in cstat])
    rows, maps = [], {}
    for t, r in zip(TRIALS, fr.itertuples()):
        mimg = np.asarray(reg[r.frame_start:r.frame_stop:5], float).mean(0)
        dy, dx, _ = phase_corr_shift(mimg, D[t]["meanImg"])  # separate(t) -> concat frame
        cmap = -np.ones(len(D[t]["stat"]), int); ciou = np.zeros(len(D[t]["stat"]))
        for a, s in enumerate(D[t]["stat"]):
            idx = pix_index(s, Lx, dy=round(dy), dx=round(dx), Ly=Ly)
            o = owner[idx]; o = o[o >= 0]
            if len(o) == 0:
                continue
            k = np.bincount(o).argmax()
            inter = (o == k).sum()
            ciou[a] = inter / (len(idx) + cnpix[k] - inter)
            cmap[a] = k
        maps[t] = (cmap, ciou, dy, dx)
        print(f"{SHORT[t]}: separate->concat shift ({dy:.2f},{dx:.2f}); median best IoU all {np.median(ciou):.3f}, "
              f"included {np.median(ciou[D[t]['included']]):.3f}")
    for IOU_T in (0.3, 0.5):
        for pop in ("all", "included"):
            tp = fp = 0; npairs_concat = 0; n_unmapped = 0
            pred = set()
            for (i, a), (j, b) in edges[pop]:
                pred.add(((i, a), (j, b)))
                ka, ia = maps[i][0][a], maps[i][1][a]; kb, ib = maps[j][0][b], maps[j][1][b]
                if ia < IOU_T or ib < IOU_T:
                    n_unmapped += 1; continue
                if ka == kb: tp += 1
                else: fp += 1
            # concat-implied pairs among ROIs of this population
            fn = 0
            for i, j in itertools.combinations(TRIALS, 2):
                si = np.arange(len(D[i]["stat"])) if pop == "all" else np.where(D[i]["included"])[0]
                sj = np.arange(len(D[j]["stat"])) if pop == "all" else np.where(D[j]["included"])[0]
                gi = {}; gj = {}
                for a in si:
                    if maps[i][1][a] >= IOU_T: gi.setdefault(maps[i][0][a], []).append(a)
                for b in sj:
                    if maps[j][1][b] >= IOU_T: gj.setdefault(maps[j][0][b], []).append(b)
                for k in set(gi) & set(gj):
                    for a in gi[k]:
                        for b in gj[k]:
                            npairs_concat += 1
                            if ((i, a), (j, b)) not in pred:
                                fn += 1
            rows.append({"iou_threshold_to_concat": IOU_T, "population": pop,
                         "centroid_pairs": len(edges[pop]), "centroid_pairs_both_mapped": tp + fp,
                         "agree_same_concat_roi": tp, "disagree_diff_concat_roi": fp,
                         "precision": round(tp / max(tp + fp, 1), 4),
                         "concat_implied_pairs": npairs_concat, "missed_by_centroid": fn,
                         "recall": round(1 - fn / max(npairs_concat, 1), 4)})
    for t in TRIALS:
        cmap, ciou, dy, dx = maps[t]
        inc = D[t]["included"]
        rows.append({"iou_threshold_to_concat": "per-trial", "population": SHORT[t],
                     "centroid_pairs": f"shift=({dy:.2f},{dx:.2f})",
                     "centroid_pairs_both_mapped": f"frac all ROIs IoU>=0.3: {np.mean(ciou >= 0.3):.3f}",
                     "agree_same_concat_roi": f"frac included IoU>=0.3: {np.mean(ciou[inc] >= 0.3):.3f}",
                     "disagree_diff_concat_roi": f"median IoU included: {np.median(ciou[inc]):.3f}"})
    for r in rows:
        print(r)
    return rows


if __name__ == "__main__":
    main()
