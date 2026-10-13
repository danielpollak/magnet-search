"""Re-segmenting the photon-starved zebrafish sessions: ROI size and quality, and what the
analyses find, production vs the two x64 re-segmentations (common.py).

    bash docs/resegmentation/run_all.sh          # drift, tiffs, suite2p runs (local disk)
    NS_DATA_DIR=<checkout>/data python docs/resegmentation/analyze.py   # magneto2, ~15 min
    python docs/resegmentation/analyze.py --figures-only

Population: the 30 experiments whose YAML points at one of the five photon-starved sessions
(common.experiments()).

Per ROI and trace set:
  * ROI properties: npix, P(iscell), suite2p's compactness, and its brightness (the mean of its
    trace above the offset: photons per pixel per frame, weighted by suite2p's lam);
  * per experiment, as in docs/nonstationarity.md: block-shuffle nonstationarity p (RMS,
    reference-window power, CV^2; K = 5/10/20); production fit_Fourier p at the YAML's analysis
    frequency and, where the YAML sets visual_f, at 1/60 Hz.
Fragmentation: how many production ROIs sit inside each re-segmented kept ROI (at least half of
their pixels), and how many re-segmented kept ROIs each production kept ROI overlaps.

Checks, asserted:
  * check_loader: common.load on production's run gives exactly the ROIs and traces the pipeline
    wrote to the NWB files, for all 30 experiments;
  * reconcile: the production trace set's p-values at the analysis frequency match the parquet
    (to 1e-5; not bit-identical, see docs/custom_registration/analyze.py) with identical suspect
    counts.

Outputs: results_rois.csv, results_sessions.csv, results_experiments.csv, results_pooled.csv,
results_fragments.csv, fig_*.png.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "docs" / "nonstationarity"))
import common as cm  # noqa: E402
import nonstationarity as ns  # noqa: E402
from magpyneto2 import engert_helpers as eh  # noqa: E402

ALPHA_NS, ALPHA_F = 0.05, 0.01
DEFAULT_VISUAL_QFRAC = 0.5     # pipeline/analysis_stages/engert.py _DEFAULT_VISUAL_Q_FRAC
C_INK, C_MUTED = "#333333", "#8a8a8a"
_orig_load = ns.sv.load


def batch(name):
    return "0.3 Hz" if ("20220914" in name or "20220915" in name) else "0.1 Hz"


def check_loader():
    """common.load on the segmentation each YAML uses now must give exactly the ROIs and (in raw
    photon units) the traces the pipeline wrote to the NWB files."""
    for names in cm.experiments().values():
        for name in names:
            tag, scale = cm.current_tag(name)
            _, F, roi = _orig_load(name, outline=True, series="full")
            _, F2, roi2 = cm.load(tag, name)
            assert np.array_equal(roi, roi2), f"{name}: ROI sets differ ({len(roi)} vs {len(roi2)})"
            err = float(np.abs(F / scale - F2).max())
            assert err < 1e-3, f"{name}: traces differ by {err}"
    print("  check: common.load on the segmentation each YAML uses reproduces the NWB files' ROIs "
          "and traces in all 30 experiments")


def fourier_p(F, cfg, f, qf):
    NFC, _, _, _, M, _ = eh.fit_Fourier(F, T=cfg.sample_period, f=f, Q_frac=qf)
    return ns.sv.pvals(np.asarray(NFC, float), M)


def job(args):
    name, tag = args
    ns.sv.load = lambda n, **kw: cm.load(tag, n, **kw)
    d, _ = ns.run(name)
    d = d[~d.synthetic].copy()
    cfg, F, roi = cm.load(tag, name)
    assert np.array_equal(roi, d.roi.values)
    d["p_mag"] = fourier_p(F, cfg, cfg.analysis.f, cfg.analysis.Q_frac)
    if cfg.analysis.visual_f > 0:
        qf = cfg.analysis.visual_Q_frac if cfg.analysis.visual_Q_frac > 0 else DEFAULT_VISUAL_QFRAC
        d["p_vis"] = fourier_p(F, cfg, cfg.analysis.visual_f, qf)
    else:
        d["p_vis"] = np.nan
    d["brightness"] = F.mean(1) - cm.OFFSET     # photons / pixel / frame, lam-weighted
    d["trace_set"] = tag
    d["session"] = cm.session_of(name)
    return d


def roi_table(session):
    """Every ROI of the three segmentations of one session: properties, selection, centre in raw
    coordinates."""
    cfg = ns.sv.schema.load_experiment(str(_REPO / "experiments" / f"{cm.experiments()[session][0]}.yml"))
    rows = []
    for tag in cm.TAGS:
        p0 = cm.plane0(tag, session)
        stat = np.load(p0 / "stat.npy", allow_pickle=True)
        p = np.load(p0 / "iscell.npy")[:, 1]
        sel = cm.selection(tag, session, cfg)
        inside = cm.selection(tag, session, cfg, outline_only=True)
        off = cm.offset(tag, session)
        for k, s in enumerate(stat):
            rows.append(dict(session=session, trace_set=tag, roi=k, npix=int(s["npix"]),
                             p_iscell=float(p[k]), compact=float(s.get("compact", np.nan)),
                             selected=bool(sel[k]), inside_outline=bool(inside[k]), med_y=float(np.median(s["ypix"]) - off[0]),
                             med_x=float(np.median(s["xpix"]) - off[1])))
    return pd.DataFrame(rows)


def fragments(session):
    """Overlap of production's ROIs with re-segmented selected ROIs (raw coordinates)."""
    cfg = ns.sv.schema.load_experiment(str(_REPO / "experiments" / f"{cm.experiments()[session][0]}.yml"))
    o = cm.ox.load_ops(str(cm.plane0("production", session)))
    Ly, Lx = int(o["Ly"]), int(o["Lx"])
    prod = np.load(cm.plane0("production", session) / "stat.npy", allow_pickle=True)
    psel = cm.selection("production", session, cfg)
    off = cm.offset("production", session)
    rows = []
    for tag in cm.TAGS[1:]:
        stat = np.load(cm.plane0(tag, session) / "stat.npy", allow_pickle=True)
        sel = np.where(cm.selection(tag, session, cfg))[0]
        lab = np.full((Ly, Lx), -1)
        for k in sel:
            lab[stat[k]["ypix"], stat[k]["xpix"]] = k
        inside = {k: [] for k in sel}
        overlaps_prod = []
        for j, s in enumerate(prod):
            y, x = s["ypix"] - off[0], s["xpix"] - off[1]
            ok = (y >= 0) & (y < Ly) & (x >= 0) & (x < Lx)
            v = lab[y[ok], x[ok]]
            if psel[j]:
                overlaps_prod.append(len(set(v[v >= 0])))
            if ok.sum() and (v >= 0).any():
                k, c = np.unique(v[v >= 0], return_counts=True)
                if c.max() >= 0.5 * len(y):
                    inside[k[np.argmax(c)]].append((j, bool(psel[j])))
        for k, js in inside.items():
            rows.append(dict(session=session, trace_set=tag, kind="reseg_roi", roi=int(k),
                             n_prod=len(js), n_prod_selected=sum(p for _, p in js),
                             prod_rois=" ".join(str(j) for j, _ in js)))
        for j, n in zip(np.where(psel)[0], overlaps_prod):
            rows.append(dict(session=session, trace_set=tag, kind="prod_roi", roi=int(j),
                             n_reseg_selected=n))
    return pd.DataFrame(rows)


def compute(tags=None):
    """Per-recording analyses for `tags` (default all); rows of other trace sets already in
    results_rois.csv are kept, so a new segmentation can be added without redoing the rest."""
    check_loader()
    tags = tags or cm.TAGS
    names = [n for v in cm.experiments().values() for n in v]
    with ProcessPoolExecutor(24) as pool:
        futs = [pool.submit(job, (n, t)) for n in names for t in tags]
        rt = [pool.submit(roi_table, s) for s in cm.SESSIONS]
        fr = [pool.submit(fragments, s) for s in cm.SESSIONS]
        d = pd.concat([f.result() for f in futs], ignore_index=True)
        R = pd.concat([f.result() for f in rt], ignore_index=True)
        Fr = pd.concat([f.result() for f in fr], ignore_index=True)
    d = d.merge(R[["session", "trace_set", "roi", "npix", "p_iscell", "compact"]],
                on=["session", "trace_set", "roi"], how="left")
    keep = ["session", "experiment", "trace_set", "roi", "coil", "visual", "npix", "p_iscell",
            "compact", "brightness", "p_mag", "p_vis"] + [c for c in d.columns if c.startswith("p_block_")]
    d = d[keep]
    if set(tags) != set(cm.TAGS) and (_HERE / "results_rois.csv").exists():
        old = pd.read_csv(_HERE / "results_rois.csv")
        old = old[~old.trace_set.isin(tags)].drop(columns=["npix", "p_iscell", "compact"])
        old = old.merge(R[["session", "trace_set", "roi", "npix", "p_iscell", "compact"]],
                        on=["session", "trace_set", "roi"], how="left")
        d = pd.concat([old[keep], d], ignore_index=True)
    d.to_csv(_HERE / "results_rois.csv", index=False)
    R.to_csv(_HERE / "results_sessions_rois.csv", index=False)
    Fr.to_csv(_HERE / "results_fragments.csv", index=False)
    return d, R, Fr


def reconcile(d):
    pq = pd.read_parquet(ns.PARQUET, columns=["rec", "freq", "id", "p_value"])
    worst = 0.0
    cur = {n: cm.current_tag(n)[0] for n in d.experiment.unique()}
    for name, g in d[d.trace_set == d.experiment.map(cur)].groupby("experiment"):
        f = ns.sv.schema.load_experiment(str(_REPO / "experiments" / f"{name}.yml")).analysis.f
        ref = pq[(pq.rec == name) & np.isclose(pq.freq, f)].set_index("id").p_value
        worst = max(worst, float(np.abs(g.set_index("roi").p_mag - ref.reindex(g.roi)).max()))
        n_h, n_p = int((g.p_mag < ALPHA_F).sum()), int((ref < ALPHA_F).sum())
        assert n_h == n_p, f"{name}: {n_h} suspects in the harness, {n_p} in the parquet"
    print(f"  harness vs parquet (the segmentation each YAML uses), analysis-frequency p-values: "
          f"max |difference| {worst:.2g}; suspect counts identical in all 30 recordings")
    assert worst < 1e-5, "the harness disagrees with production"


def summaries(d, R, Fr):
    pd.set_option("display.width", 220)
    # sessions: ROI counts and sizes
    fld = {}
    for s in cm.SESSIONS:
        o = cm.ox.load_ops(str(cm.plane0("production", s)))
        off = cm.offset("production", s)
        fld[s] = (np.array(o["yrange"]) - off[0], np.array(o["xrange"]) - off[1])
    R = R.copy()
    R["in_prod_field"] = [fld[s][0][0] <= y < fld[s][0][1] and fld[s][1][0] <= x < fld[s][1][1]
                          for s, y, x in zip(R.session, R.med_y, R.med_x)]
    S = R.groupby(["session", "trace_set"], sort=False).agg(
        rois=("roi", "size"), share_outside_outline=("inside_outline", lambda v: 1 - np.mean(v)),
        share_under_10px=("npix", lambda v: np.mean(v < 10)),
        selected=("selected", "sum"))
    sel = R[R.selected]
    S["selected_in_prod_field"] = sel[sel.in_prod_field].groupby(["session", "trace_set"]).size()
    S["npix_median"] = sel.groupby(["session", "trace_set"]).npix.median()
    S["p_iscell_median"] = sel.groupby(["session", "trace_set"]).p_iscell.median()
    S["compact_median"] = sel.groupby(["session", "trace_set"]).compact.median()
    S["brightness_median"] = d.groupby(["session", "trace_set"]).brightness.median()
    S = S.reset_index()
    S.to_csv(_HERE / "results_sessions.csv", index=False)
    print(S.round(3).to_string(index=False))
    # fragmentation
    a = Fr[Fr.kind == "reseg_roi"].groupby(["session", "trace_set"]).agg(
        reseg_rois=("roi", "size"), median_prod_inside=("n_prod", "median"),
        share_with_2plus_prod=("n_prod", lambda v: np.mean(v >= 2)),
        share_with_2plus_prod_selected=("n_prod_selected", lambda v: np.mean(v >= 2)),
        share_with_no_prod_selected=("n_prod_selected", lambda v: np.mean(v == 0)))
    b = Fr[Fr.kind == "prod_roi"].groupby(["session", "trace_set"]).agg(
        prod_selected=("roi", "size"), share_prod_on_no_reseg=("n_reseg_selected", lambda v: np.mean(v == 0)))
    FR = a.join(b).reset_index()
    FR.to_csv(_HERE / "results_fragments_summary.csv", index=False)
    print(FR.round(3).to_string(index=False))
    # per experiment
    rows = []
    for (name, tag), g in d.groupby(["experiment", "trace_set"]):
        rows.append(dict(session=g.session.iloc[0], experiment=name, batch=batch(name),
                         coil="on" if g.coil.iloc[0] else "off", visual=bool(g.visual.iloc[0]),
                         trace_set=tag, rois=len(g),
                         rms_10=(g.p_block_rms_10 < ALPHA_NS).mean(),
                         band_10=(g.p_block_band_10 < ALPHA_NS).mean(),
                         cv2_10=(g.p_block_cv2_10 < ALPHA_NS).mean(),
                         suspects=int((g.p_mag < ALPHA_F).sum()),
                         responders=int((g.p_vis < ALPHA_F).sum()) if g.p_vis.notna().any() else -1))
    E = pd.DataFrame(rows)
    E["suspects_p_binom"] = binom.sf(E.suspects - 1, E.rois, ALPHA_F)
    E.to_csv(_HERE / "results_experiments.csv", index=False)
    # pooled by batch and coil
    P = E.groupby(["batch", "coil", "trace_set"]).agg(
        recordings=("experiment", "size"), rois=("rois", "sum"), suspects=("suspects", "sum"),
        rms_10=("rms_10", "median"), band_10=("band_10", "median"), cv2_10=("cv2_10", "median"))
    P["suspect_share"] = P.suspects / P.rois
    P["suspects_p_binom"] = binom.sf(P.suspects - 1, P.rois, ALPHA_F)
    P["recordings_p_binom_lt_0.05"] = E.assign(x=E.suspects_p_binom < 0.05).groupby(
        ["batch", "coil", "trace_set"]).x.sum()
    V = E[E.responders >= 0].groupby(["batch", "trace_set"]).agg(
        recordings=("experiment", "size"), rois=("rois", "sum"), responders=("responders", "sum"))
    V["share"] = V.responders / V.rois
    P = P.reset_index()
    P.to_csv(_HERE / "results_pooled.csv", index=False)
    V.reset_index().to_csv(_HERE / "results_responders.csv", index=False)
    print(P.round(3).to_string(index=False))
    print(V.round(3).to_string())
    M = matched(d, Fr)
    return S, FR, E, P, V, M


def matched(d, Fr):
    """The same cells in both segmentations: each re-segmented ROI analysed in a recording,
    against the production ROIs inside it that are analysed in the same recording. Compares the
    photons each collects per frame (brightness x npix; for the production side the sum over its
    fragments inside) and whether it responds to the grating. One block of rows per
    re-segmentation."""
    rows = []
    for tag in cm.TAGS[1:]:
        fr = Fr[(Fr.kind == "reseg_roi") & (Fr.trace_set == tag) & Fr.prod_rois.notna()]
        for (sess, name), g in d.groupby(["session", "experiment"]):
            new = g[g.trace_set == tag].set_index("roi")
            old = g[g.trace_set == "production"].set_index("roi")
            for _, r in fr[fr.session == sess].iterrows():
                js = [int(j) for j in str(r.prod_rois).split() if int(j) in old.index]
                if not js or r.roi not in new.index:
                    continue
                o, nw = old.loc[js], new.loc[r.roi]
                rows.append(dict(trace_set=tag, session=sess, experiment=name, roi=r.roi,
                                 n_prod_analysed=len(js), npix_new=nw.npix, npix_prod=o.npix.sum(),
                                 photons_new=nw.brightness * nw.npix,
                                 photons_prod=float((o.brightness * o.npix).sum()),
                                 photons_best_prod=float((o.brightness * o.npix).max()),
                                 p_vis_new=nw.p_vis, p_vis_best_prod=o.p_vis.min()))
    M = pd.DataFrame(rows)
    M.to_csv(_HERE / "results_matched.csv", index=False)
    for tag, m in M.groupby("trace_set", sort=False):
        v = m[m.p_vis_new.notna()]
        print(f"  {tag}, matched cells: {len(m)} (ROI, recording) pairs; median npix {m.npix_new.median():.0f} vs "
              f"{m.npix_prod.median():.0f} (production fragments inside, summed); photons per frame median "
              f"{m.photons_new.median():.2f} vs {m.photons_prod.median():.2f} (best fragment "
              f"{m.photons_best_prod.median():.2f}); grating trials: responds {(v.p_vis_new < ALPHA_F).mean():.3f} "
              f"vs {(v.p_vis_best_prod < ALPHA_F).mean():.3f} (any fragment); median log10 p "
              f"{np.median(np.log10(v.p_vis_new)):.2f} vs {np.median(np.log10(v.p_vis_best_prod)):.2f} (best fragment)")
        new = d[(d.trace_set == tag) & d.p_vis.notna()]
        key = set(zip(v.experiment, v.roi))
        un = new[[(e, r) not in key for e, r in zip(new.experiment, new.roi)]]
        print(f"    ROIs holding no analysed production ROI, grating trials: {len(un)}, "
              f"responding {(un.p_vis < ALPHA_F).mean():.3f}; median photons per frame "
              f"{(un.brightness * un.npix).median():.2f}")
    return M


# ------------------------------------------------------------------ figures
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_INK, labelsize=7)


def fig_rois(R, d, out):
    """ROI size, P(iscell) and brightness, pooled over sessions."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    sel = R[R.selected]
    for tag in cm.TAGS:
        c, lab = cm.COLORS[tag], cm.LABELS[tag]
        for ax, v, xl, log in ((axes[0], R[R.trace_set == tag].npix, "ROI size, npix (all ROIs)", True),
                               (axes[1], sel[sel.trace_set == tag].npix, "ROI size, npix (selected)", True),
                               (axes[2], sel[sel.trace_set == tag].p_iscell, "P(iscell) (selected)", False),
                               (axes[3], d[d.trace_set == tag].brightness.clip(lower=1e-3),
                                "mean photons / pixel / frame (analysed ROIs)", True)):
            v = np.sort(np.asarray(v, float))
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=1.3,
                    label=f"{lab} ({len(v):,})")
            ax.set_xlabel(xl, fontsize=8)
            if log:
                ax.set_xscale("log")
    axes[0].axvline(10, color=C_INK, lw=0.6, ls=":")
    for ax in axes:
        ax.set_ylabel("share at or below", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
        _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_counts(S, out):
    """Selected ROIs per session and their median size."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    x = np.arange(len(cm.SESSIONS))
    for i, tag in enumerate(cm.TAGS):
        s = S[S.trace_set == tag].set_index("session").reindex(cm.SESSIONS)
        for ax, col in zip(axes, ("selected_in_prod_field", "npix_median")):
            ax.bar(x + (i - 1.5) * 0.2, s[col], width=0.19, color=cm.COLORS[tag], label=cm.LABELS[tag])
    for ax, yl in zip(axes, ("selected ROIs inside production's field", "median npix, selected ROIs")):
        ax.set_xticks(x)
        ax.set_xticklabels([cm.short(s) for s in cm.SESSIONS], fontsize=7)
        ax.set_ylabel(yl, fontsize=8)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_effect(E, out):
    """Per recording: nonstationarity shares, suspect share, responder share, three trace sets."""
    E = E.copy()
    E["label"] = E.experiment.str.replace("engert_", "")
    order = sorted(E.label.unique(), key=lambda s: (s.split("_")[0], s.split("_")[1], "no_" in s, s[-1]))
    E["suspect_share"] = E.suspects / E.rois
    E["responder_share"] = np.where(E.responders >= 0, E.responders / E.rois, np.nan)
    cols = [("rms_10", "RMS nonstationary"), ("band_10", "reference-window power nonstationary"),
            ("suspect_share", "share of suspects (p < 0.01) at the analysis frequency"),
            ("responder_share", "share of 1/60 Hz responders (p < 0.01)")]
    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 11), sharex=True)
    x = np.arange(len(order))
    for ax, (c, lab) in zip(axes, cols):
        for i, tag in enumerate(cm.TAGS):
            v = E[E.trace_set == tag].set_index("label").reindex(order)[c]
            ax.scatter(x + (i - 1.5) * 0.18, v, s=14, color=cm.COLORS[tag], label=cm.LABELS[tag], zorder=3)
        if c == "suspect_share":
            ax.axhline(ALPHA_F, color=C_INK, lw=0.8, ls=":")
        ax.set_ylabel(lab, fontsize=8)
        for b in range(6, len(order), 6):
            ax.axvline(b - 0.5, color=C_MUTED, lw=0.5)
        _style(ax)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(order, rotation=60, ha="right", fontsize=6.5)
    axes[0].legend(fontsize=7, frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_maps(R, out):
    """Every ROI of every segmentation over the drift-corrected run's mean image (log scale), in
    raw coordinates: selected ROIs yellow, the others red."""
    fig, axes = plt.subplots(len(cm.SESSIONS), len(cm.TAGS), figsize=(3.2 * len(cm.TAGS), 3.3 * len(cm.SESSIONS)))
    for i, s in enumerate(cm.SESSIONS):
        mean = cm.ox.load_ops(str(cm.plane0("reseg_corrected", s)))["meanImg"]
        for j, t in enumerate(cm.TAGS):
            stat = np.load(cm.plane0(t, s) / "stat.npy", allow_pickle=True)
            off = cm.offset(t, s)
            sel = set(R[(R.session == s) & (R.trace_set == t) & R.selected].roi)
            lab = np.zeros(mean.shape)
            for k, st in enumerate(stat):
                y, x = st["ypix"] - off[0], st["xpix"] - off[1]
                ok = (y >= 0) & (y < mean.shape[0]) & (x >= 0) & (x < mean.shape[1])
                lab[y[ok], x[ok]] = 2 if k in sel else 1
            ax = axes[i, j]
            ax.imshow(np.log(mean - mean.min() + 1), cmap="gray")
            ax.imshow(np.ma.masked_where(lab == 0, lab), cmap="autumn", alpha=0.55, vmin=1, vmax=2,
                      interpolation="nearest")
            ax.set_title(f"{cm.short(s)}, {cm.LABELS[t]}\n{len(stat):,} ROIs, {len(sel):,} selected", fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--tags", nargs="+", help="only (re)compute these trace sets' analyses")
    ap.add_argument("--tables-only", action="store_true",
                    help="recompute the ROI and fragment tables only (not the per-recording "
                         "analyses), then the summaries and figures")
    a = ap.parse_args()
    if a.tables_only:
        with ProcessPoolExecutor(10) as pool:
            pd.concat(pool.map(roi_table, cm.SESSIONS), ignore_index=True).to_csv(
                _HERE / "results_sessions_rois.csv", index=False)
            pd.concat(pool.map(fragments, cm.SESSIONS), ignore_index=True).to_csv(
                _HERE / "results_fragments.csv", index=False)
    if a.figures_only or a.tables_only:
        d = pd.read_csv(_HERE / "results_rois.csv")
        R = pd.read_csv(_HERE / "results_sessions_rois.csv")
        Fr = pd.read_csv(_HERE / "results_fragments.csv")
    else:
        d, R, Fr = compute(a.tags)
        reconcile(d)
    S, FR, E, P, V, M = summaries(d, R, Fr)
    fig_rois(R, d, _HERE / "fig_rois.png")
    fig_counts(S, _HERE / "fig_counts.png")
    fig_effect(E, _HERE / "fig_effect.png")
    fig_maps(R, _HERE / "fig_maps.png")


if __name__ == "__main__":
    main()
