"""Compare a session's ORIGINAL suite2p output (NAS, read-only) with its suite2p v1 re-run (local).

Run with the magneto2 env (numpy/pandas/matplotlib/tifffile) from the repo root:

    C:/Users/dan/anaconda3/envs/magneto2/python.exe docs/suite2p_v1/compare_v1.py \
        --session 2022_10_01-fish2_cytoGCaMP

The v1 output is expected where docs/suite2p_v1/run_suite2p_v1.py puts it:
<local-root>/<run-name>/output/suite2p/plane0/ (+ settings_used.json one level up).

Per session and version (old / v1):
  * ROI counts: all, and passing the production thresholds P(iscell) > 0.5 and npix >= 10
  * P(iscell) (iscell.npy column 1) and npix (stat["npix"]) distributions
Per recording (one tiff; frame ranges from cumulative frames_per_file of the respective run):
  * activity coverage of each ROI on that recording's slice of F, from pipeline/roi_coverage.py
    (share of 60 s windows with >= 3 frames above the trace's floor), using the sample period
    of the session's experiments/*.yml
Halving check: raw tiff vs v1 data.bin vs old data.bin on the first frames (dtype, mean ratio,
most common value and its pixel share), and the floor of F.

Writes (re-running a session replaces its rows):
  docs/suite2p_v1/results/sessions.csv, recordings.csv, halving.csv   (small, tracked)
  docs/suite2p_v1/results/rois_<run-name>.csv                         (per ROI, gitignored)
  docs/suite2p_v1/figs/<run-name>.png, figs/overview_counts.png
Nothing is written to the NAS.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from pipeline.roi_coverage import activity  # noqa: E402

NAS_ROOT = "//datanas/family/data_aggregated/Engert"
LOCAL_ROOT = "C:/Users/dan/suite2p_v1_runs"
ISCELL_MIN, NPIX_MIN, COVERAGE_MIN = 0.5, 10, 0.1     # production: p > 0.5, npix >= 10, cov >= 0.1
N_HALVING_FRAMES = 100
OLD_C, NEW_C = "#2a78d6", "#eb6834"                    # categorical slots 1, 2
RES, FIGS = HERE / "results", HERE / "figs"


def sample_period_for(session):
    """sample_period from the experiments/*.yml whose session_path ends with `session`."""
    vals = set()
    for y in (REPO / "experiments").glob("*.yml"):
        txt = y.read_text()
        sp = [l.split(":", 1)[1].strip().strip('"') for l in txt.splitlines()
              if l.startswith("session_path:")]
        if sp and sp[0].replace("\\", "/").rstrip("/").endswith(session.rstrip("/")):
            for l in txt.splitlines():
                if l.startswith("sample_period:"):
                    vals.add(float(l.split(":", 1)[1].split("#")[0]))
    if len(vals) != 1:
        raise ValueError(f"expected one sample_period for {session} in experiments/, got {vals}")
    return vals.pop()


def load_run(plane0):
    plane0 = Path(plane0)
    ops = np.load(plane0 / "ops.npy", allow_pickle=True).item()
    stat = np.load(plane0 / "stat.npy", allow_pickle=True)
    iscell = np.load(plane0 / "iscell.npy", allow_pickle=True)
    F = np.load(plane0 / "F.npy", allow_pickle=True)
    npix = np.array([s["npix"] for s in stat])
    return dict(ops=ops, F=F, p=iscell[:, 1].astype(float), npix=npix,
                passing=(iscell[:, 1] > ISCELL_MIN) & (npix >= NPIX_MIN))


def q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = np.asarray(x, float)
    return {f"q{int(100 * k)}": (np.quantile(x, k) if len(x) else np.nan) for k in qs}


def mode_share(a):
    v, c = np.unique(a, return_counts=True)
    return v[c.argmax()].item(), c.max() / a.size


def halving_check(tiff_path, new_bin, old_bin, Ly, Lx, nfr=N_HALVING_FRAMES):
    import tifffile
    rows = []
    with tifffile.TiffFile(tiff_path) as tf:
        raw = np.stack([tf.pages[k].asarray() for k in range(nfr)])
    for name, arr in [("raw tiff", raw),
                      ("v1 data.bin", np.memmap(new_bin, np.int16, "r", shape=(nfr, Ly, Lx))),
                      ("old data.bin", np.memmap(old_bin, np.int16, "r", shape=(nfr, Ly, Lx))
                       if old_bin is not None and Path(old_bin).exists() else None)]:
        if arr is None:
            continue
        arr = np.asarray(arr)
        m, s = mode_share(arr)
        rows.append(dict(source=name, dtype=str(arr.dtype), mean=float(arr.mean()),
                         mean_ratio_to_raw=float(arr.mean() / raw.mean()), mode=m,
                         share_at_mode=s, n_distinct=int(np.unique(arr).size),
                         min=int(arr.min()), max=int(arr.max())))
    return rows


def replace_rows(path, df, session):
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[old["session"] != session], df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="session directory relative to --nas-root")
    ap.add_argument("--nas-root", default=NAS_ROOT)
    ap.add_argument("--local-root", default=LOCAL_ROOT)
    ap.add_argument("--run-name", help="default: last component of --session")
    ap.add_argument("--sample-period", type=float, help="default: from experiments/*.yml")
    a = ap.parse_args()

    run = a.run_name or Path(a.session).name
    T = a.sample_period or sample_period_for(a.session)
    old_dir = Path(a.nas_root) / a.session / "suite2p" / "plane0"
    run_dir = Path(a.local_root) / run
    new_dir = run_dir / "output" / "suite2p" / "plane0"
    info = json.loads((run_dir / "output" / "settings_used.json").read_text())
    RES.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)

    print(f"{a.session}: loading old (NAS, read-only) and v1 outputs; sample period {T} s")
    R = {"old": load_run(old_dir), "v1": load_run(new_dir)}
    R["old"]["version"] = str(R["old"]["ops"].get("suite2p_version"))
    R["v1"]["version"] = str(info.get("suite2p_version"))
    names = info["tiffs"]
    fpf = {"old": [int(x) for x in R["old"]["ops"]["frames_per_file"]],
           "v1": [int(x) for x in info["frames_per_file_tiff"]]}
    for k in R:
        if sum(fpf[k]) != R[k]["F"].shape[1]:
            raise ValueError(f"{k}: frames_per_file sum {sum(fpf[k])} != F frames {R[k]['F'].shape[1]}")

    # ---- per session
    srows = []
    for k, r in R.items():
        srows.append(dict(session=a.session, run=k, suite2p_version=r["version"],
                          n_rois=len(r["p"]), n_pass=int(r["passing"].sum()),
                          n_p_gt_05=int((r["p"] > ISCELL_MIN).sum()),
                          n_npix_ge_10=int((r["npix"] >= NPIX_MIN).sum()),
                          p_mean=r["p"].mean(), **{f"p_{kk}": v for kk, v in q(r["p"]).items()},
                          npix_mean=r["npix"].mean(),
                          **{f"npix_{kk}": v for kk, v in q(r["npix"]).items()},
                          seconds_suite2p=info.get("seconds_suite2p") if k == "v1" else
                          (r["ops"].get("timing") or {}).get("total_plane_runtime")))
    sessions = replace_rows(RES / "sessions.csv", pd.DataFrame(srows), a.session)

    # ---- per recording
    rrows, roi_tabs, covs = [], [], {}
    for k, r in R.items():
        edges = np.concatenate([[0], np.cumsum(fpf[k])])
        tab = pd.DataFrame(dict(session=a.session, run=k, roi=np.arange(len(r["p"])),
                                p_iscell=r["p"], npix=r["npix"], passing=r["passing"]))
        for j, n in enumerate(names):
            Fs = r["F"][:, edges[j]:edges[j + 1]].astype(float)
            act = activity(Fs, T)
            cov = act["coverage"]
            covs[(k, n)] = cov
            tab[f"coverage__{n}"] = cov
            for subset, m in [("all", np.ones(len(cov), bool)), ("pass", r["passing"])]:
                c = cov[m]
                rrows.append(dict(session=a.session, recording=n, run=k, rois=subset,
                                  frames=int(edges[j + 1] - edges[j]), n=int(m.sum()),
                                  share_clipped=act["clipped"][m].mean() if m.any() else np.nan,
                                  active_frac_median=np.median(act["active_frac"][m]) if m.any() else np.nan,
                                  coverage_mean=c.mean() if len(c) else np.nan,
                                  **{f"coverage_{kk}": v for kk, v in q(c).items()},
                                  n_coverage_ge_01=int((c >= COVERAGE_MIN).sum()),
                                  share_coverage_ge_01=(c >= COVERAGE_MIN).mean() if len(c) else np.nan))
        roi_tabs.append(tab)
    recordings = replace_rows(RES / "recordings.csv", pd.DataFrame(rrows), a.session)
    pd.concat(roi_tabs).to_csv(RES / f"rois_{run}.csv", index=False)

    # ---- halving
    tiff = run_dir / "tiffs" / names[0]
    if not tiff.exists():                      # local copy deleted: read the NAS original
        tiff = Path(a.nas_root) / a.session / names[0]
    Ly, Lx = int(R["v1"]["ops"]["Ly"]), int(R["v1"]["ops"]["Lx"])
    hrows = halving_check(tiff, new_dir / "data.bin", old_dir / "data.bin", Ly, Lx)
    for k, r in R.items():
        m, s = mode_share(np.round(r["F"][r["passing"]], 3)) if r["passing"].any() else (np.nan, np.nan)
        hrows.append(dict(source=f"{k} F (passing ROIs)", dtype=str(r["F"].dtype),
                          mean=float(r["F"].mean()), mode=m, share_at_mode=s,
                          n_distinct=int(np.unique(r["F"][r["passing"]]).size)))
    for h in hrows:
        h["session"] = a.session
        h["frames_checked"] = N_HALVING_FRAMES
    halving = replace_rows(RES / "halving.csv", pd.DataFrame(hrows), a.session)

    # ---- figure for this session
    plot_session(a.session, run, R, names, covs, T)
    plot_overview(sessions)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(sessions[sessions.session == a.session][
        ["run", "suite2p_version", "n_rois", "n_pass", "n_p_gt_05", "n_npix_ge_10",
         "p_mean", "p_q50", "npix_q10", "npix_q50", "npix_q90", "seconds_suite2p"]].to_string(index=False))
    rr = recordings[(recordings.session == a.session) & (recordings.rois == "pass")]
    print(rr[["recording", "run", "n", "share_clipped", "coverage_q25", "coverage_q50", "coverage_q75",
              "share_coverage_ge_01", "n_coverage_ge_01"]].to_string(index=False))
    print(halving[halving.session == a.session][
        ["source", "dtype", "mean", "mean_ratio_to_raw", "mode", "share_at_mode", "n_distinct"]].to_string(index=False))


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    ax.set_axisbelow(True)


def plot_session(session, run, R, names, covs, T):
    nrec = len(names)
    fig = plt.figure(figsize=(13, 7.5 + 2.2 * ((nrec + 2) // 3)), constrained_layout=True)
    gs = fig.add_gridspec(2 + (nrec + 2) // 3, 3)
    lab = {k: f"{'original' if k == 'old' else 'v1'} (suite2p {r['version']})" for k, r in R.items()}
    col = {"old": OLD_C, "v1": NEW_C}

    ax = fig.add_subplot(gs[0, 0])
    cats = ["all ROIs", "P(iscell) > 0.5", "npix >= 10", "both (production)"]
    w = 0.38
    for i, k in enumerate(R):
        r = R[k]
        vals = [len(r["p"]), (r["p"] > ISCELL_MIN).sum(), (r["npix"] >= NPIX_MIN).sum(), r["passing"].sum()]
        x = np.arange(4) + (i - 0.5) * w
        ax.bar(x, vals, w * 0.95, color=col[k], label=lab[k])
        for xx, v in zip(x, vals):
            ax.text(xx, v, str(int(v)), ha="center", va="bottom", fontsize=7, color="#52514e")
    ax.set_xticks(np.arange(4), cats, fontsize=8, rotation=15)
    ax.set_ylabel("number of ROIs")
    ax.set_title("ROI counts", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)

    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 1, 41)
    for k, r in R.items():
        ax.hist(r["p"], bins, histtype="step", lw=2, color=col[k], label=lab[k])
    ax.axvline(ISCELL_MIN, color="#52514e", ls=":", lw=1)
    ax.set_xlabel("P(iscell)  (iscell.npy column 1)")
    ax.set_ylabel("ROIs per bin")
    ax.set_title("P(iscell), all ROIs", fontsize=10, loc="left")
    _style(ax)

    ax = fig.add_subplot(gs[0, 2])
    allnp = np.concatenate([r["npix"] for r in R.values()])
    bins = np.logspace(0, np.log10(max(allnp.max(), 10) * 1.05), 45)
    for k, r in R.items():
        ax.hist(r["npix"], bins, histtype="step", lw=2, color=col[k], label=lab[k])
    ax.axvline(NPIX_MIN, color="#52514e", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("npix (stat['npix'])")
    ax.set_ylabel("ROIs per bin")
    ax.set_title("ROI size, all ROIs", fontsize=10, loc="left")
    _style(ax)

    ax = fig.add_subplot(gs[1, 0])
    for k, r in R.items():
        ax.scatter(r["npix"], r["p"], s=4, alpha=0.35, color=col[k], label=lab[k], lw=0)
    ax.set_xscale("log")
    ax.axhline(ISCELL_MIN, color="#52514e", ls=":", lw=1)
    ax.axvline(NPIX_MIN, color="#52514e", ls=":", lw=1)
    ax.set_xlabel("npix")
    ax.set_ylabel("P(iscell)")
    ax.set_title("P(iscell) vs npix", fontsize=10, loc="left")
    _style(ax)

    ax = fig.add_subplot(gs[1, 1:])
    pos, ticks = 0, []
    for j, n in enumerate(names):
        for i, k in enumerate(R):
            c = covs[(k, n)][R[k]["passing"]]
            if len(c):
                bp = ax.boxplot(c, positions=[pos + i * 0.4], widths=0.32, patch_artist=True,
                                showfliers=False, medianprops=dict(color="#0b0b0b"))
                bp["boxes"][0].set(facecolor=col[k], alpha=0.75, edgecolor=col[k])
        ticks.append(pos + 0.2)
        pos += 1.2
    ax.axhline(COVERAGE_MIN, color="#52514e", ls=":", lw=1)
    ax.set_xticks(ticks, [n.replace(".tif", "") for n in names], fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("coverage")
    ax.set_title("coverage per recording, ROIs passing P(iscell) > 0.5 and npix >= 10 "
                 "(box: quartiles, whiskers 1.5 IQR)", fontsize=10, loc="left")
    _style(ax)

    for j, n in enumerate(names):
        ax = fig.add_subplot(gs[2 + j // 3, j % 3])
        for k in R:
            c = np.sort(covs[(k, n)][R[k]["passing"]])
            if len(c):
                ax.step(c, np.arange(1, len(c) + 1) / len(c), where="post", lw=2, color=col[k],
                        label=f"{k}: n={len(c)}, {np.mean(c >= COVERAGE_MIN):.0%} >= {COVERAGE_MIN}")
        ax.axvline(COVERAGE_MIN, color="#52514e", ls=":", lw=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("coverage")
        ax.set_ylabel("cumulative share of ROIs")
        ax.set_title(n.replace(".tif", ""), fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        _style(ax)

    fig.suptitle(f"{session}: original suite2p {R['old']['version']} vs suite2p {R['v1']['version']}"
                 f"  (coverage: 60 s windows, sample period {T} s)", fontsize=11)
    fig.savefig(FIGS / f"{run}.png", dpi=130)
    plt.close(fig)


def plot_overview(sessions):
    s = sessions.copy()
    s["label"] = s["session"].map(lambda x: Path(x).name)
    labels = list(dict.fromkeys(s["label"]))
    fig, axes = plt.subplots(1, 2, figsize=(max(7, 1.1 * len(labels) + 4), 4.2), constrained_layout=True)
    w = 0.38
    for ax, col_, title in [(axes[0], "n_rois", "all ROIs"),
                            (axes[1], "n_pass", "P(iscell) > 0.5 and npix >= 10")]:
        for i, (k, c) in enumerate([("old", OLD_C), ("v1", NEW_C)]):
            v = [s[(s.label == l) & (s.run == k)][col_].sum() for l in labels]
            ax.bar(np.arange(len(labels)) + (i - 0.5) * w, v, w * 0.95, color=c,
                   label="original" if k == "old" else "suite2p v1")
        ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("number of ROIs")
        ax.set_title(title, fontsize=10, loc="left")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(FIGS / "overview_counts.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
