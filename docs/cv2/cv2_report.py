"""CV^2 of the zebrafish imaging ROIs: what it looks like in a raw trace, how it follows
coverage, how it relates to the p-value bump, and where the floor that lowers it comes from.

    python docs/cv2/cv2_marginals.py           # per-unit noise-bin CV^2, ephys + imaging (Figure 3)
    python docs/cv2/cv2_report.py              # ~5 min
    python docs/cv2/cv2_report.py --figures-only

Write-up: docs/cv2.md.

CV^2 here is the whole-spectrum version: variance / mean^2 of a ROI's periodogram ordinates
over every bin from 0.05 Hz up, away from the stimulus window, each divided by the mean of its
51 neighbours (guard_band.ordinate_shape). 1 for Gaussian noise.

Zebrafish 0.4 Hz / 0.3 Hz / 0.1 Hz: every ROI's production p-value at every test frequency
(guard_band.test_bins: every 3rd bin from 0.05 Hz, away from the stimulus) under production's
null F(2, 4M). Each ROI's ECDF over its test frequencies is stored, so any grouping of ROIs is
the mean of its ROIs' ECDFs (each ROI weighted equally). Population: production before the
coverage threshold (P(iscell) > 0.5, npix >= 10, inside the outline, flatline removal). For the
0.3 Hz and 0.1 Hz recordings the same is repeated on the full-resolution traces that processing
stores next to suite2p's (pipeline/ophys_extraction.py; same ROIs, same frames).

Outputs (next to this script): results_cv2_rois.csv, results_cv2_ecdf.npz (gitignored),
fig_cv2_*.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
from scipy import stats

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE.parent / "guard_band_null"))
import guard_band as gb  # noqa: E402
from guard_band import sv, P_GRID, C_INK, C_MUTED, _style  # noqa: E402
from pipeline import nwb_io, schema  # noqa: E402
from pipeline.roi_coverage import activity  # noqa: E402

SETS = gb.SETS
CLIPPED = ["zebrafish 0.3 Hz", "zebrafish 0.1 Hz"]
SHORT = {"zebrafish 0.4 Hz (2022 Q1)": "0.4 Hz (2022 Q1)", "zebrafish 0.3 Hz": "0.3 Hz",
         "zebrafish 0.1 Hz": "0.1 Hz"}
CV2_BINS = [0, 0.5, 0.7, 0.85, 1.15, np.inf]
CV2_LABELS = ["< 0.5", "0.5-0.7", "0.7-0.85", "0.85-1.15", "> 1.15"]
CV2_COLOURS = ["#6b2a0d", "#eb6834", "#f2a36b", "#8a8a8a", "#2a78d6"]
C_S2P, C_FULL = "#eb6834", "#2a78d6"
EXAMPLE_REC = "engert_20221001_fish2_magneto_0"
EXAMPLE_COVERAGE = [0.06, 0.17, 0.33, 0.61, 1.0]
OUT_ROIS = _HERE / "results_cv2_rois.csv"
OUT_ECDF = _HERE / "results_cv2_ecdf.npz"


# ------------------------------------------------------------------ compute
def ecdf_rows(P):
    """Per row of P (ROI x test frequency), the ECDF of its p-values at P_GRID."""
    P = np.sort(P, axis=1)
    return np.stack([np.searchsorted(row, P_GRID, side="right") / P.shape[1] for row in P])


def recordings():
    """(name, batch) of every zebrafish recording in the three sets."""
    recs, _ = sv.pool()
    return [(name, batch) for name, batch, _ in recs if batch in SETS]


def spectrum(F, cfg):
    """What every analysis here needs from a set of traces: dict with the FFT Y, analysis bin
    f0, noise bins M, the whole-spectrum normalised-ordinate range (lo, excl), CV^2 and k = 1/CV^2,
    NFC at the test frequencies X, and the rows with finite NFC everywhere."""
    T, N = cfg.sample_period, F.shape[1]
    f0, M = sv.window_bins(N, T, cfg.analysis.f, cfg.analysis.Q_frac)
    res = 1.0 / (N * T)
    Y = np.fft.rfft(F - F.mean(axis=1, keepdims=True), axis=1)
    lo = max(int(np.ceil(gb.F_MIN / res)) - gb.MED_WIN, 1)
    excl = np.arange(f0 - gb.G_MAX - M, f0 + gb.G_MAX + M + 1)
    k, cv2 = gb.ordinate_shape(Y, lo, Y.shape[1] - 1, excl)
    X = np.stack([gb.nfc(Y, b, M, 0) for b in gb.test_bins(Y.shape[1], f0, M, res)], 1)
    return dict(Y=Y, f0=f0, M=M, lo=lo, excl=excl, k=k, cv2=cv2, X=X,
                ok=np.isfinite(X).all(1))


def floor_share(F):
    """Share of frames at each trace's most common value."""
    return np.array([np.unique(r, return_counts=True)[1].max() / len(r) for r in F])


def full_res(name, roi, F):
    """(suite2p, full-resolution) traces of these ROIs over the same frames as F, both from the
    experiment's NWB file (processing stores both since 2026-10-05), or None for a file
    processed before that."""
    io, nwbfile = nwb_io.read_nwbfile(schema.load_experiment(
        str(_REPO / "experiments" / f"{name}.yml")).nwb_path())
    try:
        Fs, _ = nwb_io.read_roi_data(nwbfile, "suite2p")
        Ff, _ = nwb_io.read_roi_data(nwbfile, "full")
    except KeyError:
        return None
    finally:
        io.close()
    N = F.shape[1]
    Fs, Ff = Fs[roi, :N].astype(float), Ff[roi, :N].astype(float)
    assert np.abs(Fs - F).max() < 1e-3, f"{name}: stored suite2p traces do not match"
    return Fs, Ff


def compute():
    """Per ROI: production CV^2, coverage, floor share and dev@0.5; for the floor-clipped sets
    the same on this tiff's own frames from suite2p (_s2p) and at full resolution (_full).
    ECDFs: E (production), Es and Ef (own frames, suite2p / full resolution; NaN elsewhere)."""
    rows, E, Es, Ef = [], [], [], []
    for name, batch in recordings():
        cfg, F, roi = sv.load(name, outline=True)
        s = spectrum(F, cfg)
        ok = s["ok"]
        r = pd.DataFrame(dict(experiment=name, batch=batch, roi=roi, M=s["M"], cv2=s["cv2"],
                              coverage=activity(F, cfg.sample_period)["coverage"],
                              floor_share=floor_share(F)))
        Pp = gb.p_from_nfc(s["X"], s["M"])
        r["dev"] = np.mean(Pp <= 0.5, 1) - 0.5
        own = full_res(name, roi, F) if batch in CLIPPED else None
        if own is not None:
            P_own = {}
            for tag, Fx in zip(["s2p", "full"], own):
                sx = spectrum(Fx, cfg)
                ok = ok & sx["ok"]
                P_own[tag] = gb.p_from_nfc(sx["X"], sx["M"])
                r[f"cv2_{tag}"], r[f"floor_share_{tag}"] = sx["cv2"], floor_share(Fx)
                r[f"dev_{tag}"] = np.mean(P_own[tag] <= 0.5, 1) - 0.5
            Es.append(ecdf_rows(P_own["s2p"][ok]))
            Ef.append(ecdf_rows(P_own["full"][ok]))
        else:
            Es.append(np.full((ok.sum(), len(P_GRID)), np.nan))
            Ef.append(np.full((ok.sum(), len(P_GRID)), np.nan))
        rows.append(r[ok])
        E.append(ecdf_rows(Pp[ok]))
        print(f"  {name}: {ok.sum()} ROIs, M={s['M']}, full-res {'yes' if own is not None else 'no'}",
              flush=True)
    d = pd.concat(rows, ignore_index=True)
    E, Es, Ef = np.concatenate(E), np.concatenate(Es), np.concatenate(Ef)
    d.to_csv(OUT_ROIS, index=False)
    np.savez_compressed(OUT_ECDF, prod=E, s2p=Es, full=Ef)
    return d, E, Es, Ef


def cv2_bin(c):
    return pd.cut(c, CV2_BINS, labels=CV2_LABELS, right=False)


# ------------------------------------------------------------------ figures
def examples(d):
    """ROIs of EXAMPLE_REC spanning coverage: per target, among ROIs within one 60 s window of
    it, the one with the median CV^2."""
    e = d[d["experiment"] == EXAMPLE_REC]
    out = []
    for t in EXAMPLE_COVERAGE:
        c = e[(e["coverage"] - t).abs() < 0.03].sort_values("cv2")
        out.append(int(c.iloc[len(c) // 2]["roi"]))
    return out


def load_examples(d):
    """(cfg, rows of d, production traces, full-resolution traces) of the examples."""
    cfg, F, roi = sv.load(EXAMPLE_REC, outline=True)
    ids = examples(d)
    pos = [int(np.where(roi == i)[0][0]) for i in ids]
    own = full_res(EXAMPLE_REC, roi, F)
    rows = d[d["experiment"] == EXAMPLE_REC].set_index("roi").loc[ids]
    return cfg, rows, F[pos], (own[1][pos] if own is not None else None)


def fig_traces(d, out):
    """Figure 1: example traces from low to high coverage, and the distribution of their
    normalised power across frequencies."""
    cfg, rows, F, _ = load_examples(d)
    s = spectrum(F, cfg)
    I = gb.normalised_ordinates(s["Y"], s["lo"], s["Y"].shape[1] - 1, s["excl"])
    t = np.arange(F.shape[1]) * cfg.sample_period
    fig, axes = plt.subplots(len(F), 2, figsize=(13, 1.9 * len(F)),
                             gridspec_kw=dict(width_ratios=[3.2, 1]))
    x = np.linspace(0, 6, 200)
    for i, (ax, axh) in enumerate(axes):
        r = rows.iloc[i]
        ax.plot(t, F[i], color=C_INK, lw=0.6)
        ax.set_xlim(0, t[-1])
        ax.set_ylabel("F (suite2p)", fontsize=7)
        ax.set_title(f"ROI {rows.index[i]}: coverage {r['coverage']:.2f}, CV$^2$ {r['cv2']:.2f}, "
                     f"{r['floor_share']:.0%} of frames at the floor", fontsize=8, loc="left")
        _style(ax)
        axh.hist(I[i], bins=np.linspace(0, 6, 31), density=True, color=C_S2P, alpha=0.85)
        axh.plot(x, np.exp(-x), color=C_INK, lw=1, ls="--")
        axh.set_yscale("log")
        axh.set_ylim(2e-3, 3)
        axh.set_xlim(0, 6)
        axh.set_ylabel("density", fontsize=7)
        _style(axh)
    axes[-1, 0].set_xlabel("time (s)", fontsize=8)
    axes[-1, 1].set_xlabel("power / local mean power", fontsize=8)
    axes[0, 1].set_title("power across frequencies\n(dashed: exponential, CV$^2$ = 1)", fontsize=8)
    fig.suptitle(f"{EXAMPLE_REC}: five ROIs from low to high coverage", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_coverage(d, out):
    """Figure 2: CV^2 against coverage, every ROI of the floor-clipped recordings."""
    rng = np.random.default_rng(0)
    ex = examples(d)
    fig, axes = plt.subplots(1, len(CLIPPED), figsize=(11, 4.4), sharey=True)
    for ax, s in zip(axes, CLIPPED):
        e = d[d["batch"] == s]
        ax.scatter(e["coverage"] + rng.uniform(-0.012, 0.012, len(e)), e["cv2"], s=3,
                   color=C_MUTED, alpha=0.35, lw=0, rasterized=True)
        b = pd.cut(e["coverage"], np.linspace(0, 1.0001, 11))
        med = e.groupby(b, observed=True)["cv2"].median()
        ax.plot([iv.mid for iv in med.index], med.values, color=C_INK, lw=2, marker="o", ms=4,
                label="median per 0.1 of coverage")
        if s == "zebrafish 0.1 Hz":
            x = e[(e["experiment"] == EXAMPLE_REC) & e["roi"].isin(ex)]
            ax.scatter(x["coverage"], x["cv2"], s=60, facecolor="none", edgecolor=C_S2P, lw=1.6,
                       label="Figure 1's examples", zorder=3)
        ax.axhline(1, color=C_INK, lw=0.6, ls=":")
        ax.axvline(0.1, color=C_MUTED, lw=0.8, ls="--")
        rho = stats.spearmanr(e["coverage"], e["cv2"])[0]
        ax.set_title(f"zebrafish {SHORT[s]}: {len(e)} ROIs, Spearman {rho:.2f}", fontsize=9)
        ax.set_xlabel("coverage (share of 60 s windows with >= 3 frames above the floor)",
                      fontsize=8)
        ax.set_ylim(0, 2)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        _style(ax)
    axes[0].set_ylabel("whole-spectrum CV$^2$", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_binned(d, rows, out):
    """ECDF deviation where nothing was presented, ROIs binned by CV^2: one row of panels per
    (ECDF array, title) in `rows`. Figure 4 here; dispersion_null.py reuses it."""
    fig, axes = plt.subplots(len(rows), len(SETS), figsize=(4.6 * len(SETS), 3.8 * len(rows)),
                             sharey=True, squeeze=False)
    b = cv2_bin(d["cv2"])
    for i, (E, null) in enumerate(rows):
        for ax, s in zip(axes[i], SETS):
            for lab, c in zip(CV2_LABELS, CV2_COLOURS):
                sel = ((d["batch"] == s) & (b == lab)).values
                if sel.sum() < 20:
                    continue
                y = E[sel].mean(0) - P_GRID
                ax.plot(P_GRID, y, color=c, lw=2, label=f"CV$^2$ {lab}: {sel.sum()} ROIs, "
                        f"dev@0.5 {np.interp(0.5, P_GRID, y):+.3f}")
            ax.axhline(0, color=C_INK, lw=0.6)
            ax.set_title(f"zebrafish {SHORT[s]}, {null}", fontsize=9)
            ax.set_xlabel("p", fontsize=8)
            ax.legend(fontsize=6.5, frameon=False, loc="best")
            _style(ax)
        axes[i, 0].set_ylabel("ECDF(p) - p over test frequencies", fontsize=8)
    fig.suptitle("Where nothing was presented, ROIs binned by whole-spectrum CV$^2$ "
                 "(flat at 0 = calibrated)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.04 / len(rows)))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def pixel_values(n=200):
    """Share of EXAMPLE_REC's pixels at each value, in the first n frames: raw tiff (offset
    100), after suite2p's // 2, and in suite2p's registered movie (data.bin), all expressed in
    raw levels above the offset."""
    cfg = sv.load(EXAMPLE_REC, outline=True)[0]
    plane = f"{cfg.session_path}/suite2p/plane0/"
    ops = np.load(plane + "ops.npy", allow_pickle=True).item()
    raw = tifffile.imread(f"{cfg.session_path}/{cfg.tiff_name}", key=range(n)).astype(int)
    files = [Path(f.replace("\\", "/")).name for f in ops["filelist"]]
    off = int(np.sum(ops["frames_per_file"][:files.index(cfg.tiff_name)]))
    reg = np.memmap(plane + "data.bin", dtype=np.int16, mode="r").reshape(
        -1, ops["Ly"], ops["Lx"])[off:off + n].astype(int)
    out = {}
    for lab, v in [("raw tiff", raw - 100), ("raw // 2 (x 2)", 2 * (raw // 2) - 100),
                   ("suite2p data.bin (x 2)", 2 * reg - 100)]:
        u, c = np.unique(v, return_counts=True)
        out[lab] = (u, c / c.sum())
    return out


def fig_floor(d, out):
    """Figure 5: pixel values at each suite2p step. Figure 6: suite2p and full-resolution
    traces of Figure 1's ROIs, on top of each other."""
    pv = pixel_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.27
    for j, ((lab, (u, c)), col) in enumerate(zip(pv.items(), [C_INK, C_MUTED, C_S2P])):
        keep = (u >= -1) & (u <= 8)
        ax.bar(u[keep] + (j - 1) * w, c[keep], width=w, color=col, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("pixel value above the offset (raw tiff levels)", fontsize=8)
    ax.set_ylabel("share of pixels", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out[0], dpi=150, facecolor="white")
    plt.close(fig)

    cfg, rows, F, Ff = load_examples(d)
    t = np.arange(F.shape[1]) * cfg.sample_period
    fig, axes = plt.subplots(len(F), 2, figsize=(13, 1.9 * len(F)),
                             gridspec_kw=dict(width_ratios=[3, 1.2]))
    z0, z1 = 300, 420
    for i, (ax, axz) in enumerate(axes):
        r = rows.iloc[i]
        for a in (ax, axz):
            a.plot(t, Ff[i] - 100, color=C_FULL, lw=0.6, label="full resolution (raw tiff)")
            a.plot(t, 2 * F[i] - 100, color=C_S2P, lw=0.6, label="suite2p F x 2")
            _style(a)
        ax.set_xlim(0, t[-1])
        axz.set_xlim(z0, z1)
        ax.axvspan(z0, z1, color=C_MUTED, alpha=0.15, lw=0)
        ax.set_ylabel("above offset", fontsize=7)
        ax.set_title(f"ROI {rows.index[i]}: CV$^2$ suite2p {r['cv2_s2p']:.2f}, full resolution "
                     f"{r['cv2_full']:.2f}; frames at the most common value "
                     f"{r['floor_share_s2p']:.0%} vs {r['floor_share_full']:.0%}",
                     fontsize=8, loc="left")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper right")
    axes[0, 1].set_title(f"zoom, {z0}-{z1} s", fontsize=8)
    axes[-1, 0].set_xlabel("time (s)", fontsize=8)
    axes[-1, 1].set_xlabel("time (s)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out[1], dpi=150, facecolor="white")
    plt.close(fig)


def fig_full(d, Es, Ef, out):
    """Figure 7: CV^2 and the p-value distribution where nothing was presented, suite2p traces
    against full-resolution traces of the same ROIs."""
    fig, axes = plt.subplots(2, len(CLIPPED), figsize=(10, 7.6))
    for j, s in enumerate(CLIPPED):
        ins = (d["batch"] == s).values & d["cv2_full"].notna().values
        e = d[ins]
        ax = axes[0, j]
        ax.scatter(e["cv2_s2p"], e["cv2_full"], s=3, color=C_MUTED, alpha=0.4, lw=0, rasterized=True)
        ax.plot([0, 2], [0, 2], color=C_INK, lw=0.6, ls=":")
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xlabel("CV$^2$, suite2p trace", fontsize=8)
        ax.set_ylabel("CV$^2$, full-resolution trace", fontsize=8)
        ax.set_title(f"zebrafish {SHORT[s]}: median {e['cv2_s2p'].median():.2f} -> "
                     f"{e['cv2_full'].median():.2f}", fontsize=9)
        _style(ax)
        ax = axes[1, j]
        for A, c, lab in [(Es, C_S2P, "suite2p"), (Ef, C_FULL, "full resolution")]:
            y = A[ins].mean(0) - P_GRID
            ax.plot(P_GRID, y, color=c, lw=2,
                    label=f"{lab}: dev@0.5 {np.interp(0.5, P_GRID, y):+.3f}")
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_xlabel("p", fontsize=8)
        ax.set_ylabel("ECDF(p) - p over test frequencies", fontsize=8)
        ax.set_title(f"zebrafish {SHORT[s]}: {ins.sum()} ROIs", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="lower center")
        _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def summary(d, Es, Ef):
    b = cv2_bin(d["cv2"])
    t = d.assign(bin=b).groupby(["batch", "bin"], observed=True).agg(
        n=("cv2", "size"), dev=("dev", "mean"), dev_full=("dev_full", "mean"))
    print(t.round(3).to_string())
    below = lambda c: np.mean(c < 0.5)
    g = d.groupby("batch").agg(n=("cv2", "size"), cv2=("cv2", "median"),
                               cv2_s2p=("cv2_s2p", "median"), cv2_full=("cv2_full", "median"),
                               floor=("floor_share", "median"),
                               floor_full=("floor_share_full", "median"),
                               below05=("cv2", below), below05_full=("cv2_full", below))
    print(g.round(3).to_string())
    for s in CLIPPED:
        e = d[d["batch"] == s]
        print(s, "Spearman coverage vs CV2", round(stats.spearmanr(e["coverage"], e["cv2"])[0], 3))
        ins = (d["batch"] == s).values & d["cv2_full"].notna().values
        print(s, "dev@0.5 suite2p / full (own frames)",
              round(np.interp(0.5, P_GRID, Es[ins].mean(0) - P_GRID), 4),
              round(np.interp(0.5, P_GRID, Ef[ins].mean(0) - P_GRID), 4))
    fb = d.assign(bin=b)[d["cv2_full"].notna()].groupby(["batch", "bin"], observed=True).agg(
        n=("cv2", "size"), cv2_full=("cv2_full", "median"), dev_s2p=("dev_s2p", "mean"),
        dev_full=("dev_full", "mean"))
    print(fb.round(3).to_string())


def main():
    if "--figures-only" in sys.argv:
        d = pd.read_csv(OUT_ROIS)
        z = np.load(OUT_ECDF)
        E, Es, Ef = z["prod"], z["s2p"], z["full"]
    else:
        d, E, Es, Ef = compute()
    summary(d, Es, Ef)
    fig_traces(d, _HERE / "fig_cv2_traces.png")
    fig_coverage(d, _HERE / "fig_cv2_coverage.png")
    fig_binned(d, [(E, "production null")], _HERE / "fig_cv2_binned.png")
    fig_floor(d, [_HERE / "fig_cv2_pixels.png", _HERE / "fig_cv2_overlay.png"])
    fig_full(d, Es, Ef, _HERE / "fig_cv2_full_resolution.png")
    print("  wrote fig_cv2_*.png")


if __name__ == "__main__":
    main()
