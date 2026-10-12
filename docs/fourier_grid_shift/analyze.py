"""Testing the stimulus frequency exactly when a trace is not a whole number of cycles:
grid shift vs truncation vs the standard FFT grid.

    python docs/fourier_grid_shift/analyze.py                  # full run (~5 min, magneto2 env)
    python docs/fourier_grid_shift/analyze.py --figures-only   # replot from saved CSVs

Outputs, written next to this script:
    results_alignment.csv     every imaging recording x analysed frequency: N, T, cycles,
                              bin offset, trimmed length, residual offset
    results_sinusoid.csv      pure sinusoid through production fit_Fourier, per mode:
                              power kept in the tested coefficient, frames used
    results_null.csv          white noise through production fit_Fourier + corrected_pvalues,
                              per mode: share of p < 0.01, KS test, neighbour correlation
    results_summary.csv       real 2022 Q1 recordings, per (recording, frequency, mode):
                              ROIs, suspects, 95% bound, NFC^2 gain of responders
    results_rois.csv          per-ROI NFC and p-value behind results_summary (gitignored)
    results_control.csv       recordings already on the grid: max |NFC difference| between modes
    results_reconcile.csv     NWB-persisted NFC vs this harness, per recording
    results_zero_pad.csv      the rejected alternative: power correlation of zero-padded bins
    results_sqrt2.csv         remove_flatlines' sqrt(2) test: ROIs dropped vs expected by chance
    results_example.csv       one ROI's trace for Figure 2
    fig_grids.png, fig_example.png

fit_Fourier reads the coefficient at the standard FFT bin nearest the stimulus
frequency f. When the trace's duration N*T is not a whole number of stimulus
cycles, that bin misses f. Two remedies are compared, both implemented in
production (magpyneto2.engert_helpers.fit_Fourier, `align=`): shifting the grid
so that it lands on f ("shift"), and dropping frames from the end until the
duration is close to whole cycles ("trim"). "none" is the standard grid, the
behaviour before 2026-10-12.
"""
import argparse
import functools
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from magpyneto2.engert_helpers import fit_Fourier, whole_cycle_length  # noqa: E402
from magpyneto2.statistics import corrected_pvalues, eps_from_Q, suspect_count_significance  # noqa: E402
from pipeline.schema import load_experiment  # noqa: E402
from pipeline import nwb_io  # noqa: E402
import pipeline.analysis_stages.engert as engert_stage  # noqa: E402
import pipeline.analysis_stages.medaka as medaka_stage  # noqa: E402

MODES = ["none", "trim", "shift"]
SHIFT_MIN = 0.01        # fit_Fourier keeps the standard grid below this |offset| (bins)
TRIM_TOL = 0.05         # whole_cycle_length's tolerance (cycles = bins)
N_NULL = 20000          # white-noise traces per (N, f, mode)
SEED = 0
EXAMPLE_REC = "engert_20220221_visual"
EXAMPLE_F = 1 / 60

# --------------------------------------------------------------------- style
C_NONE, C_TRIM, C_SHIFT = "#eb6834", "#1baf7a", "#2a78d6"
C_INK, C_MUTED = "#333333", "#8a8a8a"
MODE_COLOR = {"none": C_NONE, "trim": C_TRIM, "shift": C_SHIFT}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=C_INK, labelsize=9)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)


# ----------------------------------------------------------------- helpers
def offset(f, N, T):
    """Fractional bin offset of f from the nearest standard bin, f*N*T - round(f*N*T)."""
    k = f * N * T
    return k - round(k)


def kept(d, N):
    """Share of a sinusoid's power that lands in the nearest bin, offset d bins away
    (the Dirichlet kernel, |D_N(d)|^2 / N^2)."""
    return 1.0 if d == 0 else float((np.sin(np.pi * d) / (N * np.sin(np.pi * d / N))) ** 2)


def imaging_configs():
    """Every engert/medaka YAML, i.e. the same set the pipeline runs."""
    paths = sorted(glob.glob(str(_REPO_ROOT / "experiments" / "engert_*.yml"))
                   + glob.glob(str(_REPO_ROOT / "experiments" / "medaka_*.yml")))
    return [load_experiment(p) for p in paths]


def analysed_freqs(cfg, with_qfrac=False):
    """(label, frequency[, Q_frac]) for each fit the analysis stage makes for this recording."""
    f, T, q = cfg.analysis.f, cfg.sample_period, cfg.analysis.Q_frac
    out = [("1F", f, q)]
    if cfg.paradigm == "engert":
        if 2 * f < 0.5 / T:
            out.append(("2F", 2 * f, q))
        if cfg.analysis.visual_f > 0:
            qv = cfg.analysis.visual_Q_frac if cfg.analysis.visual_Q_frac > 0                 else engert_stage._DEFAULT_VISUAL_Q_FRAC
            out.append(("visual", cfg.analysis.visual_f, qv))
    else:
        out.append(("visual", medaka_stage._VISUAL_FREQ, medaka_stage._VISUAL_Q_FRAC))
    return out if with_qfrac else [o[:2] for o in out]


def stage_results(cfg, mode):
    """Production compute_fourier_results with fit_Fourier pinned to `mode`."""
    mod = engert_stage if cfg.paradigm == "engert" else medaka_stage
    orig = mod.fit_Fourier
    mod.fit_Fourier = functools.partial(fit_Fourier, align=mode)
    try:
        return mod.compute_fourier_results(cfg, verbose=False)
    finally:
        mod.fit_Fourier = orig


def long_rows(cfg, res, mode):
    """One row per (ROI, analysed frequency): NFC, p-value and bin count M."""
    df = res["fourier_df"]
    rows = []
    for lab, f in analysed_freqs(cfg):
        if lab == "2F":
            sub = df[np.isclose(df.freq, cfg.analysis.f)]
            NFC, p, M = sub["2f_NFC"].values, sub["2f_p_value"].values, res["Q_2f"]
        else:
            sub = df[np.isclose(df.freq, f)]
            NFC, p, M = sub["NFC"].values, sub["p_value"].values, int(sub["Q"].iloc[0])
        rows.append(pd.DataFrame(dict(rec=cfg.name, harmonic=lab, freq=f, mode=mode,
                                      id=sub["id"].values, NFC=NFC, p_value=p, M=M)))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- analyses
def alignment(cfgs):
    rows = []
    for cfg in cfgs:
        with h5py.File(cfg.nwb_path(), "r") as h:
            n = h["processing/ophys/Fluorescence/RoiResponseSeries/data"].shape[0]
        N = min(120 * (n // 60), n)
        T = cfg.sample_period
        for lab, f, q in analysed_freqs(cfg, with_qfrac=True):
            d = offset(f, N, T)
            Nt = whole_cycle_length(N, f, T, tol=TRIM_TOL)
            dt = offset(f, Nt, T)
            rows.append(dict(rec=cfg.name, harmonic=lab, freq=f, Q_frac=q, frames=n, N=N, T=T,
                             cycles=f * N * T, offset=d, kept_none=kept(d, N),
                             N_trim=Nt, dropped=N - Nt, offset_trim=dt, kept_trim=kept(dt, Nt),
                             off_grid=abs(d) >= SHIFT_MIN))
    return pd.DataFrame(rows)


def sinusoid(al):
    """A noiseless cosine at f through fit_Fourier: power in the tested coefficient,
    relative to the full (A*N_used/2)^2, and relative to the untrimmed N."""
    rows = []
    for (N, T, lab, f, q), _ in al[al.off_grid].groupby(["N", "T", "harmonic", "freq", "Q_frac"]):
        x = np.cos(2 * np.pi * f * np.arange(N) * T)[None, :]
        for mode in MODES:
            _, on, _, _, _, _ = fit_Fourier(x, T=T, f=f, Q_frac=q, align=mode)
            Nu = whole_cycle_length(N, f, T, tol=TRIM_TOL) if mode == "trim" else N
            p_frac = abs(on[0]) ** 2 / (Nu / 2) ** 2
            # A weak sinusoid's expected NFC^2 scales as (frames used) x (power kept).
            rows.append(dict(N=N, T=T, harmonic=lab, freq=f, Q_frac=q, mode=mode, N_used=Nu,
                             power_kept=p_frac, sensitivity=p_frac * Nu / N))
    return pd.DataFrame(rows)


def null(al):
    """White noise through production fit_Fourier + corrected_pvalues, same N, T, f and
    Q_frac as the real recordings. The p-values must be uniform in every mode."""
    rng = np.random.default_rng(SEED)
    rows = []
    for (N, T, lab, f, q), _ in al[al.off_grid].groupby(["N", "T", "harmonic", "freq", "Q_frac"]):
        W = rng.standard_normal((N_NULL, N))
        for mode in MODES:
            NFC, _, off, _, M, _ = fit_Fourier(W, T=T, f=f, Q_frac=q, align=mode)
            p = corrected_pvalues(np.array(NFC), M)
            P = np.abs(np.array(off)) ** 2          # (traces, 2M) off-frequency power
            r_nb = np.corrcoef(P[:, M], P[:, M + 1])[0, 1]   # two neighbours above f
            rows.append(dict(N=N, T=T, harmonic=lab, freq=f, Q_frac=q, mode=mode, M=M, n=N_NULL,
                             frac_p01=np.mean(p < 0.01), frac_p05=np.mean(p < 0.05),
                             ks_p=scipy.stats.kstest(p, "uniform").pvalue,
                             neighbour_corr=r_nb))
    return pd.DataFrame(rows)


def zero_pad(al):
    """The rejected alternative: zero-pad the trace to 4N and read the finer FFT. Its
    neighbouring bins are 1/4 of a resolution bin apart; correlation of their power
    in white noise, at 1/4 ... 1 bin of separation (1 bin = what the null assumes, 0)."""
    rng = np.random.default_rng(SEED + 1)
    N, T, f = 1194, 1.0212, 0.4
    W = rng.standard_normal((N_NULL, N))
    X = np.fft.fft(W - W.mean(axis=1, keepdims=True), n=4 * N, axis=1)
    k = int(round(f * 4 * N * T)) + 8          # a padded bin away from the stimulus
    P = np.abs(X) ** 2
    return pd.DataFrame([dict(N=N, T=T, freq=f, pad=4, separation_bins=s / 4,
                              power_corr=np.corrcoef(P[:, k], P[:, k + s])[0, 1])
                         for s in (1, 2, 3, 4)])


def sqrt2_test(cfgs):
    """remove_flatlines drops a trace whose 0.2 Hz NFC is within 1% of sqrt(2) (the value
    an isolated delta spike gives). An ordinary trace lands there by chance: under the
    null NFC^2/2 ~ Exp(1), so P = exp(-0.99^2) - exp(-1.01^2) = 1.47%. Count the drops."""
    import magpyneto2.engert_helpers as eh
    rows = []
    for cfg in cfgs:
        mod = engert_stage if cfg.paradigm == "engert" else medaka_stage
        seen = {}
        orig = mod.remove_flatlines

        def spy(F, *a, **k):
            seen["F"] = F
            return orig(F, *a, **k)
        mod.remove_flatlines = spy
        try:
            mod._load_from_nwb(cfg.nwb_path(), cfg.iscell_threshold, cfg.npix_threshold,
                               outline=mod.body_outline.params_for(cfg.body_outline),
                               coverage_min=cfg.coverage_threshold, sample_period=cfg.sample_period)
        finally:
            mod.remove_flatlines = orig
        F = seen["F"]
        out = {}
        for mode in ("none", "shift"):
            NFC = np.array(eh.fit_Fourier(F, T=1, f=0.2, Q_frac=0.15, align=mode)[0])
            out[mode] = np.isclose(NFC, np.sqrt(2), rtol=0.01)
        rows.append(dict(rec=cfg.name, rois=len(F), N=F.shape[1], dropped=int(out["none"].sum()),
                         expected=len(F) * (np.exp(-0.99 ** 2) - np.exp(-1.01 ** 2)),
                         dropped_if_shifted=int(out["shift"].sum()),
                         differ_if_shifted=int((out["none"] != out["shift"]).sum())))
    return pd.DataFrame(rows)


def real(cfgs, al):
    """Production analysis-stage results for the off-grid recordings, in all three modes."""
    off = set(al.loc[al.off_grid, "rec"])
    parts = []
    for cfg in cfgs:
        if cfg.name not in off:
            continue
        for mode in MODES:
            parts.append(long_rows(cfg, stage_results(cfg, mode), mode))
        print(f"  {cfg.name}")
    return pd.concat(parts, ignore_index=True)


def summarise(rois):
    rows = []
    for (rec, lab, f), g in rois.groupby(["rec", "harmonic", "freq"], sort=False):
        piv = g.pivot(index="id", columns="mode", values="NFC")
        sus_any = g[g.p_value < 0.01].id.unique()
        for mode, gm in g.groupby("mode"):
            # Same test as Fig 2 (fig2.flag_sessions_with_excess_suspects): suspects above
            # the eps-corrected null's 99th percentile, against the binomial 95% bound.
            n, _, _, hi = suspect_count_significance(gm.NFC.values, 0.99, 0.95,
                                                     eps=eps_from_Q(gm.M.iloc[0]))
            # NFC^2 is the power statistic; its ratio on the same ROIs isolates the grid.
            ratio = (piv.loc[sus_any, mode] ** 2 / piv.loc[sus_any, "none"] ** 2).median() \
                if len(sus_any) else np.nan
            rows.append(dict(rec=rec, harmonic=lab, freq=f, mode=mode, rois=len(gm),
                             M=int(gm.M.iloc[0]), suspects=int(n), bound=float(hi), flagged=n > hi,
                             responders=len(sus_any), median_nfc2_ratio=ratio))
    return pd.DataFrame(rows)


def control(cfgs, al):
    """Recordings already on the grid (|offset| < SHIFT_MIN at every analysed frequency):
    all three modes must give identical NFC, ROI for ROI."""
    on = set(al.rec) - set(al.loc[al.off_grid, "rec"])
    rows = []
    for cfg in cfgs:
        if cfg.name not in on:
            continue
        r = {m: long_rows(cfg, stage_results(cfg, m), m) for m in MODES}
        ref = r["none"].NFC.values
        rows.append(dict(rec=cfg.name, rois=r["none"].id.nunique(),
                         max_diff_shift=float(np.nanmax(np.abs(r["shift"].NFC.values - ref))),
                         max_diff_trim=float(np.nanmax(np.abs(r["trim"].NFC.values - ref)))))
    return pd.DataFrame(rows)


def reconcile(cfgs, rois):
    """The NFC persisted in each recording's NWB file vs this harness. The NWB must
    match exactly one mode: "none" before the pipeline re-run, "shift" (the new
    default) after it. Any other outcome means the harness is not the production path."""
    rows = []
    for cfg in cfgs:
        io, nwb = nwb_io.read_nwbfile(cfg.nwb_path())
        try:
            per = nwb_io.read_fourier_results_as_full_fourier_df(nwb)
        finally:
            io.close()
        mine = rois[rois.rec == cfg.name]
        if mine.empty:      # on-grid recording: all modes equal, recompute once
            mine = long_rows(cfg, stage_results(cfg, "shift"), "shift")
        row = dict(rec=cfg.name)
        for mode, gm in mine.groupby("mode"):
            gm = gm[gm.harmonic != "2F"]
            m = gm.merge(per, on=["id", "freq"], suffixes=("", "_nwb"))
            same_set = len(m) == len(gm) == len(per)
            row[f"max_diff_{mode}"] = float(np.nanmax(np.abs(m.NFC - m.NFC_nwb))) if same_set else np.inf
        rows.append(row)
    return pd.DataFrame(rows)


def example(rois, cfgs):
    """The strongest 1/60 Hz responder of EXAMPLE_REC (shift mode), its raw trace."""
    cfg = next(c for c in cfgs if c.name == EXAMPLE_REC)
    g = rois[(rois.rec == EXAMPLE_REC) & (rois["mode"] == "shift")
             & np.isclose(rois.freq, EXAMPLE_F, atol=1e-5)]
    best = int(g.loc[g.NFC.idxmax(), "id"])
    io, nwb = nwb_io.read_nwbfile(cfg.nwb_path())
    try:
        F_all, _ = nwb_io.read_roi_data(nwb, "full")
    finally:
        io.close()
    return pd.DataFrame(dict(rec=EXAMPLE_REC, id=best, frame=np.arange(F_all.shape[1]),
                             F=F_all[best].astype(float)))


# ------------------------------------------------------------------ figures
def dtft(x, freqs, T):
    n = np.arange(len(x))
    return np.exp(-2j * np.pi * np.outer(freqs, n) * T) @ x


def grid_panel(ax, x, T, f, N, mode, label):
    """|X(nu)| of x[:N] (continuous curve) and the coefficients `mode` reads."""
    y = x[:N] - x[:N].mean()
    res = 1 / (N * T)
    nu = f + np.linspace(-4.5, 4.5, 901) * res
    ax.plot((nu - f) / res, np.abs(dtft(y, nu, T)), color=C_MUTED, lw=1)
    k0 = round(f * N * T)
    d = 0.0 if mode != "shift" else f * N * T - k0
    ks = np.arange(-4, 5)
    grid = (k0 + ks + d) / (N * T)
    mag = np.abs(dtft(y, grid, T))
    pos = (grid - f) / res
    ax.vlines(pos, 0, mag, color=MODE_COLOR[mode], lw=1.5)
    ax.plot(pos[ks != 0], mag[ks != 0], "o", ms=5, mfc="white", mec=MODE_COLOR[mode], mew=1.5)
    ax.plot(pos[ks == 0], mag[ks == 0], "o", ms=7, color=MODE_COLOR[mode])
    ax.axvline(0, color=C_INK, lw=0.8, ls=":")
    ax.set_title(label, fontsize=9.5, color=C_INK, loc="left")
    ax.set_xlim(-4.6, 4.6)
    _style(ax)


def fig_grids(out):
    """A pure 0.4 Hz cosine, N = 1194, T = 1.0212 s (2022_02_21/23), on the three grids."""
    N, T, f = 1194, 1.0212, 0.4
    Nt = whole_cycle_length(N, f, T, tol=TRIM_TOL)
    x = np.cos(2 * np.pi * f * np.arange(N) * T)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    grid_panel(axes[0], x, T, f, N, "none", f"a  standard grid, N = {N}")
    grid_panel(axes[1], x, T, f, Nt, "trim", f"b  trimmed, N = {Nt}")
    grid_panel(axes[2], x, T, f, N, "shift", f"c  shifted grid, N = {N}")
    axes[0].set_ylabel("|X|")
    for ax in axes:
        ax.set_xlabel("frequency − 0.4 Hz  (bins of 1/(N·T))")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_example(ex, out):
    """One ROI's raw trace and its spectrum around 1/60 Hz on the three grids."""
    x = ex.F.values
    N, T, f = len(x), 1.0212, EXAMPLE_F
    Nt = whole_cycle_length(N, f, T, tol=TRIM_TOL)
    fig = plt.figure(figsize=(10, 5.2))
    ax0 = fig.add_subplot(2, 1, 1)
    t = np.arange(N) * T
    ax0.plot(t, x, color=C_INK, lw=0.6)
    ax0.set_xlabel("time (s)")
    ax0.set_ylabel("F")
    ax0.set_xlim(0, N * T)
    ax0.set_title(f"a  {ex.rec.iloc[0]}, ROI {ex.id.iloc[0]}", fontsize=9.5, color=C_INK, loc="left")
    _style(ax0)
    axes = [fig.add_subplot(2, 3, 4 + i) for i in range(3)]
    grid_panel(axes[0], x, T, f, N, "none", f"b  standard grid, N = {N}")
    grid_panel(axes[1], x, T, f, Nt, "trim", f"c  trimmed, N = {Nt}")
    grid_panel(axes[2], x, T, f, N, "shift", f"d  shifted grid, N = {N}")
    top = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, top)
        ax.set_xlabel("frequency − 1/60 Hz  (bins)")
    axes[0].set_ylabel("|X|")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--figures-only", action="store_true",
                    help="replot from the saved CSVs without re-running the analysis")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    if not args.figures_only:
        cfgs = imaging_configs()
        print("Alignment")
        al = alignment(cfgs)
        al.to_csv(_HERE / "results_alignment.csv", index=False)
        print(al[al.off_grid].to_string(index=False))

        print("Sinusoid")
        sn = sinusoid(al)
        sn.to_csv(_HERE / "results_sinusoid.csv", index=False)
        print(sn.to_string(index=False))

        print("White-noise null")
        nl = null(al)
        nl.to_csv(_HERE / "results_null.csv", index=False)
        print(nl.to_string(index=False))
        # Tripwire: a grid that breaks the null shows up as an excess of small p-values.
        assert (nl.frac_p01 - 0.01).abs().max() < 0.004, "a mode's null p-values are off"

        print("Zero-padding (rejected)")
        zp = zero_pad(al)
        zp.to_csv(_HERE / "results_zero_pad.csv", index=False)
        print(zp.to_string(index=False))

        print("remove_flatlines' sqrt(2) test")
        sq = sqrt2_test(cfgs)
        sq.to_csv(_HERE / "results_sqrt2.csv", index=False)
        print(sq.to_string(index=False))
        print(f"  total dropped {sq.dropped.sum()} of {sq.rois.sum()} ROIs "
              f"({sq.dropped.sum() / sq.rois.sum():.2%}), expected by chance {sq.expected.sum():.0f}")

        print("Real recordings (off the grid)")
        rois = real(cfgs, al)
        rois.to_csv(_HERE / "results_rois.csv", index=False)
        sm = summarise(rois)
        sm.to_csv(_HERE / "results_summary.csv", index=False)
        print(sm.to_string(index=False))

        print("Control: recordings on the grid")
        ct = control(cfgs, al)
        ct.to_csv(_HERE / "results_control.csv", index=False)
        print(f"  {len(ct)} recordings, max |dNFC| shift {ct.max_diff_shift.max():.1e}, "
              f"trim {ct.max_diff_trim.max():.1e}")
        assert (ct.max_diff_shift == 0).all() and (ct.max_diff_trim == 0).all(), \
            "modes differ on a recording that is already on the grid"

        print("Reconciliation with the NWB files")
        rc = reconcile(cfgs, rois)
        rc.to_csv(_HERE / "results_reconcile.csv", index=False)
        for mode in MODES:
            rc[f"nwb_is_{mode}"] = rc[f"max_diff_{mode}"] < 1e-9
        rc.to_csv(_HERE / "results_reconcile.csv", index=False)
        print(rc.to_string(index=False))

        example(rois, cfgs).to_csv(_HERE / "results_example.csv", index=False)

    fig_grids(_HERE / "fig_grids.png")
    fig_example(pd.read_csv(_HERE / "results_example.csv"), _HERE / "fig_example.png")


if __name__ == "__main__":
    main()
