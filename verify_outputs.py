"""
Compares magnet_search's current NWB output against the frozen originals in
MagnetSearch/data/.

Old naming convention:  {name}_{GOOD}_modulation_df.pickle / {name}_{GOOD}_full_fourier_df.pickle
Current output:         {name}.nwb (see pipeline/nwb_io.py)

As of the Phase 7 cutover (.claude/plans — NWB replatform), no paradigm
writes `{name}_processing.pickle`/`{name}_analysis.pickle` anymore -- the
old fixtures above are the ONLY frozen ground truth left, and this script
compares them directly against frames reconstructed from `{name}.nwb`
(`nwb_io.build_modulation_frame`/`read_fourier_results_as_full_fourier_df`).
Row order is never meaningful (dict/table iteration order, not a semantic
property), so every comparison here sorts both frames first — see
`_frames_equal_sorted`.

Run after migrating each experiment to confirm outputs match.
"""
import os
import pickle
import numpy as np
import pandas as pd

OLD_DATA = r"C:\Users\dan\Documents\MagnetSearch\data"
NEW_DATA = r"C:\Users\dan\Documents\magnet_search\data"

# Numeric tolerance for the NWB-vs-legacy comparison arm (see
# `_frames_equal_sorted`/`DRIFT_TOLERATED` below). This is deliberately far
# looser than pickle-vs-pickle's exact-match PASS: the NWB path derives
# phase/period on demand from period_crossings via
# `correct_theta_at_samples`/`compute_period_at_samples` (int64 sample-domain
# arithmetic, exact by construction — see pipeline/nwb_io.py's module
# docstring) rather than loading a precomputed column, so any real mismatch
# here indicates a genuine logic bug, not floating-point noise. The
# tolerance exists only to absorb float round-trip through HDF5 storage.
NWB_DRIFT_ATOL = 1e-9
# Relative counterpart to NWB_DRIFT_ATOL -- needed because reordering units
# (e.g. good-only NWB writes changing dict/array iteration order feeding
# fit_fourier_sig's vectorized ops) shifts float64 summation order, producing
# differences that scale with a column's own magnitude, not a fixed
# absolute size. Confirmed on real data (20220916's sens/sens_2f columns,
# not O(1) like pp/rr): up to ~1.5e-8 absolute but only ~1e-11 *relative* --
# atol alone can't cover every column's scale without being loose enough to
# risk hiding a real bug in the O(1) columns.
NWB_DRIFT_RTOL = 1e-6


# Map experiment name → (old_processing_file, old_analysis_file)
# old files use {name}_{good} prefix; update if GOOD=False for any experiment
PAIRS = {
    # --- spikeglx_direct ---
    "Q146_20230815_g0": (
        "Q146_20230815_g0_modulation_df.pickle",
        "Q146_20230815_g0_full_fourier_df.pickle",
    ),
    "Q148_20241219_g0": (
        "Q148_20241219_g0_modulation_df.pickle",
        "Q148_20241219_g0_full_fourier_df.pickle",
    ),
    "Q148_20241219_g1": (
        "Q148_20241219_g1_modulation_df.pickle",
        "Q148_20241219_g1_full_fourier_df.pickle",
    ),
    "magnerNPX2_g0": (
        "magnerNPX2_g0_modulation_df.pickle",
        "magnerNPX2_g0_full_fourier_df.pickle",
    ),
    # --- gutfreund ---
    "Q117_20221213": (
        "Q117_20221213_modulation_df.pickle",
        "Q117_20221213_full_fourier_df.pickle",
    ),
    "Q117_20221214": (
        "Q117_20221214_modulation_df.pickle",
        "Q117_20221214_full_fourier_df.pickle",
    ),
    "Q134_20240111_s01": (
        "Q134_20240111_s01_modulation_df.pickle",
        "Q134_20240111_s01_full_fourier_df.pickle",
    ),
    # --- openephys ---
    "20220228_firstsite": (
        "20220228_firstsite_True_modulation_df.pickle",
        "20220228_firstsite_True_full_fourier_df.pickle",
    ),
    "20220228_secondsite": (
        "20220228_secondsite_True_modulation_df.pickle",
        "20220228_secondsite_True_full_fourier_df.pickle",
    ),
    "20220314": (
        "20220314_True_modulation_df.pickle",
        "20220314_True_full_fourier_df.pickle",
    ),
    "20220408": (
        "20220408_True_modulation_df.pickle",
        "20220408_True_full_fourier_df.pickle",
    ),
    "20220421": (
        "20220421_True_modulation_df.pickle",
        "20220421_True_full_fourier_df.pickle",
    ),
    "20220621": (
        "20220621_True_modulation_df.pickle",
        "20220621_True_full_fourier_df.pickle",
    ),
    # --- openephys_multistim ---
    "20220916": (
        "20220916_True_modulation_df.pickle",
        "20220916_True_full_fourier_df.pickle",
    ),
    "20230216": (
        "20230216_True_modulation_df.pickle",
        "20230216_True_full_fourier_df.pickle",
    ),
    "20230221": (
        "20230221_True_modulation_df.pickle",
        "20230221_True_full_fourier_df.pickle",
    ),
    "20230228": (
        "20230228_True_modulation_df.pickle",
        "20230228_True_full_fourier_df.pickle",
    ),
    "20230413_firstsite": (
        "20230413_firstsite_True_modulation_df.pickle",
        "20230413_firstsite_True_full_fourier_df.pickle",
    ),
    "20230413_secondsite": (
        "20230413_secondsite_True_modulation_df.pickle",
        "20230413_secondsite_True_full_fourier_df.pickle",
    ),
    "20230414_firstsite": (
        "20230414_firstsite_True_modulation_df.pickle",
        "20230414_firstsite_True_full_fourier_df.pickle",
    ),
    "20230415": (
        "20230415_True_modulation_df.pickle",
        "20230415_True_full_fourier_df.pickle",
    ),
}


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _frames_equal_sorted(a, b, sort_cols, atol=0.0, rtol=0.0):
    """Sorts both frames by `sort_cols` first, then compares, with an
    optional numeric tolerance.

    Needed for the NWB comparison arm: NWB-reconstructed frames are built by
    iterating Units/epochs tables (dict/DataFrame order), which need not
    match the old fixture's original row order — a plain positional
    `reset_index` comparison would spuriously fail on pure reordering even
    when every value matches.

    `rtol` matters alongside `atol`: confirmed on real data (20220916's
    `sens`/`sens_2f` columns) that excluding MUA units reorders the array
    feeding fit_fourier_sig's vectorized ops, shifting float64 summation
    order enough to produce ~1.5e-8 absolute (but only ~1e-11 *relative*)
    differences in columns whose values aren't O(1) like pp/rr are --
    atol alone can't cover every column's scale without also being loose
    enough to hide a real bug in the O(1) columns. Default 0 for the
    strict/PASS-tier check (see _compare_tiered), non-zero only for the
    DRIFT_TOLERATED tier.
    """
    try:
        a_sorted = a.sort_values(sort_cols).reset_index(drop=True)
        b_sorted = b.sort_values(sort_cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            a_sorted, b_sorted, check_like=True, check_dtype=False,
            atol=atol, rtol=rtol,
        )
        return True, None
    except AssertionError as e:
        return False, e


def _load_experiment_by_name(cfg_name):
    """Load the experiment YAML whose `name:` field equals `cfg_name`.

    Usually that's just `experiments/{cfg_name}.yml`, but a few YAMLs' own
    filenames don't match their internal `name:` (e.g.
    `experiments/Q146_20230815.yml` declares `name: "Q146_20230815_g0"`,
    matching the PAIRS key and `data/Q146_20230815_g0*` output files but not
    its own filename) -- fall back to scanning every YAML for a `name:`
    match rather than assuming the filename convention holds.
    """
    from pipeline.schema import load_experiment
    experiments_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
    direct_path = os.path.join(experiments_dir, f"{cfg_name}.yml")
    if os.path.exists(direct_path):
        cfg = load_experiment(direct_path)
        if cfg.name == cfg_name:
            return cfg
    for fname in os.listdir(experiments_dir):
        if not fname.endswith(".yml"):
            continue
        cfg = load_experiment(os.path.join(experiments_dir, fname))
        if cfg.name == cfg_name:
            return cfg
    raise FileNotFoundError(f"no experiment YAML with name '{cfg_name}' found in {experiments_dir}")


def _compare_tiered(old_df, nwb_df, sort_cols):
    """Sorted comparison with two tolerance tiers, shared by every arm that
    diffs a frozen fixture (or an independently-recomputed frame, for
    engert/medaka) against an NWB-reconstructed frame.

    Row order is never meaningful here (dict/table iteration order, not a
    semantic property of either side), so this sorts both frames FIRST via
    `_frames_equal_sorted` and treats that as the primary/only comparison
    mode -- unlike the old pickle-vs-pickle check this replaces, there's no
    separate "REORDERED" tier needed: a pure reordering difference already
    passes the atol=0 sorted compare.

    Returns (status: "PASS"/"DRIFT_TOLERATED: <detail>"/"FAIL: <detail>",
    exact_err: the atol=0 comparison's AssertionError, for STALE_OLD
    reconciliation to fall back to on a real FAIL).
    """
    exact_ok, exact_err = _frames_equal_sorted(old_df, nwb_df, sort_cols, atol=0.0)
    if exact_ok:
        return "PASS", None

    tol_ok, tol_err = _frames_equal_sorted(
        old_df, nwb_df, sort_cols, atol=NWB_DRIFT_ATOL, rtol=NWB_DRIFT_RTOL)
    if tol_ok:
        return (f"DRIFT_TOLERATED (within atol={NWB_DRIFT_ATOL}, rtol={NWB_DRIFT_RTOL}): "
                f"{exact_err}", None)

    return f"FAIL: {tol_err}", exact_err


_OK_TAGS = ("PASS", "SKIP", "STALE_OLD", "STALE_SEMANTICS", "DRIFT_TOLERATED")


def _is_ok(tag):
    return any(t in tag for t in _OK_TAGS)


def _print_result_row(name, results, cols):
    """Print one summary line (truncated, for a quick-glance sweep) plus,
    for any arm that's actually failing, the FULL untruncated detail on its
    own line right below -- no more needing an ad-hoc script to see the
    real DataFrame diff every time something fails (a recurring friction
    point before this rewrite: the old fixed `[:40]`/`[:50]` truncation hid
    the one thing you actually need to see).

    `cols`: list of (key, label) pairs identifying which `results` entries
    to render, in order.
    """
    status = "OK" if all(_is_ok(results.get(k, "-")) for k, _ in cols) else "!!"
    short = "  ".join(f"{label}={results.get(k, '-')[:50]}" for k, label in cols)
    print(f"[{status}] {name:40s}  {short}")
    if status == "!!":
        for k, label in cols:
            val = results.get(k, "-")
            if not _is_ok(val):
                print(f"    {label}: {val}")
    return status == "OK"


def _reconcile_under_current_Q_semantics(name, old_proc_df, nwb_anal_df):
    """Re-run fit_fourier_sig() on the OLD modulation_df using THIS
    experiment's actual analysis config, and check whether the result now
    matches nwb_anal_df (the frame reconstructed from `{name}.nwb`) -- i.e.
    is the analysis-stage mismatch fully explained by fit_fourier_sig() having
    evolved since the old fixture was generated, not a real bug?

    Uses per-experiment Q_frac/per-type Q_frac from THIS experiment's own
    YAML, matching analysis_stages/simple.py (single Q_frac) and
    analysis_stages/multistim.py (per-type Q_frac splitting) instead of
    guessing a single global value.

    NOTE: as of the Q-redefinition (raw integer bin count -> fraction of the
    analyzed frequency), a mismatch reconciled here is virtually never "the
    algorithm evolved" in the STALE_OLD sense anymore -- `old_anal_df` was
    generated under the OLD integer-Q semantics entirely, so re-running under
    the NEW fraction semantics reconciling successfully mostly reflects that
    deliberate redefinition, not an incidental evolution. verify_experiment
    labels a successful reconciliation `STALE_SEMANTICS` rather than
    `STALE_OLD` to keep that distinct.

    Returns (ok: bool, fresh_ff: DataFrame or None).
    """
    from magpyneto2.statistics import fit_fourier_sig as _fo
    cfg = _load_experiment_by_name(name)
    a = cfg.analysis

    if cfg.paradigm == "openephys_multistim" and a.mag_Q_frac > 0:
        sub_dfs = []
        mag_mask = [a.mag_rec_substring in rec for rec in old_proc_df.rec]
        visual_mask = [a.visual_rec_substring in rec for rec in old_proc_df.rec]
        wn_mask = [a.wn_rec_substring in rec for rec in old_proc_df.rec]
        for mask, frac, _label in [(mag_mask, a.mag_Q_frac, "mag"),
                                    (visual_mask, a.visual_Q_frac, "visual"),
                                    (wn_mask, a.WN_Q_frac, "WN")]:
            sub = old_proc_df.loc[mask]
            if sub.empty:
                continue
            fourier_df, _ = _fo(sub, Q_frac=frac, diagnostics=False)
            sub_dfs.append(fourier_df)
        fresh_ff = pd.concat(sub_dfs).reset_index(drop=True) if sub_dfs else None
    else:
        fresh_ff, _ = _fo(old_proc_df, Q_frac=a.Q_frac, diagnostics=False)

    if fresh_ff is None:
        return False, None
    cols = [c for c in fresh_ff.columns if c in nwb_anal_df.columns]
    # atol=NWB_DRIFT_ATOL, not 0.0: confirmed on real data (20220314) that
    # reordering units (e.g. excluding MUA from the NWB Units table changes
    # dict/array iteration order feeding fit_fourier_sig) shifts float64
    # summation order enough to produce ~1e-13-relative-magnitude
    # differences in `pp`/`rr` -- the exact same class of harmless
    # numerical noise NWB_DRIFT_ATOL already exists to absorb elsewhere,
    # not a real correctness difference (values match to 10+ significant
    # digits).
    status, _ = _compare_tiered(fresh_ff[cols], nwb_anal_df[cols], ["rec", "id"])
    ok = status == "PASS" or status.startswith("DRIFT_TOLERATED")
    return ok, fresh_ff


def verify_experiment(name, old_proc_file, old_anal_file, verbose=True):
    """Compare the frozen MagnetSearch/data/ fixture directly against the
    frame reconstructed from this experiment's `{name}.nwb` -- the only
    comparison possible post-Phase-7 cutover, since no paradigm writes a
    `{name}_processing.pickle`/`{name}_analysis.pickle` anymore for there to
    be a "new pickle" middle-man to diff against.
    """
    results = {}

    nwb_path = os.path.join(NEW_DATA, f"{name}.nwb")
    old_proc_path = os.path.join(OLD_DATA, old_proc_file)
    old_anal_path = os.path.join(OLD_DATA, old_anal_file)

    old_proc_df = _load(old_proc_path)
    old_anal_df = _load(old_anal_path)

    if not os.path.exists(nwb_path):
        results["processing"] = "SKIP (no .nwb yet)"
        results["analysis"] = "SKIP (no .nwb yet)"
        if verbose:
            _print_result_row(name, results, [("processing", "processing"), ("analysis", "analysis")])
        return results

    from pipeline import nwb_io
    cfg = _load_experiment_by_name(name)
    io_r, nwbfile = nwb_io.read_nwbfile(nwb_path)
    try:
        nwb_proc_df = nwb_io.build_modulation_frame(nwbfile, good_only=cfg.good)
        nwb_anal_df = nwb_io.read_fourier_results_as_full_fourier_df(nwbfile)
    except Exception as exc:
        results["processing"] = f"FAIL: could not read {nwb_path}: {exc}"
        results["analysis"] = results["processing"]
        io_r.close()
        if verbose:
            _print_result_row(name, results, [("processing", "processing"), ("analysis", "analysis")])
        return results
    finally:
        io_r.close()

    # --- processing ---
    if old_proc_df is None:
        results["processing"] = "SKIP (old file missing)"
    else:
        # build_modulation_frame deliberately reconstructs only the
        # canonical 6-column modulation_df shape (period, spk, phase, freq,
        # id, rec) -- some paradigms' old fixtures carry extra diagnostic-
        # only columns on top of that (e.g. gutfreund's spk_samples/label/
        # recname). Compare on nwb_proc_df's own columns only.
        cols = [c for c in nwb_proc_df.columns if c in old_proc_df.columns]
        results["processing"], _ = _compare_tiered(old_proc_df[cols], nwb_proc_df[cols], ["rec", "id", "spk"])

    # --- analysis ---
    if old_anal_df is None:
        results["analysis"] = "SKIP (old file missing)"
    else:
        cols = [c for c in nwb_anal_df.columns if c in old_anal_df.columns]
        status, exact_err = _compare_tiered(old_anal_df[cols], nwb_anal_df[cols], ["rec", "id"])
        if status == "PASS" or status.startswith("DRIFT_TOLERATED"):
            results["analysis"] = status
        elif old_proc_df is not None:
            # Check whether the discrepancy is fully explained by re-running
            # fit_fourier_sig on the old modulation_df with THIS experiment's
            # actual (post Q-redefinition) Q_frac config (see
            # _reconcile_under_current_Q_semantics) -- almost always the
            # Q-semantics redefinition itself for these ancient fixtures, not
            # an incidental algorithm change, hence STALE_SEMANTICS rather
            # than STALE_OLD (see CLAUDE.md's verify_outputs.py behaviour section).
            try:
                ok2, _ = _reconcile_under_current_Q_semantics(name, old_proc_df, nwb_anal_df)
                results["analysis"] = (
                    "STALE_SEMANTICS (Q redefined int-bins -> freq-fraction; "
                    "new matches fresh run under current Q_frac)"
                    if ok2 else status)
            except Exception:
                results["analysis"] = status
        else:
            results["analysis"] = status

    if verbose:
        _print_result_row(name, results, [("processing", "processing"), ("analysis", "analysis")])

    return results


def _discover_ophys_experiments():
    """engert/medaka experiments, auto-discovered from experiments/*.yml by
    paradigm (unlike PAIRS' NPIX entries, there's no hardcoded list here --
    processing was a no-op before Phase 4, so there's no legacy
    MagnetSearch/data/ fixture for these at all; see verify_ophys_experiment
    for why that changes what "verify" even means for this paradigm)."""
    from pipeline.schema import load_experiment
    names = []
    experiments_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
    for fname in sorted(os.listdir(experiments_dir)):
        if not fname.endswith(".yml"):
            continue
        cfg = load_experiment(os.path.join(experiments_dir, fname))
        if cfg.paradigm in ("engert", "medaka"):
            names.append(cfg.name)
    return names


def verify_ophys_experiment(name, verbose=True):
    """engert/medaka have no legacy MagnetSearch/data/ fixture (processing
    was a no-op before Phase 4 -- there was never a frozen "ground truth"
    pickle to diff against), and post-Phase-7 cutover there's no longer a
    parallel `{name}_analysis.pickle` from "this run" either. The only
    meaningful check left is serialization fidelity: independently
    recompute fourier_df via the SAME production code path
    (`analysis_stages.{engert,medaka}.compute_fourier_results`, which does
    no NWB writing itself) and diff it against what
    write_imaging_fourier_results/read_fourier_results_as_full_fourier_df
    actually persisted. This exercises the exact same round trip the old
    pickle-based check did (PlaneSegmentation/RoiResponseSeries read,
    iscell/npix+flatline filtering, Fourier fit, results write+read) --
    just sourced from a fresh in-memory recomputation instead of a
    now-retired pickle side-channel.

    Returns a results dict with a single "analysis_nwb" key, matching the
    NPIX arms' status vocabulary (PASS/DRIFT_TOLERATED/FAIL/SKIP).
    """
    nwb_path = os.path.join(NEW_DATA, f"{name}.nwb")
    if not os.path.exists(nwb_path):
        results = {"analysis_nwb": "SKIP (no .nwb yet)"}
        if verbose:
            _print_result_row(name, results, [("analysis_nwb", "analysis_nwb")])
        return results

    try:
        from pipeline import nwb_io
        cfg = _load_experiment_by_name(name)
        if cfg.paradigm == "engert":
            from pipeline.analysis_stages.engert import compute_fourier_results
        else:
            from pipeline.analysis_stages.medaka import compute_fourier_results
        fresh_df = compute_fourier_results(cfg, verbose=False)["fourier_df"]

        io_r, nwbfile = nwb_io.read_nwbfile(nwb_path)
        try:
            nwb_df = nwb_io.read_fourier_results_as_full_fourier_df(nwbfile)
        finally:
            io_r.close()

        # medaka's own fourier_df has 5 extra columns (date/area/ID/species/
        # contingency) the shared results tables don't persist (see
        # analysis_stages/medaka.py) -- compare only what both sides have.
        shared_cols = [c for c in nwb_df.columns if c in fresh_df.columns]
        status, _ = _compare_tiered(fresh_df[shared_cols], nwb_df[shared_cols], ["freq", "id"])
        results = {"analysis_nwb": status}
    except Exception as exc:
        results = {"analysis_nwb": f"FAIL: could not verify {nwb_path}: {exc}"}

    if verbose:
        _print_result_row(name, results, [("analysis_nwb", "analysis_nwb")])

    return results


def main(experiments=None):
    pairs = {k: v for k, v in PAIRS.items() if k in experiments} if experiments else PAIRS
    ophys_names = [n for n in _discover_ophys_experiments()
                   if experiments is None or n in experiments]
    all_pass = True
    for name, (old_proc, old_anal) in pairs.items():
        r = verify_experiment(name, old_proc, old_anal)
        if not all(_is_ok(v) for v in r.values()):
            all_pass = False
    for name in ophys_names:
        r = verify_ophys_experiment(name)
        if not all(_is_ok(v) for v in r.values()):
            all_pass = False

    print("\nAll PASS" if all_pass else "\nSome FAIL — see above")


if __name__ == "__main__":
    import sys
    exps = sys.argv[1:] if len(sys.argv) > 1 else None
    main(exps)
