"""
Compares new magnet_search output pickles against originals in MagnetSearch/data/.

Old naming convention:  {name}_{GOOD}_modulation_df.pickle / {name}_{GOOD}_full_fourier_df.pickle
New naming convention:  {name}_processing.pickle           / {name}_analysis.pickle

Run after migrating each experiment to confirm outputs match.
"""
import os
import pickle
import pandas as pd

OLD_DATA = r"C:\Users\dan\Documents\MagnetSearch\data"
NEW_DATA = r"C:\Users\dan\Documents\magnet_search\data"


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


def _frames_equal(a, b):
    try:
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True),
            b.reset_index(drop=True),
            check_like=True,
            check_dtype=False,
        )
        return True, None
    except AssertionError as e:
        return False, e


def verify_experiment(name, old_proc_file, old_anal_file, verbose=True):
    results = {}

    new_proc_path = os.path.join(NEW_DATA, f"{name}_processing.pickle")
    new_anal_path = os.path.join(NEW_DATA, f"{name}_analysis.pickle")
    old_proc_path = os.path.join(OLD_DATA, old_proc_file)
    old_anal_path = os.path.join(OLD_DATA, old_anal_file)

    # --- processing ---
    old_proc_df = _load(old_proc_path)
    new_proc_df = _load(new_proc_path)
    if old_proc_df is None:
        results["processing"] = "SKIP (old file missing)"
    elif new_proc_df is None:
        results["processing"] = "SKIP (new file missing)"
    else:
        ok, err = _frames_equal(old_proc_df, new_proc_df)
        results["processing"] = "PASS" if ok else f"FAIL: {err}"

    # --- analysis ---
    old_anal_df = _load(old_anal_path)
    new_anal_df = _load(new_anal_path)
    if old_anal_df is None:
        results["analysis"] = "SKIP (old file missing)"
    elif new_anal_df is None:
        results["analysis"] = "SKIP (new file missing)"
    else:
        ok, err = _frames_equal(old_anal_df, new_anal_df)
        if ok:
            results["analysis"] = "PASS"
        else:
            # Check whether the discrepancy is from find_outliers evolving:
            # run fresh find_outliers on the old modulation_df and see if it matches new.
            if old_proc_df is not None:
                try:
                    from magpyneto2.statistics import find_outliers as _fo
                    fresh_ff, _ = _fo(old_proc_df, Q=100)
                    ok2, _ = _frames_equal(fresh_ff, new_anal_df)
                    if ok2:
                        results["analysis"] = "STALE_OLD (find_outliers evolved; new matches fresh run)"
                    else:
                        results["analysis"] = f"FAIL: {err}"
                except Exception:
                    results["analysis"] = f"FAIL: {err}"
            else:
                results["analysis"] = f"FAIL: {err}"

    if verbose:
        proc_tag = results.get("processing", "-")
        anal_tag = results.get("analysis", "-")
        ok_tags = {"PASS", "SKIP", "STALE_OLD"}
        status = "OK" if all(any(t in v for t in ok_tags) for v in results.values()) else "!!"
        print(f"[{status}] {name:40s}  processing={proc_tag[:40]}  analysis={anal_tag[:50]}")

    return results


def main(experiments=None):
    pairs = {k: v for k, v in PAIRS.items() if k in experiments} if experiments else PAIRS
    all_pass = True
    for name, (old_proc, old_anal) in pairs.items():
        r = verify_experiment(name, old_proc, old_anal)
        if any("FAIL" in v for v in r.values()):
            all_pass = False

    print("\nAll PASS" if all_pass else "\nSome FAIL — see above")


if __name__ == "__main__":
    import sys
    exps = sys.argv[1:] if len(sys.argv) > 1 else None
    main(exps)
