"""Aggregate every experiment's {name}.nwb into data/manuscript/all_fourier_df.parquet.

Legacy *_analysis.pickle files are also picked up if any are still on disk
(pre-Phase-7-cutover leftovers) — see _read_nwb_fallback's docstring.

Usage:
    python pipeline/aggregate.py
    python pipeline/aggregate.py --out data/manuscript/all_fourier_df.parquet
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import schema

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
DEFAULT_OUT = os.path.join(DATA_DIR, "manuscript", "all_fourier_df.parquet")

# date/subject_id/species/area/contingency now live on each experiment's own YAML
# (ExperimentConfig.date/.subject_id/.species/.area/.contingency) instead of a
# hardcoded rec-name lookup here -- see _annotate_experiment_df below. Mouse and
# Owl are the sole exception: precomputed pickles in data/precomputed/ with no
# YAML at all (not reprocessable), so they keep a small residual annotation dict.
# Maps rec name → [date, ID, area, species, contingency]
_PRECOMPUTED_ANNOT_D = {
    # Mouse (KyuHyunLee) — Intan RHD .mat recordings, area=SC, loaded from data/precomputed/
    'magnetic4hz_170928_031259': ['20230928', 'mouse1', 'SC', 'mouse', "mag"],
    'magnetic_170928_025824':    ['20230928', 'mouse1', 'SC', 'mouse', "mag"],
    # Owl (Gutfreund) — pre-computed coefficients, loaded from data/precomputed/
    'exp1': ['20221210', 'Owl 7 cage C', 'pallium', 'Owl', "mag"],
    'exp2': ['20221210', 'Owl 7 cage C', 'pallium', 'Owl', "mag"],
    'exp3': ['20221210', 'Owl 7 cage C', 'pallium', 'Owl', "mag"],
    'exp4': ['20221210', 'Owl 7 cage C', 'pallium', 'Owl', "mag"],
    'exp5': ['20221210', 'Owl 7 cage C', 'pallium', 'Owl', "mag"],
}

# Legacy pickle-era column names -> current names. These frozen files
# (data/precomputed/mouse_analysis.pickle, owl_analysis.pickle, and any
# stale pre-Phase-7 *_analysis.pickle leftover) must never be edited on
# disk, but they still carry the OLD column names (they're ephys-style --
# engert/medaka never wrote a standalone pickle even before the NWB
# cutover -- so `nn` always maps to `spk_count` here, never `n_frames`).
_LEGACY_PICKLE_COL_RENAME = {"pp": "p_value", "nn": "spk_count", "rr": "NFC",
                              "2f_rr": "2f_NFC", "2f_pp": "2f_p_value"}


def _rename_legacy_pickle_columns(df):
    return df.rename(columns=_LEGACY_PICKLE_COL_RENAME)


def _read_nwb_analysis_df(nwb_path):
    """Reconstruct an analysis-pickle-shaped DataFrame directly from an NWB
    file -- the primary read path for every experiment as of the Phase 7
    cutover (no paradigm writes `_analysis.pickle` anymore). date/area/ID/
    species/contingency are per-cfg constants, not Fourier intermediates, so
    the shared Fourier-results tables never persist them (see
    analysis_stages/medaka.py) -- they come out missing here and are filled
    in by _annotate_experiment_df, called by build_all_fourier_df right after
    this, from the experiment's own YAML.
    """
    from pipeline import nwb_io
    io_r, nwbfile = nwb_io.read_nwbfile(nwb_path)
    try:
        return nwb_io.read_fourier_results_as_full_fourier_df(nwbfile)
    finally:
        io_r.close()


def _annotate_experiment_df(df, cfg):
    """Annotate one experiment's rows from its own YAML (ExperimentConfig)
    instead of a hardcoded rec-name lookup. date/ID/species/area are uniform
    across the whole experiment. contingency is per-row: any rec produced by
    one of cfg.auxiliary_stimuli (visual gratings/white noise/oddball/bars --
    always non-magnetic controls) is "positive control" regardless of
    cfg.contingency; matched by prefix too, since gratings/bars recs get an
    orientation/degree suffix appended (e.g. recname + "_45") that the
    aux_cfg's own `recname` doesn't include. Everything else (trials, or the
    experiment's own rec for 1:1 paradigms) falls back to cfg.contingency.
    """
    df = df.copy()
    df["date"] = cfg.date
    df["ID"] = cfg.subject_id
    df["species"] = cfg.species
    df["area"] = cfg.area

    contingency = pd.Series(cfg.contingency, index=df.index)
    for aux in cfg.auxiliary_stimuli:
        mask = (df["rec"] == aux.recname) | df["rec"].str.startswith(aux.recname + "_")
        contingency.loc[mask] = "positive control"
    df["contingency"] = contingency
    return df


def build_all_fourier_df(data_dir: str) -> pd.DataFrame:
    # NWB is the primary, authoritative source as of the Phase 7 cutover --
    # read from {name}.nwb whenever one exists, regardless of whether a
    # stale {name}_analysis.pickle also happens to still be sitting on disk
    # (no paradigm writes that pickle anymore, so any such file predates
    # the cutover and could silently be out of date; preferring it over the
    # freshly-reconstructed NWB frame would be a real correctness hazard,
    # not just redundant). The legacy pickle is read only as a last-resort
    # fallback for an experiment that has NO .nwb at all.
    nwb_paths = sorted(glob.glob(os.path.join(data_dir, "*.nwb")))
    nwb_names = {os.path.basename(p)[:-len(".nwb")] for p in nwb_paths}

    files = sorted(glob.glob(os.path.join(data_dir, "*_analysis.pickle")))
    pickle_only_files = [
        f for f in files
        if os.path.basename(f)[:-len("_analysis.pickle")] not in nwb_names
    ]

    if not files and not nwb_paths:
        raise FileNotFoundError(f"No *_analysis.pickle or *.nwb files found in {data_dir}")

    # Keyed by cfg.name, NOT by YAML filename -- a couple of YAML files
    # (e.g. experiments/Q_magnerNPX2_g0.yml) live under a filename that
    # differs from their own `name:` field (and hence from the {name}.nwb
    # they produce), so a filename-based guess would silently miss them.
    cfg_by_name = {cfg.name: cfg for cfg in schema.load_all_experiments(EXPERIMENTS_DIR)}

    dfs = []
    n_nwb = 0
    for nwb_path in nwb_paths:
        name = os.path.basename(nwb_path)[:-len(".nwb")]
        try:
            df = _read_nwb_analysis_df(nwb_path)
            cfg = cfg_by_name.get(name)
            if cfg is not None:
                df = _annotate_experiment_df(df, cfg)
            dfs.append(df)
            n_nwb += 1
        except Exception as exc:
            print(f"  skip {os.path.basename(nwb_path)} (.nwb): {exc}")
    print(f"Loaded {n_nwb} / {len(nwb_paths)} experiments from .nwb")

    for f in pickle_only_files:
        try:
            df = pd.read_pickle(f)
            if not isinstance(df, pd.DataFrame):
                print(f"  skip {os.path.basename(f)}: not a DataFrame")
                continue
            df = _rename_legacy_pickle_columns(df)
            dfs.append(df)
        except Exception as exc:
            print(f"  skip {os.path.basename(f)}: {exc}")
    if pickle_only_files:
        print(f"Loaded {len(pickle_only_files)} additional experiment(s) from legacy "
              f"_analysis.pickle (no .nwb found for these)")

    # Also load pre-computed species pickles (mouse, owl — formats not portable to YAML pipeline)
    precomputed_dir = os.path.join(data_dir, "precomputed")
    precomputed_files = sorted(glob.glob(os.path.join(precomputed_dir, "*.pickle")))
    n_pre = 0
    for f in precomputed_files:
        try:
            df = pd.read_pickle(f)
            if not isinstance(df, pd.DataFrame):
                print(f"  skip precomputed/{os.path.basename(f)}: not a DataFrame")
                continue
            df = _rename_legacy_pickle_columns(df)
            if "contingency" not in df.columns:
                # infer: owl control row gets "control" (will be dropped later); rest → "mag"
                df = df.copy()
                df["contingency"] = df["rec"].apply(
                    lambda r: "control" if str(r).lower() == "control" else "mag"
                )
            dfs.append(df)
            n_pre += 1
        except Exception as exc:
            print(f"  skip precomputed/{os.path.basename(f)}: {exc}")
    if n_pre:
        print(f"Loaded {n_pre} precomputed pickle(s) from {precomputed_dir}")

    all_df = pd.concat(dfs, ignore_index=True)

    # Fallback annotation for the precomputed species (mouse/owl) that have no
    # YAML and so weren't annotated per-experiment above by _annotate_experiment_df.
    annot_df = pd.DataFrame.from_dict(
        _PRECOMPUTED_ANNOT_D, orient="index",
        columns=["date", "ID", "area", "species", "contingency"]).rename_axis("rec")
    for col in ["date", "ID", "area", "species", "contingency"]:
        if col not in all_df.columns:
            all_df[col] = np.nan
        missing = all_df[col].isna() | (all_df[col] == "")
        all_df.loc[missing, col] = all_df.loc[missing, "rec"].map(annot_df[col])

    # Drop legacy Owl control row (no magnetic stimulus)
    all_df = all_df.loc[~((all_df.species == "Owl") & (all_df.rec == "control"))]

    unannotated = all_df.loc[all_df["species"].isna(), "rec"].unique()
    if len(unannotated):
        print(f"  {len(unannotated)} rec(s) have no annotation — set date/subject_id/"
              f"species/area on its experiments/*.yml, or extend _PRECOMPUTED_ANNOT_D "
              f"in pipeline/aggregate.py for a non-YAML-driven species:")
        for r in unannotated[:20]:
            print(f"    {r!r}")

    return all_df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Aggregate analysis pickles -> parquet")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"Output path (default: {DEFAULT_OUT})")
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help=f"Directory containing *_analysis.pickle files (default: {DATA_DIR})")
    args = parser.parse_args()

    print(f"Scanning {args.data_dir} ...")
    df = build_all_fourier_df(args.data_dir)
    print(f"Total rows: {len(df):,}   columns: {list(df.columns)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
