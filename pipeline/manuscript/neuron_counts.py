# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (magneto2)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neuron / animal / species counts for the manuscript
#
# Reproducible derivation of the headline numbers that get cited in text ("we recorded from N neurons across M animals and 7 species..."), so there's a single artifact to point to instead of re-deriving these by hand each time the question comes up.
#
# **Requires:** `data/manuscript/all_fourier_df.parquet` (run `python pipeline/aggregate.py` first if it's missing or stale).
#
# **Re-run this whenever `all_fourier_df.parquet` changes** (new experiments added, YAML thresholds retuned, etc.) so the numbers here never drift from what `aggregate.py` actually produced.
#
# ---
#
# ## Provenance: two rounds of counting-bug fixes (2026-08-25)
#
# While investigating a discrepancy between Fig2's magnetic-population N (7128) and Fig3's (7542), several real bugs/gaps were found in `magpyneto2.statistics.get_poscontrols_negresults` (and, for the first one, in `fig2.py`'s own local recomputation of the same dedup). All are fixed as of this notebook; the "before/after" section below quantifies the effect of each round rather than just asserting it.
#
# **Round 1 -- three bugs:**
#
# 1. **Unique-neuron dedup key was missing the subject `ID`.** `drop_duplicates(subset=["species", "date", "id"])` treats two *different* animals recorded on the same calendar date as the same neuron whenever their cluster/ROI `id` numbers happen to collide -- which they do for zebrafish `20221001` and `20221002` (each date has two different fish, `fish1`/`fish2`, and suite2p ROI numbering restarts per fish). Fixed by adding `"ID"` to the dedup key, matching what `fig3.py`'s occurrence-wave split already did correctly.
# 2. **Owl: only recording `exp5` was counted as a magnetic experiment**, silently dropping `exp1`-`exp4` (same owl, same 4Hz stimulus, same session family -- there was even a pre-existing code comment flagging this as questionable: `# What's going on here? isn't owl included above?`). Fixed to include all 5 owl recordings.
# 3. **`freq > 1` (strict) excluded every genuine 1Hz magnetic experiment.** This filter's real purpose is excluding oddball (~0.98Hz) and WN (0.8Hz) rows that aren't already caught by the substring checks (oddball's rec name contains none of "visual"/"WN"/"3D"), but a strict `>` also silently dropped pigeon's `mag1_upright` and zebra finch's `2nd site 1 Hz` / `mag1Hz` -- three real 1Hz magnetic recordings. Fixed to `freq >= 1`, which still excludes oddball/WN as intended.
#
# **Round 2 -- two more gaps, found while auditing what was left uncovered by *either* bucket after round 1:**
#
# 4. **Zebra finch `bars4_*` (session `20220916`, `visual_bars` paradigm, 8 orientations) was excluded from the positive-control population.** The filter only checked for the literal substrings `"visual"/"oddball"/"WN"/"3D"` in the rec name; this session's own YAML declares its visual condition as `visual_rec_substring: "bars"` (every other session's `visual_rec_substring` is already one of `"visual"`/`"oddball"`, both already covered). Fixed by adding `"bars"` to the substring check.
# 5. **Zebrafish `visualmagnet` recordings (4 recs: `20220221`, `20220223`, `20220301` a/b) had no visual-frequency fit at all**, and were excluded from *both* buckets even for their 0.4Hz magnetic-frequency row (blocked by a blanket `"visual" not in rec` check in the fish magnetic-trial filter, on top of the trial-name regex not recognizing "visualmagnet" as a magnet trial). Per direction: these should be fit at BOTH frequencies and counted in BOTH buckets. Fixed by (a) adding `analysis.visual_f: 0.016667` to all four YAMLs and re-running analysis, so each now has an independent 1/60Hz visual-frequency fit alongside its 0.4Hz magnetic one, and (b) widening the trial-name regex to recognize "visualmagnet"/"visualmagnet_a"/"visualmagnet_b" as magnetic trials and dropping the now-redundant (and, for this case, actively harmful) `"visual" not in rec` check -- the `freq > 0.02` filter already does the real work of keeping the two frequencies in their correct respective buckets.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display  # prints the table when run as a plain script

pd.set_option("future.no_silent_downcasting", True)

from magpyneto2 import statistics

# Repo-root-relative or notebook-cwd-relative, whichever exists -- lets this
# run both as a script from the repo root (`python pipeline/manuscript/neuron_counts.py`)
# and interactively as a notebook with cwd=pipeline/manuscript/
_candidates = [
    Path("data/manuscript/all_fourier_df.parquet"),
    Path("../../data/manuscript/all_fourier_df.parquet"),
]
PARQUET_PATH = next(p for p in _candidates if p.exists())
print(f"Loading {PARQUET_PATH.resolve()}")

all_fourier_df = pd.read_parquet(PARQUET_PATH)
print(f"{len(all_fourier_df)} total rows, {all_fourier_df.species.nunique()} species")

# %% [markdown]
# ## 1. Before / after: impact of each round of fixes
#
# `_get_poscontrols_negresults_V1` and `_get_poscontrols_negresults_V2` below are frozen, self-contained reproductions of the buggy/partially-fixed logic exactly as they shipped before each round above (not imported from anywhere -- the real function has already been fully fixed in `magpyneto2/statistics.py`). They exist here purely so this notebook can show its own receipts for each round's impact, without relying on git history.

# %%
import re as _re


def _get_poscontrols_negresults_V1(all_fourier_df):
    """Frozen reproduction of get_poscontrols_negresults as it shipped before
    ANY of the 2026-08-25 fixes (round 1 or round 2). For before/after
    comparison only -- not used by the live pipeline.
    """
    nn_filter = ((all_fourier_df.spk_count.values > 50) | (all_fourier_df.species.values == "Owl"))
    include_fish = np.array([("fish" in elem) or (elem == "medaka") for elem in all_fourier_df.species.values])

    not_fish_pos_control = all_fourier_df.loc[
        nn_filter & ~include_fish & (
            np.array(["visual" in elem for elem in all_fourier_df.rec.values])
            | np.array(["oddball" in elem for elem in all_fourier_df.rec.values])
            | np.array(["WN" in elem for elem in all_fourier_df.rec.values])
            | np.array(["3D" in elem for elem in all_fourier_df.rec.values])
            # BUG: no "bars" check -- misses session 20220916's visual_bars paradigm
        ), :]
    medaka_trial0_no_visual = np.array([rec.endswith("magneto_0.tif") for rec in all_fourier_df.rec.values])
    fish_pos_control = all_fourier_df.loc[include_fish
        & np.array(["nostim" not in rec for rec in all_fourier_df.rec.values])
        & (all_fourier_df.freq.values < 0.02)
        & ~medaka_trial0_no_visual, :]
    pos_control = pd.concat([fish_pos_control, not_fish_pos_control])

    not_fish_mag_exp = all_fourier_df.loc[nn_filter & ~include_fish
        & (all_fourier_df.freq > 1)  # BUG: strict, drops real 1Hz experiments
        & np.array(["visual" not in elem for elem in all_fourier_df.rec.values])
        & np.array(["WN" not in elem for elem in all_fourier_df.rec.values])
        & np.array(["3D" not in elem for elem in all_fourier_df.rec.values]), :]
    owl_mag_exp = all_fourier_df.loc[  # BUG: only exp5, drops exp1-exp4
        (all_fourier_df.species == "Owl") & (all_fourier_df.rec == "exp5"), :]
    _magnet_trial_re_v1 = _re.compile(r"(^|_)(magnet|magneto_\d+)(\.tif)?$")
    fish_mag_exp = all_fourier_df.loc[
        include_fish
        & np.array(["nostim" not in rec for rec in all_fourier_df.rec.values])
        & np.array(["no_magnet" not in rec and "no-magnet" not in rec for rec in all_fourier_df.rec.values])
        & np.array(["visual" not in elem for elem in all_fourier_df.rec.values])  # BUG: also blocks "visualmagnet"
        & (all_fourier_df.freq.values > 0.02)
        & np.array([bool(_magnet_trial_re_v1.search(elem)) for elem in all_fourier_df.rec.values]), :]
    mag_exp = pd.concat([fish_mag_exp, not_fish_mag_exp, owl_mag_exp])

    return mag_exp, pos_control


def _get_poscontrols_negresults_V2(all_fourier_df):
    """Frozen reproduction after round 1 (dedup key, owl exp1-4, freq>=1) but
    before round 2 (bars substring, visualmagnet dual-fit/regex). For
    before/after comparison only -- not used by the live pipeline.
    """
    nn_filter = ((all_fourier_df.spk_count.values > 50) | (all_fourier_df.species.values == "Owl"))
    include_fish = np.array([("fish" in elem) or (elem == "medaka") for elem in all_fourier_df.species.values])

    not_fish_pos_control = all_fourier_df.loc[
        nn_filter & ~include_fish & (
            np.array(["visual" in elem for elem in all_fourier_df.rec.values])
            | np.array(["oddball" in elem for elem in all_fourier_df.rec.values])
            | np.array(["WN" in elem for elem in all_fourier_df.rec.values])
            | np.array(["3D" in elem for elem in all_fourier_df.rec.values])
            # still no "bars" check at this point
        ), :]
    medaka_trial0_no_visual = np.array([rec.endswith("magneto_0.tif") for rec in all_fourier_df.rec.values])
    fish_pos_control = all_fourier_df.loc[include_fish
        & np.array(["nostim" not in rec for rec in all_fourier_df.rec.values])
        & (all_fourier_df.freq.values < 0.02)
        & ~medaka_trial0_no_visual, :]
    pos_control = pd.concat([fish_pos_control, not_fish_pos_control])

    not_fish_mag_exp = all_fourier_df.loc[nn_filter & ~include_fish
        & (all_fourier_df.freq >= 1)  # round 1 fix
        & np.array(["visual" not in elem for elem in all_fourier_df.rec.values])
        & np.array(["WN" not in elem for elem in all_fourier_df.rec.values])
        & np.array(["3D" not in elem for elem in all_fourier_df.rec.values]), :]
    owl_mag_exp = all_fourier_df.loc[all_fourier_df.species == "Owl", :]  # round 1 fix: all 5 exps
    _magnet_trial_re_v1 = _re.compile(r"(^|_)(magnet|magneto_\d+)(\.tif)?$")  # still narrow at this point
    fish_mag_exp = all_fourier_df.loc[
        include_fish
        & np.array(["nostim" not in rec for rec in all_fourier_df.rec.values])
        & np.array(["no_magnet" not in rec and "no-magnet" not in rec for rec in all_fourier_df.rec.values])
        & np.array(["visual" not in elem for elem in all_fourier_df.rec.values])  # still blocks "visualmagnet"
        & (all_fourier_df.freq.values > 0.02)
        & np.array([bool(_magnet_trial_re_v1.search(elem)) for elem in all_fourier_df.rec.values]), :]
    mag_exp = pd.concat([fish_mag_exp, not_fish_mag_exp, owl_mag_exp])

    return mag_exp, pos_control


v1_neg_res, v1_pos_control = _get_poscontrols_negresults_V1(all_fourier_df)
v1_neg_unique = v1_neg_res.drop_duplicates(subset=["species", "date", "id"], keep="first")  # BUG: no ID
v1_pos_unique = v1_pos_control.drop_duplicates(subset=["species", "date", "id"], keep="first")  # BUG: no ID

v2_neg_res, v2_pos_control = _get_poscontrols_negresults_V2(all_fourier_df)
v2_neg_unique = v2_neg_res.drop_duplicates(subset=["species", "ID", "date", "id"], keep="first")
v2_pos_unique = v2_pos_control.drop_duplicates(subset=["species", "ID", "date", "id"], keep="first")

post_fix_neg_res, post_fix_pos_control, post_fix_unique_pos_control = statistics.get_poscontrols_negresults(all_fourier_df)
post_fix_unique = post_fix_neg_res.drop_duplicates(subset=["species", "ID", "date", "id"], keep="first")

neg_before_after = pd.DataFrame({
    "V1: original (buggy)": v1_neg_unique.groupby("species").size(),
    "V2: after round 1": v2_neg_unique.groupby("species").size(),
    "V3: after round 2 (current)": post_fix_unique.groupby("species").size(),
}).fillna(0).astype(int)
neg_before_after["round 1 delta"] = neg_before_after["V2: after round 1"] - neg_before_after["V1: original (buggy)"]
neg_before_after["round 2 delta"] = neg_before_after["V3: after round 2 (current)"] - neg_before_after["V2: after round 1"]
neg_before_after.loc["TOTAL"] = neg_before_after.sum()
print("Magnetic (negative-result) population, by fix round:")
display(neg_before_after)

pos_before_after = pd.DataFrame({
    "V1: original (buggy)": v1_pos_unique.groupby("species").size(),
    "V2: after round 1": v2_pos_unique.groupby("species").size(),
    "V3: after round 2 (current)": post_fix_unique_pos_control.groupby("species").size(),
}).fillna(0).astype(int)
pos_before_after["round 1 delta"] = pos_before_after["V2: after round 1"] - pos_before_after["V1: original (buggy)"]
pos_before_after["round 2 delta"] = pos_before_after["V3: after round 2 (current)"] - pos_before_after["V2: after round 1"]
pos_before_after.loc["TOTAL"] = pos_before_after.sum()
print("\nPositive-control population, by fix round:")
display(pos_before_after)

# %% [markdown]
# ## 2. Unique neurons across all magnetic experiments
#
# This is the number to cite for "we recorded N neurons across all magnetic-stimulus experiments" -- deduplicated by `(species, ID, date, id)` so a neuron seen across repeat magnetic trials within one session is counted once, and two different animals recorded on the same date are never conflated.

# %%
unique_neg_res = post_fix_neg_res.drop_duplicates(subset=["species", "ID", "date", "id"], keep="first")

neurons_by_species = unique_neg_res.groupby("species").size().rename("n_unique_neurons").to_frame()
neurons_by_species.loc["TOTAL"] = neurons_by_species.sum()
print(f"Total unique neurons, all magnetic experiments, all species: {len(unique_neg_res)}")
display(neurons_by_species)

# %% [markdown]
# ## 3. Unique neurons across all positive-control experiments
#
# Same dedup, for the visual/auditory/oddball positive-control population (Fig2 panel D / Fig3's positive-control rows). Owl and mouse have no positive-control recordings in this dataset (Owl is precomputed-Bayesian-coefficients only; mouse likewise has no matching visual/auditory block), so they're absent here, not zero-filled by mistake.

# %%
unique_pos_control = post_fix_unique_pos_control

poscontrol_by_species = unique_pos_control.groupby("species").size().rename("n_unique_neurons").to_frame()
poscontrol_by_species.loc["TOTAL"] = poscontrol_by_species.sum()
print(f"Total unique neurons, all positive-control experiments, all species: {len(unique_pos_control)}")
display(poscontrol_by_species)

# %% [markdown]
# ## 4. Species and individual animals
#
# Individual-animal counts come from the `ID` (subject) column of the FULL dataset (not restricted to the magnetic or positive-control subsets above) -- an animal counts here if it appears anywhere in the paper, regardless of which stimulus conditions survived the significance-testing filters.
#
# One genuine data gap: three early zebrafish sessions (`20220221`, `20220223`, `20220301`) have `subject_id: "unknown"` in their own source YAMLs -- that identity was never recorded at acquisition time, not lost by this pipeline. Per lab convention (one fish per multi-hour 2-photon session, never reused across days), each of these three sessions is presumed to be a *different* individual fish; the table below reports both the literal `ID`-based count and this best estimate, so the assumption stays visible rather than silently baked in.

# %%
n_species = all_fourier_df.species.nunique()
print(f"Species: {n_species}\n")

rows = []
for species, g in all_fourier_df.groupby("species"):
    ids = sorted(g.ID.astype(str).unique())
    unknown_dates = sorted(g.loc[g.ID.astype(str) == "unknown", "date"].unique())
    n_recorded = len(ids)
    # Each "unknown"-ID date is presumed a distinct individual (see markdown
    # above) and isn't itself counted in `ids` as a real ID, so add it back.
    n_best_estimate = n_recorded + (len(unknown_dates) if "unknown" in ids else 0) - (1 if "unknown" in ids else 0)
    rows.append({
        "species": species,
        "n_individuals (recorded)": n_recorded,
        "n_individuals (best estimate)": n_best_estimate,
        "IDs": ", ".join(i for i in ids if i != "unknown"),
        "unresolved sessions (presumed separate individuals)": ", ".join(unknown_dates) if unknown_dates else "",
    })

animals_df = pd.DataFrame(rows).set_index("species")
totals = pd.Series({
    "n_individuals (recorded)": animals_df["n_individuals (recorded)"].sum(),
    "n_individuals (best estimate)": animals_df["n_individuals (best estimate)"].sum(),
    "IDs": "",
    "unresolved sessions (presumed separate individuals)": "",
}, name="TOTAL")
animals_df = pd.concat([animals_df, totals.to_frame().T])
display(animals_df)

# %% [markdown]
# ## 5. Summary table (for citation in the paper)

# %%
summary = pd.DataFrame({
    "n_individuals_recorded": animals_df["n_individuals (recorded)"].drop("TOTAL"),
    "n_individuals_best_estimate": animals_df["n_individuals (best estimate)"].drop("TOTAL"),
    "n_unique_neurons_magnetic": neurons_by_species["n_unique_neurons"].drop("TOTAL", errors="ignore"),
    "n_unique_neurons_poscontrol": poscontrol_by_species["n_unique_neurons"].drop("TOTAL", errors="ignore"),
}).fillna(0).infer_objects(copy=False)
summary[["n_individuals_recorded", "n_individuals_best_estimate", "n_unique_neurons_magnetic", "n_unique_neurons_poscontrol"]] = \
    summary[["n_individuals_recorded", "n_individuals_best_estimate", "n_unique_neurons_magnetic", "n_unique_neurons_poscontrol"]].astype(int)
summary.loc["TOTAL"] = summary.sum()

print(f"{n_species} species, "
      f"{summary.loc['TOTAL', 'n_individuals_recorded']} individuals recorded "
      f"({summary.loc['TOTAL', 'n_individuals_best_estimate']} best estimate), "
      f"{summary.loc['TOTAL', 'n_unique_neurons_magnetic']} unique neurons across all magnetic experiments, "
      f"{summary.loc['TOTAL', 'n_unique_neurons_poscontrol']} unique neurons across all positive-control experiments.")
display(summary)

# %%
out_path = PARQUET_PATH.parent / "neuron_counts_summary.csv"
summary.to_csv(out_path)
print(f"Saved {out_path.resolve()}")

# %% [markdown]
# ## 6. Export to the manuscript
#
# Writes every number above into the Overleaf clone's `numbers.tex` (see `tex_numbers.py`), so `main.tex` cites e.g. `\importnumber{n_neurons_magnetic}` and picks up new values whenever this notebook is re-run -- no hand-copying. Per-species keys use the species name lowercased with non-alphanumerics replaced by `_` (e.g. `n_neurons_magnetic_zebra_finch`).

# %%
import re
import sys

sys.path.insert(0, str(PARQUET_PATH.resolve().parents[2] / "pipeline" / "manuscript"))
import tex_numbers


def _slug(species):
    return re.sub(r"[^a-z0-9]+", "_", species.lower()).strip("_")


numbers = {
    "n_species": n_species,
    "n_individuals": summary.loc["TOTAL", "n_individuals_recorded"],
    "n_individuals_best_estimate": summary.loc["TOTAL", "n_individuals_best_estimate"],
    "n_neurons_magnetic": summary.loc["TOTAL", "n_unique_neurons_magnetic"],
    "n_neurons_poscontrol": summary.loc["TOTAL", "n_unique_neurons_poscontrol"],
}
for species, row in summary.drop("TOTAL").iterrows():
    s = _slug(species)
    numbers[f"n_individuals_{s}"] = row["n_individuals_recorded"]
    numbers[f"n_individuals_best_estimate_{s}"] = row["n_individuals_best_estimate"]
    numbers[f"n_neurons_magnetic_{s}"] = row["n_unique_neurons_magnetic"]
    numbers[f"n_neurons_poscontrol_{s}"] = row["n_unique_neurons_poscontrol"]

tex_numbers.write_numbers("neuron_counts", numbers)
display(pd.Series(numbers, name="value").to_frame())
