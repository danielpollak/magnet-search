"""Metadata tables: one row per recording (GCaMP) or per recording x stimulus frequency (ephys).

    python pipeline/metadata_tables.py     # run after aggregate.py

Reads experiments/*.yml, data/{name}.nwb and data/manuscript/all_fourier_df.parquet, and writes

    data/metadata/gcamp_recordings.csv   + gcamp_recordings_columns.csv
    data/metadata/ephys_recordings.csv   + ephys_recordings_columns.csv
    data/metadata/recordings.xlsx        (both tables + column definitions; needs openpyxl)

The columns files define every column. "Fig 2 panel" uses the same split as the figures
(statistics.get_poscontrols_negresults).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from magpyneto2 import statistics  # noqa: E402
from pipeline import nwb_io  # noqa: E402

OUT = os.path.join(REPO, "data", "metadata")
PARQUET = os.path.join(REPO, "data", "manuscript", "all_fourier_df.parquet")
IMAGING = ("engert", "medaka")


def _yamls():
    for p in sorted(glob.glob(os.path.join(REPO, "experiments", "*.yml"))):
        with open(p, encoding="utf-8") as f:
            yield yaml.safe_load(f)


def _panel_lookup(df):
    """(rec, rounded freq) -> which Fig 2 population the rows belong to."""
    mag, pos, _ = statistics.get_poscontrols_negresults(df)
    out = {}
    for frame, label in ((mag, "2A/2C magnetic"), (pos, "2B/2D visual & auditory")):
        for rec, f in frame[["rec", "freq"]].drop_duplicates().itertuples(index=False):
            out[(rec, round(f, 6))] = label
    return lambda rec, f: out.get((rec, round(f, 6)), "not in Fig 2")


def _suspects(rows, col="p_value"):
    if not len(rows) or rows[col].isna().all():
        return ""
    return int((rows[col] < 0.01).sum())


def gcamp_table(df, panel):
    rows = []
    for y in _yamls():
        if y.get("paradigm") not in IMAGING:
            continue
        name, a = y["name"], y.get("analysis", {})
        T, f = float(y.get("sample_period", 1.0)), float(a["f"])
        medaka = y["paradigm"] == "medaka"
        vf = 1 / 60 if medaka else float(a.get("visual_f", -1))
        rec = os.path.basename(y["session_path"]) + ".tif" if medaka else name
        r = df[df.rec == rec]
        rm = r[np.isclose(r.freq, f)]
        rv = r[np.isclose(r.freq, vf)] if vf > 0 else r.iloc[:0]
        io, nwb = nwb_io.read_nwbfile(os.path.join(REPO, "data", f"{name}.nwb"))
        F, roi = nwb_io.read_roi_data(nwb)
        n_frames = F.shape[1]
        io.close()
        N = min(120 * (n_frames // 60), n_frames)   # fit_Fourier's analysed length
        res, fs = 1 / (N * T), 1 / T
        low = name.lower()
        if any(d in name for d in ("20220221", "20220223", "20220301")):
            batch, idx, stitched = "2022 Q1", "", "no"
            cond = ("visual+magnet" if "visualmagnet" in low else "visual" if "visual" in low
                    else "magnet" if "magnet" in low else "nostim")
            coil = "on" if "magnet" in low else "off"
            grating = "yes" if "visual" in low else "no"
        else:
            batch = "medaka" if medaka else f"{f:g} Hz zebrafish"
            idx = name.rsplit("_", 1)[-1]
            coil = "off" if "no_magneto" in low else "on"
            cond = f"{'no_magneto' if coil == 'off' else 'magneto'}_{idx}"
            grating = "yes" if idx in ("1", "2") and "20221002_fish1" not in name else "no"
            stitched = "yes"
        rows.append({
            "recording": name, "species": y["species"], "area": y.get("area", ""),
            "batch": batch, "date": y["date"], "subject": y.get("subject_id", ""),
            "condition": cond, "condition index": idx, "coil": coil, "visual grating": grating,
            "stitched from interleaved 60 s chunks": stitched,
            "Fig 2 panel (primary f)": panel(rec, f),
            "Fig 2 panel (visual f)": panel(rec, vf) if vf > 0 else "",
            "frames": n_frames, "frame period T (s)": T, "sampling rate (Hz)": round(fs, 5),
            "duration (min)": round(n_frames * T / 60, 2), "frames analysed N": N,
            "frequency resolution (Hz)": round(res, 6), "Nyquist (Hz)": round(fs / 2, 4),
            "primary f (Hz)": round(f, 6), "primary f is": "visual" if f < 0.02 else "magnetic",
            "primary f in bins": round(f * N * T, 3), "Q_frac": a.get("Q_frac"),
            "half-window (Hz)": round(a.get("Q_frac") * f, 5),
            "Q (bins per side)": int(rm.Q.iloc[0]) if len(rm) else "",
            "2F (Hz)": round(2 * f, 6),
            "Q_2f (bins per side)": (int(rm.Q_2f.iloc[0]) if len(rm) and pd.notna(rm.Q_2f.iloc[0])
                                     else ""),
            "visual f (Hz)": round(vf, 6) if vf > 0 else "",
            "visual f in bins": round(vf * N * T, 3) if vf > 0 else "",
            "visual Q_frac": (0.5 if medaka else a.get("visual_Q_frac", 0.5)) if vf > 0 else "",
            "visual Q (bins per side)": int(rv.Q.iloc[0]) if len(rv) else "",
            "ROIs segmented": len(roi), "iscell_threshold": y.get("iscell_threshold"),
            "npix_threshold (strict >)": y.get("npix_threshold"), "ROIs analysed": len(rm),
            "1F suspects": _suspects(rm), "2F suspects": _suspects(rm, "2f_p_value"),
            "visual suspects": _suspects(rv),
            "tiff": y.get("tiff_name", ""), "NAS session folder": y["session_path"],
            "suite2p plane0 (local)": y["suite2p_path"],
        })
    return pd.DataFrame(rows)


GCAMP_COLUMNS = [
    ("recording", "experiment name: experiments/<name>.yml, data/<name>.nwb"),
    ("batch", "2022 Q1: a separate 20 or 50 min recording per condition. 0.3/0.1 Hz zebrafish "
              "and medaka: one run of 61 interleaved 60 s chunks split into conditions"),
    ("condition", "2022 Q1: magnet / visual / visual+magnet / nostim. Others: magneto_k or "
                  "no_magneto_k, k = the condition index in the raw chunk names"),
    ("condition index", "k above: a stimulus condition, not a repeat number (1 and 2 carry the "
                        "grating)"),
    ("coil", "magnetic stimulus on or off, by recording name"),
    ("visual grating", "60 s grating present (2022 Q1 by condition; others from population "
                       "spectra, docs/full_resolution_rerun.md section 3)"),
    ("stitched from interleaved 60 s chunks", "the trial tiff concatenates this condition's "
                                              "chunks, so it has a seam every 60 frames"),
    ("Fig 2 panel", "population the rows fall in (statistics.get_poscontrols_negresults)"),
    ("frame period T (s)", "sample_period: 1.0212 measured for 2022 Q1 (docs/frame_timing.md); "
                           "1.0 elsewhere (not measured)"),
    ("frames analysed N", "fit_Fourier analyses min(120*(frames//60), frames) frames"),
    ("frequency resolution (Hz)", "FFT bin spacing, 1/(N*T)"),
    ("primary f (Hz)", "analysis.f: the magnetic frequency, or 1/60 Hz for 2022 Q1 visual-only"),
    ("primary f in bins", "f*N*T; the analysis uses the nearest bin, so a non-integer is off-bin"),
    ("Q_frac", "half-width of the reference window as a fraction of f"),
    ("half-window (Hz)", "Q_frac * f"),
    ("Q (bins per side)", "off-frequency bins on each side of f that normalise NFC (2Q in all)"),
    ("Q_2f (bins per side)", "same for the second harmonic (engert, below Nyquist only)"),
    ("visual f / visual Q", "independent 1/60 Hz fit (visual positive control)"),
    ("ROIs segmented", "all suite2p ROIs in the segmentation (shared by a session's trials)"),
    ("ROIs analysed", "after P(iscell) > 0.5, npix > 9, fish outline and flatline removal"),
    ("suspects", "ROIs with p < 0.01 at that frequency"),
]


def ephys_table(df, panel):
    rows = []
    for y in _yamls():
        if y.get("paradigm") in IMAGING + ("manual",):
            continue
        name = y["name"]
        path = os.path.join(REPO, "data", f"{name}.nwb")
        if not os.path.exists(path):
            continue
        io, nwb = nwb_io.read_nwbfile(path)
        groups, _ = nwb_io.read_fourier_group_and_unit_tables(nwb)
        ep = nwb.intervals["stimulus_epochs"].to_dataframe() if "stimulus_epochs" in nwb.intervals \
            else None
        n_units = len(nwb.units) if nwb.units is not None else np.nan
        io.close()
        g1 = groups[groups.harmonic == "1F"]
        for _, g in g1.iterrows():
            rec, f, T = g["rec"], float(g["frequency"]), float(g["T"])
            g2 = groups[(groups.rec == rec) & (groups.harmonic == "2F")
                        & np.isclose(groups.frequency, 2 * f)]
            r = df[(df.rec == rec) & np.isclose(df.freq, f)]
            e = ep[ep.rec == rec] if ep is not None else None
            first = r.iloc[0] if len(r) else None
            rows.append({
                "experiment": name, "recording": rec,
                "species": first.species if first is not None else "",
                "area": first.area if first is not None else y.get("area", ""),
                "date": first.date if first is not None else "",
                "subject": first.ID if first is not None else "",
                "paradigm": y["paradigm"],
                "stimulus type": ", ".join(sorted(e.stim_type.astype(str).unique()))
                if e is not None and len(e) else "",
                "stimulus windows": len(e) if e is not None else "",
                "Fig 2 panel": panel(rec, f),
                "stimulus f (Hz)": round(f, 6), "analysed duration T (s)": round(T, 3),
                "cycles of f": round(f * T, 2), "frequency resolution 1/T (Hz)": round(1 / T, 6),
                "Q_frac (effective, Q/(f*T))": round(int(g["Q"]) / (f * T), 4),
                "Q (bins per side)": int(g["Q"]),
                "2F (Hz)": round(2 * f, 6),
                "Q_2f (bins per side)": int(g2.Q.iloc[0]) if len(g2) else "",
                "units in NWB": n_units,
                "units analysed": len(r),
                "units with > 50 spikes": int((r.spk_count > 50).sum()) if len(r) else "",
                "1F suspects": _suspects(r), "2F suspects": _suspects(r, "2f_p_value"),
                "source": "pipeline (data/<experiment>.nwb)",
            })
    # mouse and owl: precomputed, parquet rows only
    for (sp, rec, f), r in df[df.species.isin(["mouse", "Owl"])].groupby(
            ["species", "rec", "freq"]):
        rows.append({
            "experiment": "", "recording": rec, "species": sp, "area": r.area.iloc[0],
            "date": r.date.iloc[0], "subject": r.ID.iloc[0], "paradigm": "precomputed",
            "Fig 2 panel": panel(rec, f), "stimulus f (Hz)": round(f, 6),
            "Q (bins per side)": int(r.Q.iloc[0]) if "Q" in r and pd.notna(r.Q.iloc[0]) else "",
            "units analysed": len(r), "units with > 50 spikes": int((r.spk_count > 50).sum()),
            "1F suspects": _suspects(r), "2F suspects": _suspects(r, "2f_p_value"),
            "source": "data/precomputed/*_analysis.pickle (not re-processable)",
        })
    return pd.DataFrame(rows)


EPHYS_COLUMNS = [
    ("experiment", "experiments/<experiment>.yml, data/<experiment>.nwb (blank: precomputed)"),
    ("recording", "rec: one stimulus block within the experiment"),
    ("paradigm", "pipeline paradigm (see CLAUDE.md), or precomputed for mouse and owl"),
    ("stimulus type", "stim_type of the recording's stimulus_epochs rows"),
    ("stimulus windows", "rows in stimulus_epochs for this recording (trials or blocks)"),
    ("Fig 2 panel", "population the rows fall in (statistics.get_poscontrols_negresults)"),
    ("analysed duration T (s)", "length of the (stitched) spike train the Fourier fit uses"),
    ("frequency resolution 1/T (Hz)", "bin spacing of the spike-train spectrum"),
    ("Q_frac (effective, Q/(f*T))", "reference half-window as a fraction of f, back-computed "
                                    "from Q (the YAML sets Q_frac; Q is rounded)"),
    ("Q (bins per side)", "off-frequency bins on each side of f that normalise NFC"),
    ("units in NWB", "Kilosort clusters written by processing (only 'good' ones when the YAML "
                     "says good: true), for the whole experiment"),
    ("units analysed", "units with a row in the Fourier results (Kilosort 'good' when the YAML "
                       "says good: true)"),
    ("units with > 50 spikes", "the spike-count filter Fig 2 applies (owl exempt)"),
    ("suspects", "units with p < 0.01 at that frequency (before the spike-count filter)"),
]


def main():
    df = pd.read_parquet(PARQUET)
    panel = _panel_lookup(df)
    os.makedirs(OUT, exist_ok=True)
    tables = {
        "gcamp_recordings": (gcamp_table(df, panel), GCAMP_COLUMNS),
        "ephys_recordings": (ephys_table(df, panel), EPHYS_COLUMNS),
    }
    for stem, (t, cols) in tables.items():
        t.to_csv(os.path.join(OUT, f"{stem}.csv"), index=False)
        pd.DataFrame(cols, columns=["column", "meaning"]).to_csv(
            os.path.join(OUT, f"{stem}_columns.csv"), index=False)
        print(f"{stem}: {len(t)} rows -> {OUT}")
    try:
        with pd.ExcelWriter(os.path.join(OUT, "recordings.xlsx")) as w:
            for stem, (t, cols) in tables.items():
                t.to_excel(w, sheet_name=stem, index=False)
                pd.DataFrame(cols, columns=["column", "meaning"]).to_excel(
                    w, sheet_name=f"{stem[:5]}_columns", index=False)
                w.sheets[stem].freeze_panes = "B2"
    except ImportError:
        print("openpyxl not installed: wrote the CSVs only")


if __name__ == "__main__":
    main()
