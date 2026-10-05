"""Marginal distribution of CV^2 for every ephys unit and imaging ROI in the Fig 2 magnetic pool.

    python docs/cv2/cv2_marginals.py

CV^2 here is computed the same way for both modalities, from exactly the bins the test uses:
the 2M off-frequency coefficients at the analysis frequency (1F) persisted in each NWB file's
per_unit_fourier_results (fou_alt_real/imag). CV^2 = var(|c|^2) / mean(|c|^2)^2 over those 2M
powers. Gaussian noise gives exponential powers, CV^2 = 1.

With only 2M powers the sample CV^2 is noisy and biased low, so each unit is paired with a
reference draw: the sample CV^2 of 2M independent exponentials, with the same M. The reference
ECDF is what CV^2 would look like if every unit obeyed the null exactly.

For ephys a unit's coefficient is a sum of one unit arrow per spike, so the K-event result
CV^2 = 1 - 1/K applies with K = the spike count (ignoring bursts).

The population is the Fig 2 magnetic pool (statistics.get_poscontrols_negresults), joined on
(rec, freq, id). Mouse and owl have no NWB file (precomputed) and are left out.

Outputs (next to this script): results_cv2_marginals.csv (gitignored), fig_cv2_marginals.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE.parent / "nfc_finite_sample_bias"))
from pipeline import nwb_io  # noqa: E402
from magpyneto2 import statistics as st  # noqa: E402
from slow_variation import C_INK, C_MUTED, _style  # noqa: E402

PARQUET = _REPO / "data" / "manuscript" / "all_fourier_df.parquet"
OUT_CSV = _HERE / "results_cv2_marginals.csv"
DUPLICATE_RECS = {"engert_20221002_fish1_magneto_2", "engert_20221002_fish1_magneto_3"}
GROUPS = [  # (label, modality, colour)
    ("zebra finch", "ephys", "#0d366b"),
    ("Pigeon", "ephys", "#2a78d6"),
    ("Quail", "ephys", "#86b6ef"),
    ("zebrafish 0.4 Hz (2022 Q1)", "imaging", "#8a8a8a"),
    ("zebrafish 0.3 Hz", "imaging", "#eb6834"),
    ("zebrafish 0.1 Hz", "imaging", "#b8431a"),
    ("medaka 0.1 Hz", "imaging", "#6b2a0d"),
]


def cv2_rows(nwb_path, rng):
    io, nwbfile = nwb_io.read_nwbfile(str(nwb_path))
    try:
        if "analysis" not in nwbfile.processing:
            return []
        module = nwbfile.processing["analysis"]
        g = module["fourier_group_results"].to_dataframe()
        u = module["per_unit_fourier_results"].to_dataframe()
    finally:
        io.close()
    out = []
    for gi, row in g[g["harmonic"] == "1F"].iterrows():
        rows = u[u["group_1f_index"] == gi]
        if not len(rows):
            continue
        c = np.stack(rows["fou_alt_real"].values) + 1j * np.stack(rows["fou_alt_imag"].values)
        P = np.abs(c) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            cv2 = P.var(1) / P.mean(1) ** 2
        E = rng.exponential(size=P.shape)
        ref = E.var(1) / E.mean(1) ** 2
        d = pd.DataFrame(dict(rec=row["rec"], freq=round(float(row["frequency"]), 4),
                              id=rows["unit_id"].values, M=int(row["Q"]), cv2=cv2, cv2_ref=ref))
        if "spk_count" in rows:
            d["spk_count"] = rows["spk_count"].values
        out.append(d)
    return out


def group_of(r):
    if r["species"] == "zebrafish":
        return {0.4: GROUPS[3][0], 0.3: GROUPS[4][0], 0.1: GROUPS[5][0]}.get(round(r["freq"], 1))
    if r["species"] == "medaka":
        return GROUPS[6][0]
    return r["species"]


def compute():
    rng = np.random.default_rng(0)
    parts = []
    for p in sorted((_REPO / "data").glob("*.nwb")):
        parts += cv2_rows(p, rng)
        print(f"  {p.name}", flush=True)
    cv = pd.concat(parts, ignore_index=True)
    df = pd.read_parquet(PARQUET)
    neg, _, _ = st.get_poscontrols_negresults(df)
    neg = neg[~neg["rec"].isin(DUPLICATE_RECS)].copy()
    neg["freq"] = neg["freq"].astype(float).round(4)
    m = neg[["species", "rec", "freq", "id", "p_value"]].merge(cv, on=["rec", "freq", "id"], how="inner")
    m["group"] = m.apply(group_of, axis=1)
    m = m[m["group"].notna()]
    print(f"  joined {len(m)} of {len(neg)} magnetic-pool rows "
          f"(mouse/owl have no NWB file)")
    m.to_csv(OUT_CSV, index=False)
    return m


def figure(m, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, modality in zip(axes[:2], ["ephys", "imaging"]):
        for label, mod, c in GROUPS:
            if mod != modality:
                continue
            d = m[m["group"] == label]
            if not len(d):
                continue
            x = np.sort(d["cv2"].dropna().values)
            r = np.sort(d["cv2_ref"].values)
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=2,
                    label=f"{label}: n = {len(x)}, median {np.median(x):.2f}")
            ax.plot(r, np.arange(1, len(r) + 1) / len(r), color=c, lw=1, ls=":")
        ax.axvline(1, color=C_MUTED, lw=0.6)
        ax.set_xlim(0, 2.5)
        ax.set_xlabel("CV$^2$ of the 2M noise-bin powers at the analysis frequency", fontsize=8)
        ax.set_ylabel("ECDF over units", fontsize=8)
        ax.set_title(f"{'AB'[modality == 'imaging']}. {modality} (dotted: the same M, exact "
                     "null)", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        _style(ax)
    ax = axes[2]
    e = m[m["group"].isin([g for g, mod, _ in GROUPS if mod == "ephys"])].dropna(subset=["spk_count"])
    e = e[e["spk_count"] > 0]
    bins = np.unique(np.round(np.logspace(0, np.log10(e["spk_count"].max() + 1), 25)))
    e = e.assign(b=pd.cut(e["spk_count"], bins, include_lowest=True))
    s = e.groupby("b", observed=True).agg(n=("spk_count", "median"), cv=("cv2", "median"),
                                           ref=("cv2_ref", "median"), k=("cv2", "size"))
    s = s[s["k"] >= 20]
    ax.plot(s["n"], s["cv"], "o-", color=GROUPS[1][2], lw=2, ms=4,
            label="ephys units: median CV$^2$ per spike-count bin")
    ax.plot(s["n"], s["ref"], ":", color=C_INK, lw=1.2, label="exact null, same M (median)")
    k = np.logspace(0, np.log10(e["spk_count"].max()), 200)
    ax.plot(k, (1 - 1 / k) * np.median(s["ref"]), color=C_MUTED, lw=1.2,
            label="(1 - 1/spikes) x the null median")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("spikes in the analysed window", fontsize=8)
    ax.set_ylabel("median CV$^2$", fontsize=8)
    ax.set_title("C. ephys: CV$^2$ against spike count", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    m = pd.read_csv(OUT_CSV) if "--figures-only" in sys.argv else compute()
    t = m.groupby("group").agg(n=("cv2", "size"), M=("M", "median"),
                               median=("cv2", "median"), ref_median=("cv2_ref", "median"),
                               below_085=("cv2", lambda v: np.mean(v < 0.85)),
                               ref_below_085=("cv2_ref", lambda v: np.mean(v < 0.85)),
                               below_05=("cv2", lambda v: np.mean(v < 0.5)),
                               ref_below_05=("cv2_ref", lambda v: np.mean(v < 0.5)))
    print(t.round(3).to_string())
    figure(m, _HERE / "fig_cv2_marginals.png")


if __name__ == "__main__":
    main()
