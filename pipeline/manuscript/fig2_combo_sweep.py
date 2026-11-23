"""Fig 2 supplement -- exhaustive pairwise combo-scatter sweep for panel F.

Panel F ("Visual + Audio") pairs one specific cross-modal condition
(secondsite's WN_SamCh vs. its 3Hz visual-gratings orientation 0) chosen by a
manual, undocumented sweep (see the comment above that call in fig2.py). This
script automates that sweep: for every (species, date, ID) session in the
visual/auditory positive-control population (statistics.get_poscontrols_
negresults's all_pos_control -- the same population panel B/F draw from), it
plots an NFC-vs-NFC scatter for every pair of (rec, freq) conditions recorded
in that session (matched by shared unit `id`), so a stronger example than the
current hardcoded one can be spotted by eye -- e.g. whether pigeon cerebellum
sessions show a cleaner quadrant than the current HP-based panel F.

Combos are sorted by "# both-significant suspects" descending (matching the
metric already used to justify the current panel F combo), one page per
combo grid, so the strongest candidates are up front.

Requires:
  data/manuscript/all_fourier_df.parquet  (run python pipeline/aggregate.py first)

Usage:
    python pipeline/manuscript/fig2_combo_sweep.py
    python pipeline/manuscript/fig2_combo_sweep.py --min-shared 15 --per-page 9
"""
import argparse
import itertools
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from magpyneto2 import statistics

import format_parameters as FP


def _modality(rec):
    """Coarse stimulus-modality label for a rec name, using the same
    substring conventions as statistics.get_poscontrols_negresults (WN/
    oddball -> auditory; visual/bars -> visual gratings). Fish/medaka recs
    have neither substring but are only ever in the positive-control
    population via a visual-frequency (<0.02Hz) fit, so they're visual too.
    """
    r = rec.lower()
    if "wn" in r or "oddball" in r:
        return "auditory"
    if "visual" in r or "bars" in r:
        return "visual"
    return "visual (fish/medaka)"


_TIMESTAMP_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_")


def _short_rec(rec):
    """Drop a leading recording timestamp (pigeon rec names carry one; other
    species' don't) so axis labels stay short enough to not collide with
    neighboring panels."""
    return _TIMESTAMP_PREFIX_RE.sub("", rec)


def _combo_stats(sess_df, cond_a, cond_b, thres, min_shared):
    rec_a, freq_a = cond_a
    rec_b, freq_b = cond_b
    a = (sess_df.loc[(sess_df.rec == rec_a) & (sess_df.freq == freq_a)]
         .drop_duplicates("id").set_index("id")["NFC"])
    b = (sess_df.loc[(sess_df.rec == rec_b) & (sess_df.freq == freq_b)]
         .drop_duplicates("id").set_index("id")["NFC"])
    common = a.index.intersection(b.index)
    if len(common) < min_shared:
        return None
    xa, xb = a.loc[common].values, b.loc[common].values
    both_sig = int(np.sum((xa > thres) & (xb > thres)))
    return dict(xa=xa, xb=xb, n=len(common), both_sig=both_sig)


def sweep_combos(all_fourier_df, min_shared=10):
    """Every pairwise (rec, freq)-vs-(rec, freq) combo within each session of
    the visual/auditory positive-control population, ranked by # suspects
    significant at both conditions (descending).
    """
    _, all_pos_control, _ = statistics.get_poscontrols_negresults(all_fourier_df)
    thres = statistics.inverse_Rayleigh_CDF(0.99)

    combos = []
    # Grouping by (species, date, ID) -- not "rec"/"area" -- matches the
    # unique-neuron dedup key used everywhere else in this codebase (see
    # CLAUDE.md's "Fixed (2026-08-25)" note): unit `id` numbering is only
    # stable/comparable within one subject's one session.
    for (species, date, ID), sess_df in all_pos_control.groupby(["species", "date", "ID"]):
        conditions = sorted(set(zip(sess_df["rec"], sess_df["freq"])))
        if len(conditions) < 2:
            continue
        areas = ", ".join(sorted(sess_df["area"].unique()))
        for cond_a, cond_b in itertools.combinations(conditions, 2):
            stats_d = _combo_stats(sess_df, cond_a, cond_b, thres, min_shared)
            if stats_d is None:
                continue
            mod_a, mod_b = _modality(cond_a[0]), _modality(cond_b[0])
            combos.append(dict(
                species=species, date=date, ID=ID, area=areas,
                rec_a=cond_a[0], freq_a=cond_a[1],
                rec_b=cond_b[0], freq_b=cond_b[1],
                modality_a=mod_a, modality_b=mod_b,
                cross_modal=(mod_a != mod_b),
                thres=thres, **stats_d))

    combos.sort(key=lambda d: d["both_sig"], reverse=True)
    return combos


def _plot_combo(ax, combo):
    xa, xb, thres = combo["xa"], combo["xb"], combo["thres"]
    both = (xa > thres) & (xb > thres)
    ax.scatter(xa[~both], xb[~both], s=10, c="grey", alpha=0.5)
    ax.scatter(xa[both], xb[both], s=10, c="tab:red", alpha=0.8,
               label=f"{combo['both_sig']}/{combo['n']} suspects")
    ax.axvline(thres, color="k", linewidth=FP.LW_REFERENCE)
    ax.axhline(thres, color="k", linewidth=FP.LW_REFERENCE)
    ax.set_xlim(0, max(7.5, float(xa.max()) * 1.05))
    ax.set_ylim(0, max(4.5, float(xb.max()) * 1.05))
    ax.set_xlabel(f"{_short_rec(combo['rec_a'])} ({combo['freq_a']:.4g} Hz) NFC",
                  fontsize=FP.FS_LEGEND - 1)
    ax.set_ylabel(f"{_short_rec(combo['rec_b'])} ({combo['freq_b']:.4g} Hz) NFC",
                  fontsize=FP.FS_LEGEND - 1)
    ax.legend(fontsize=FP.FS_LEGEND, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    tag = "" if combo["cross_modal"] else "  [same modality]"
    ax.set_title(f"{combo['species']} {combo['date']} {combo['ID']} ({combo['area']}){tag}",
                 fontsize=FP.FS_TITLE)


def _save_grid_pdf(combos, out_path, per_page, ncols=3):
    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY}
    matplotlib.rc("font", **font)
    with PdfPages(out_path) as pdf:
        for page_start in range(0, len(combos), per_page):
            page_combos = combos[page_start:page_start + per_page]
            nrows = int(np.ceil(len(page_combos) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows),
                                      squeeze=False)
            axes = axes.flatten()
            for ax, combo in zip(axes, page_combos):
                _plot_combo(ax, combo)
            for ax in axes[len(page_combos):]:
                ax.axis("off")
            fig.tight_layout(h_pad=3.5, w_pad=2.0)
            pdf.savefig(fig)
            plt.close(fig)


def plot_sweep(all_fourier_df, out_dir: Path, min_shared=10, per_page=6):
    combos = sweep_combos(all_fourier_df, min_shared=min_shared)
    cross_combos = [c for c in combos if c["cross_modal"]]
    print(f"{len(combos)} combos with >= {min_shared} shared units "
          f"({len(cross_combos)} genuinely cross-modal, i.e. auditory vs. visual)")

    print(f"\nTop 15 overall (any modality pairing), by # both-significant:")
    for c in combos[:15]:
        tag = "" if c["cross_modal"] else " [same modality]"
        print(f"  {c['both_sig']:>3d}/{c['n']:<4d}  {c['species']:<10s} {c['date']} {c['ID']:<10s} "
              f"({c['area']})  {c['rec_a']} [{c['freq_a']:.4g} Hz]  vs  "
              f"{c['rec_b']} [{c['freq_b']:.4g} Hz]{tag}")

    print(f"\nTop 15 cross-modal (auditory vs. visual) only, by # both-significant:")
    for c in cross_combos[:15]:
        print(f"  {c['both_sig']:>3d}/{c['n']:<4d}  {c['species']:<10s} {c['date']} {c['ID']:<10s} "
              f"({c['area']})  {c['rec_a']} [{c['freq_a']:.4g} Hz]  vs  "
              f"{c['rec_b']} [{c['freq_b']:.4g} Hz]")

    # Full sweep (every modality pairing -- includes same-modality reliability
    # checks, e.g. zebrafish visual-vs-visualmagnet, which dominate the top of
    # the ranking by raw both-significant count but aren't candidates for
    # panel F's specifically CROSS-modal comparison)
    out_path_all = out_dir / "Fig2_F_combo_sweep.pdf"
    _save_grid_pdf(combos, out_path_all, per_page)
    print(f"\nSaved {out_path_all} ({len(combos)} combos, "
          f"{int(np.ceil(len(combos) / per_page))} pages)")

    # Cross-modal-only subset -- the population panel F is actually drawn
    # from; this is the ranking that matters for picking a replacement combo.
    out_path_cross = out_dir / "Fig2_F_combo_sweep_crossmodal.pdf"
    _save_grid_pdf(cross_combos, out_path_cross, per_page)
    print(f"Saved {out_path_cross} ({len(cross_combos)} combos, "
          f"{int(np.ceil(len(cross_combos) / per_page))} pages)")


def main():
    parser = argparse.ArgumentParser(
        description="Exhaustive pairwise combo-scatter sweep for Fig2 panel F")
    parser.add_argument("--out-dir", default=FP.DIAGNOSTIC_DIR, help="Output directory for the PDFs")
    parser.add_argument("--parquet", default=FP.PARQUET_PATH,
                        help=f"Path to all_fourier_df.parquet (default: {FP.PARQUET_PATH})")
    parser.add_argument("--min-shared", type=int, default=10,
                        help="Minimum units shared between two conditions to plot a combo (default 10)")
    parser.add_argument("--per-page", type=int, default=6, help="Scatter panels per PDF page")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.parquet} ...")
    all_fourier_df = pd.read_parquet(args.parquet)
    plot_sweep(all_fourier_df, out_dir, min_shared=args.min_shared, per_page=args.per_page)


if __name__ == "__main__":
    main()
