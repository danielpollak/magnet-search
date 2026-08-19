"""
Processing-stage diagnostic plots.

plot_recording_timeline  — full-session raster with epoch shading (PDF, rasterized spikes)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

AP_SR = 30_000  # Hz

# (color, alpha) for each stimulus kind
_KIND_STYLE = {
    "magnetic":        ("steelblue",   0.20),
    "visual_gratings": ("darkorange",  0.20),
    "white_noise":     ("dimgray",     0.15),
    "visual_bars":     ("forestgreen", 0.20),
    "oddball":           ("teal",          0.20),
    "oddball_long_on":   ("crimson",       0.20),
    # goldenrod, not darkorange (visual_gratings' color above) -- a session
    # combining visual_gratings + oddball would otherwise be ambiguous here.
    "oddball_long_off":  ("goldenrod",     0.20),
    "oddball_long_both": ("mediumorchid",  0.20),
}


def plot_recording_timeline(cfg, nwbfile, modulation_df, save_dir, cat_df=None):
    """Save a rasterized PDF of the full recording session.

    When an NWB file is available (openephys / openephys_multistim
    paradigms) the plot reads spike times and epoch windows straight from
    its `Units` / `stimulus_epochs` tables -- the same artifacts the
    analysis stage consumes. When `nwbfile` is None (gutfreund /
    spikeglx_direct) the plot falls back to reconstructing approximate
    epoch boundaries from modulation_df spike times.

    Parameters
    ----------
    cfg          : ExperimentConfig
    nwbfile      : pynwb NWBFile already containing Units + stimulus_epochs,
                   or None
    modulation_df: pd.DataFrame  columns [period, spk, phase, freq, id, rec]
    save_dir     : Path-like  destination directory (must already exist)
    cat_df       : optional pd.DataFrame, the full concatenated-session
                   metadata table (columns incl. "recname", "cumulate", in
                   AP-sample units) -- when given, every recording block in
                   the raw session gets shaded/labelled, not just the ones
                   this experiment's YAML actually analyzes. Without it, any
                   recording absent from stimulus_epochs (e.g. this date's
                   un-ported visual/audio/WN blocks) just reads as
                   unexplained empty time -- see _handle_oddball's neighbour
                   handlers for why an openephys-paradigm YAML may only cover
                   a handful of the many recordings in a concatenated session.
    """
    save_dir = Path(save_dir)
    if nwbfile is not None:
        _timeline_from_nwb(cfg, nwbfile, save_dir, cat_df=cat_df)
    else:
        _timeline_from_modulation_df(cfg, modulation_df, save_dir)


# ---------------------------------------------------------------------------
# NWB path (openephys / openephys_multistim)
# ---------------------------------------------------------------------------

def _timeline_from_nwb(cfg, nwbfile, save_dir, cat_df=None):
    units_df = nwbfile.units.to_dataframe()
    spike_trains_sec = {
        int(uid): np.asarray(row["spike_times"])
        for uid, row in units_df.iterrows()
    }
    unit_ids = sorted(spike_trains_sec.keys())
    spike_trains = [spike_trains_sec[uid] for uid in unit_ids]
    n_units = len(unit_ids)

    epochs_df = nwbfile.intervals["stimulus_epochs"].to_dataframe()

    non_empty = [st for st in spike_trains if len(st) > 0]
    total_dur = max(st.max() for st in non_empty) if non_empty else 1.0
    if len(epochs_df):
        total_dur = max(total_dur, epochs_df["stop_time"].max())
    if cat_df is not None and len(cat_df):
        total_dur = max(total_dur, cat_df["cumulate"].iloc[-1] / AP_SR)

    # Scale figure: ~1 inch per 30 s, capped so the PDF isn't absurd
    fig_w = float(np.clip(total_dur / 30, 20, 80))
    fig_h = float(np.clip(n_units * 0.05, 5, 30))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── Spike eventplot (rasterized so PDF stays small) ─────────────────────
    artists = ax.eventplot(
        spike_trains,
        lineoffsets=np.arange(n_units),
        linelengths=0.8,
        linewidths=0.3,
        colors=["black"] * n_units,
        alpha=0.25,
    )
    for a in artists:
        a.set_rasterized(True)

    # ── Epoch shading ────────────────────────────────────────────────────────
    legend_handles = {}
    analyzed_recs = []
    for _, epoch in epochs_df.iterrows():
        kind = epoch["stim_type"]
        color, alpha = _KIND_STYLE.get(kind, ("mediumpurple", 0.20))
        t0, t1 = float(epoch["start_time"]), float(epoch["stop_time"])
        analyzed_recs.append(epoch["rec"])

        ax.axvspan(t0, t1, facecolor=color, edgecolor="none", alpha=alpha, zorder=0)

        if kind not in legend_handles:
            legend_handles[kind] = mpatches.Patch(
                facecolor=color, alpha=0.6,
                label=kind.replace("_", " "))

        # Annotate each epoch block with its recording name
        ax.text((t0 + t1) / 2, n_units + 0.5, epoch["rec"],
                ha="center", va="top", fontsize=8,
                rotation=-90, clip_on=False)

    # ── "Not analyzed" shading ──────────────────────────────────────────────
    # cat_df lists every recording block in the raw concatenated session;
    # stimulus_epochs only has rows for the ones this experiment's YAML
    # actually turned into trials/auxiliary_stimuli (e.g. 20220621 analyzes
    # 4 of its 17 recording blocks -- the rest are other stimulus types never
    # ported to this pipeline for that date, matching the original legacy
    # script's own scope, not a bug). Without this, those un-analyzed blocks
    # look like unexplained empty time even though they're full of spikes.
    if cat_df is not None and len(cat_df):
        cumulate = cat_df["cumulate"].values / AP_SR
        starts = np.concatenate([[0.0], cumulate[:-1]])
        labelled_unanalyzed = False
        for (t0, t1), recname in zip(zip(starts, cumulate), cat_df["recname"]):
            covered = any(k == recname or k.startswith(recname + "_")
                          for k in analyzed_recs)
            if covered:
                continue
            ax.axvspan(t0, t1, facecolor="lightgray", edgecolor="none", alpha=0.35, zorder=-1)
            ax.text((t0 + t1) / 2, n_units + 0.5, f"{recname} (not analyzed)",
                    ha="center", va="top", fontsize=8, rotation=-90,
                    clip_on=False, color="dimgray")
            labelled_unanalyzed = True
        if labelled_unanalyzed:
            legend_handles["not analyzed"] = mpatches.Patch(
                facecolor="lightgray", alpha=0.6, label="not analyzed")

    ax.set_xlim(0, total_dur * 1.02)
    ax.set_ylim(-1, n_units + 4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Unit  (N={n_units})")
    ax.set_yticks([0, n_units - 1])
    ax.set_yticklabels(["0", str(n_units - 1)])
    if legend_handles:
        ax.legend(handles=list(legend_handles.values()),
                  loc="upper right", fontsize=7, framealpha=0.8)
    ax.set_title(f"{cfg.name}  —  recording timeline")

    fig.tight_layout()
    fig.savefig(save_dir / f"{cfg.name}_timeline.pdf",
                bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# modulation_df fallback (gutfreund / spikeglx_direct)
# ---------------------------------------------------------------------------

_FALLBACK_COLORS = [
    "steelblue", "darkorange", "forestgreen", "dimgray",
    "teal", "mediumpurple", "firebrick",
]


def _timeline_from_modulation_df(cfg, modulation_df, save_dir):
    """Approximate timeline when no NWB file is available.

    Spike times in modulation_df are per-recording-relative, so epochs of
    different recordings may overlap on the X-axis.  The plot is still useful
    for spotting within-recording alignment issues.
    """
    unit_ids = sorted(modulation_df.id.unique())
    n_units = len(unit_ids)

    fig_w = 20
    fig_h = float(np.clip(n_units * 0.05, 5, 20))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Epoch shading (approximate boundaries from spike extent per rec)
    recs = list(modulation_df.rec.unique())
    legend_handles = []
    for rec_i, rec in enumerate(recs):
        spk_sub = modulation_df.loc[modulation_df.rec == rec, "spk"]
        t0, t1 = spk_sub.min(), spk_sub.max()
        color = _FALLBACK_COLORS[rec_i % len(_FALLBACK_COLORS)]
        ax.axvspan(t0, t1, facecolor=color, edgecolor="none", alpha=0.15, zorder=0)
        legend_handles.append(
            mpatches.Patch(facecolor=color, alpha=0.5,
                           label=rec if len(rec) <= 30 else rec[-30:]))

    # Spike eventplot (rasterized)
    spike_trains = [
        modulation_df.loc[modulation_df.id == uid, "spk"].values
        for uid in unit_ids
    ]
    artists = ax.eventplot(
        spike_trains,
        lineoffsets=np.arange(n_units),
        linelengths=0.8,
        linewidths=0.3,
        colors=["black"] * n_units,
        alpha=0.3,
    )
    for a in artists:
        a.set_rasterized(True)

    ax.set_xlabel("Time within epoch (s)  [per-recording-relative]")
    ax.set_ylabel(f"Unit  (N={n_units})")
    ax.set_yticks([0, n_units - 1])
    ax.set_yticklabels(["0", str(n_units - 1)])
    ax.legend(handles=legend_handles[:10], loc="upper right",
              fontsize=5, framealpha=0.8, ncol=2)
    ax.set_title(f"{cfg.name}  —  recording timeline  (modulation_df fallback)")

    fig.tight_layout()
    fig.savefig(save_dir / f"{cfg.name}_timeline.pdf",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
