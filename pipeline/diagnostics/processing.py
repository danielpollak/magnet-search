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
    "oddball":         ("teal",        0.20),
}


def _infer_kind(key, cfg):
    """Return the stimulus kind for an MM_d["aux"] key.

    Keys for auxiliary stimuli may have an orientation suffix appended
    (e.g. "recname_45"), so we check both exact match and prefix match.
    Keys that don't match any auxiliary stimulus are assumed to be magnetic.
    """
    for aux in cfg.auxiliary_stimuli:
        if key == aux.recname or key.startswith(aux.recname + "_"):
            return aux.kind
    return "magnetic"


def _windows_to_seconds(periods):
    """Normalise periods to an (N, 2) float array in seconds."""
    arr = np.asarray(periods, dtype=float) / AP_SR
    return arr[np.newaxis, :] if arr.ndim == 1 else arr


def plot_recording_timeline(cfg, MM_d, modulation_df, save_dir):
    """Save a rasterized PDF of the full recording session.

    When MM_d is available (openephys / openephys_multistim paradigms) the
    plot uses absolute spike times and shaded epoch windows derived from the
    raw MM_d data.  When MM_d is None (gutfreund / spikeglx_direct) the plot
    falls back to reconstructing approximate epoch boundaries from
    modulation_df spike times.

    Parameters
    ----------
    cfg          : ExperimentConfig
    MM_d         : dict {"spikes": {unit_id: np.ndarray samples},
                         "aux":    {rec_key: (periods_array, freq)}}
                   or None
    modulation_df: pd.DataFrame  columns [period, spk, phase, freq, id, rec]
    save_dir     : Path-like  destination directory (must already exist)
    """
    save_dir = Path(save_dir)
    if MM_d is not None:
        _timeline_from_MM_d(cfg, MM_d, save_dir)
    else:
        _timeline_from_modulation_df(cfg, modulation_df, save_dir)


# ---------------------------------------------------------------------------
# MM_d path (openephys / openephys_multistim)
# ---------------------------------------------------------------------------

def _timeline_from_MM_d(cfg, MM_d, save_dir):
    all_sts_sec = {k: v / AP_SR for k, v in MM_d["spikes"].items()}
    unit_ids = sorted(all_sts_sec.keys())
    spike_trains = [all_sts_sec[uid] for uid in unit_ids]
    n_units = len(unit_ids)

    non_empty = [st for st in spike_trains if len(st) > 0]
    total_dur = max(st.max() for st in non_empty) if non_empty else 1.0

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
    for key, (periods, freq) in MM_d["aux"].items():
        kind = _infer_kind(key, cfg)
        color, alpha = _KIND_STYLE.get(kind, ("mediumpurple", 0.20))
        windows = _windows_to_seconds(periods)  # (N, 2) in seconds

        for t0, t1 in windows:
            ax.axvspan(t0, t1, color=color, alpha=alpha, zorder=0)

        if kind not in legend_handles:
            legend_handles[kind] = mpatches.Patch(
                facecolor=color, alpha=0.6,
                label=kind.replace("_", " "))

        # Annotate each epoch block with its recording name
        label_x = (windows[:, 0].min() + windows[:, 1].max()) / 2
        ax.text(label_x, n_units + 0.5, key,
                ha="center", va="top", fontsize=8,
                rotation=-90, clip_on=False)

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
    """Approximate timeline when MM_d is unavailable.

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
        ax.axvspan(t0, t1, color=color, alpha=0.15, zorder=0)
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
