"""The oddball stimulus, reconstructed from its recorded photodiode trace.

The oddball (pigeon cerebellum, 20230221 and 20230228) is a full-screen flash train: the
screen alternates on/off every ~0.515 s and on ~5% of frames holds its
state, giving a longer flash ("long flash") or a longer dark gap ("long
gap"); the deviants differ only in duration (PsychoPy cell "ON OFF
stimuli" in MagnetSearch/code/experiments/2023 03 15 expts/
sound_visual_stim.py). Only the standard trials are Fourier-analysed.

Every flash below is a real one, from the photodiode onsets/offsets that
oddball_extract_ttl.py saved to data/oddball_ttl/ -- not a fresh random
draw. The screen is drawn white while it was lit and black while it was
dark: in A's screen row, and as a bar above each raster in B-D.

UNITS were chosen by eye from the units of both sessions with p < 0.01 for
a flash response visible in the raster itself: the lowest-p units either
fire too fast to read as ticks (20230221 unit 6, ~75 Hz) or are only weakly
modulated (20230221 unit 659, 1.5x between PSTH peak and trough).

Panel B shows each standard flash's spikes over exactly the window the
analysis counts them in (_pad_oddball_windows: half a flash before onset to
half a flash after offset, about one cycle), so no spike appears twice.

Outputs (figs/paper/supplement/):
  oddball_stimulus_strip.pdf -- one page per unit in UNITS. A: 16 s of the
      flash train with the unit's spikes; B-D: the unit's spikes aligned to
      flash onset, raster above, PSTH below: B every standard flash (within
      its analysis window), C every long flash and D every long gap (each
      from the previous flash's end to the next flash's start). Ticks left
      of B's raster mark where each long flash (red) and long gap (orange)
      fell among the standard flashes.
  oddball_stimulus_{session}.mp4 -- 21 s, real time, one per session: the
      screen as the bird saw it, above a scrolling timeline of the flash
      train and the spikes of each of the session's UNITS, stacked.

    python pipeline/manuscript/oddball_extract_ttl.py   # once, NAS access
    python pipeline/manuscript/oddball_stimulus.py
"""
from pathlib import Path

import matplotlib
import matplotlib.transforms
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

import format_parameters as FP

REPO = Path(__file__).resolve().parents[2]
SESSIONS = {
    # strip: 16 s, s from recording start, with long flashes and long gaps in it
    # video: 21 s around it, more long flashes and long gaps
    "20230221": dict(rec="pigeon_CB_oddball_5percent_2023-02-21_13-36-00", strip=(48.0, 64.0),
                     video=(43.0, 64.0)),
    "20230228": dict(rec="oddball_2023-02-28_17-25-32", strip=(28.0, 44.0), video=(22.5, 43.5)),
}
UNITS = [("20230228", 140), ("20230228", 133), ("20230228", 829), ("20230221", 9)]

# Same on/off split as _handle_oddball (experiments/20230221.yml): <= 0.7 s
# normal, >= 0.9 s long.
LONG_MIN_S = 0.9

VIDEO_FPS = 30
VIDEO_SPAN_S = 8.0            # width of the scrolling timeline
PSTH_BIN_S = 0.05
# No ffmpeg in magneto2; the one in the magneto env.
FFMPEG = Path.home() / "anaconda3" / "envs" / "magneto" / "Scripts" / "ffmpeg.exe"

LIT, DARK = "white", "black"     # dark is possibly mid-grey in reality (PsychoPy colour 0)
COLOR_LONG_FLASH = "crimson"     # _CATEGORY_HUE["long_on"] in pipeline/diagnostics/oddball.py
COLOR_LONG_GAP = "darkorange"    # _CATEGORY_HUE["long_off"]


def load_flashes(exp):
    z = np.load(REPO / "data" / "oddball_ttl" / f"{SESSIONS[exp]['rec']}.npz")
    t_up, t_down = z["t_up"], z["t_down"]
    gap_before = np.r_[np.nan, t_up[1:] - t_down[:-1]]
    return t_up, t_down, gap_before, float(z["rec_start_s"])


def unit_p_value(df, exp, cluster_id):
    return df.loc[(df.rec == SESSIONS[exp]["rec"]) & (df.id == cluster_id), "p_value"].iloc[0]


def load_spikes(exp, cluster_id, rec_start_s):
    with NWBHDF5IO(str(REPO / "data" / f"{exp}.nwb"), "r") as io:
        units = io.read().units.to_dataframe()
    st = np.asarray(units.loc[units.cluster_id == cluster_id, "spike_times"].iloc[0])
    return st - rec_start_s


def draw_train(ax, t_up, t_down, gap_before, window, y0=0.0, h=1.0):
    """The screen: dark, with each flash a lit bar. Each long flash and long
    gap is named above the bar, in its own colour, with its duration inside."""
    ax.add_patch(Rectangle((window[0], y0), window[1] - window[0], h, facecolor=DARK, linewidth=0))
    # Outline, so a lit bar at either end still shows where the screen ends.
    ax.add_patch(Rectangle((window[0], y0), window[1] - window[0], h, facecolor="none",
                           edgecolor=DARK, linewidth=FP.LW_THIN, zorder=3))
    for on, off, gap in zip(t_up, t_down, gap_before):
        if off < window[0] or on > window[1]:
            continue
        ax.add_patch(Rectangle((on, y0), off - on, h, facecolor=LIT, linewidth=0))
        spans = []
        if off - on >= LONG_MIN_S:
            spans.append((on, off - on, "long flash", COLOR_LONG_FLASH, DARK))
        if gap >= LONG_MIN_S and on - gap > window[0]:
            spans.append((on - gap, gap, "long gap", COLOR_LONG_GAP, LIT))
        for start, dur, kind, color, inside in spans:
            ax.text(start + dur / 2, y0 + h + 0.05, kind, color=color, ha="center", va="bottom",
                    fontsize=FP.FS_LEGEND - 1, clip_on=True)
            ax.text(start + dur / 2, y0 + h / 2, f"{dur:.2f} s", color=inside, ha="center",
                    va="center", fontsize=FP.FS_LEGEND - 1, clip_on=True)


def aligned_panel(ax_r, ax_p, spikes, onsets, lo, hi, on, bin_s):
    """Spikes of each trial between onset+lo and onset+hi, aligned at onset:
    raster, one row per trial, and below it the PSTH, each bin divided by
    the time the windows cover it; above the raster, the screen (lit from 0
    to the median flash duration). Returns the PSTH's peak rate."""
    aligned = []
    for row, (t0, a, b) in enumerate(zip(onsets, lo, hi)):
        s = spikes[(spikes > t0 + a) & (spikes < t0 + b)] - t0
        ax_r.vlines(s, row, row + 1, color="k", linewidth=FP.LW_THIN)
        aligned.append(s)
    pre, post, flash = -np.median(lo), np.median(hi), np.median(on)
    edges = np.r_[np.arange(-pre, post - bin_s / 2, bin_s), post]
    cover = np.clip(np.minimum(edges[1:, None], hi) - np.maximum(edges[:-1, None], lo), 0, None).sum(1)
    rate = np.histogram(np.concatenate(aligned), edges)[0] / cover
    ax_p.stairs(rate, edges, color="k", linewidth=FP.LW_TRACE)
    # The flash lasts the same in every trial to within a few ms (except
    # C's one 2.55 s flash), so the bar shows the median trial's screen.
    bar = matplotlib.transforms.blended_transform_factory(ax_r.transData, ax_r.transAxes)
    for x0, x1 in ((-pre, 0), (flash, post)):
        ax_r.add_patch(Rectangle((x0, 1.03), x1 - x0, 0.07, facecolor=DARK, linewidth=0,
                                 transform=bar, clip_on=False))
    ax_r.add_patch(Rectangle((-pre, 1.03), pre + post, 0.07, facecolor="none", edgecolor=DARK,
                             linewidth=FP.LW_THIN, transform=bar, clip_on=False))
    ax_r.set_xlim(-pre, post)
    ax_r.set_ylim(0, len(onsets))
    ax_r.set_yticks([0, len(onsets)])
    ax_r.tick_params(labelbottom=False)
    for ax in (ax_r, ax_p):
        ax.spines[["top", "right"]].set_visible(False)
    return rate.max()


def plot_strip(t_up, t_down, gap_before, spikes, exp, cluster_id, pval, strip_window):
    on = t_down - t_up
    inner = np.arange(1, len(t_up) - 1)
    std = inner[(on[inner] < LONG_MIN_S) & (gap_before[inner] < LONG_MIN_S)]
    long_flash = inner[on[inner] >= LONG_MIN_S]
    long_gap = inner[gap_before[inner] >= LONG_MIN_S]

    # B: every standard flash (normal flash after a normal gap) over its own
    # analysis window (_pad_oddball_windows: pad half the flash on each side,
    # never into a neighbouring flash), so no spike is drawn twice.
    std_lo = np.maximum(t_up[std] - on[std] / 2, t_down[std - 1]) - t_up[std]
    std_hi = np.minimum(t_down[std] + on[std] / 2, t_up[std + 1]) - t_up[std]

    # C, D: each deviant from the end of the previous flash to the start of
    # the next one (deviants are never adjacent, so again no spike twice);
    # D is aligned at the onset of the flash that ends the long gap. Neither
    # is Fourier-analysed.
    def neighbours(idx):
        return t_down[idx - 1] - t_up[idx], t_up[idx + 1] - t_up[idx]

    panels = [
        ("B", "Standard", "k", std, std_lo, std_hi, PSTH_BIN_S),
        ("C", "Long flash", COLOR_LONG_FLASH, long_flash, *neighbours(long_flash), 2 * PSTH_BIN_S),
        ("D", "Long gap", COLOR_LONG_GAP, long_gap, *neighbours(long_gap), 2 * PSTH_BIN_S),
    ]
    # Widths proportional to the time shown, so all three share one time scale.
    spans_s = [np.median(hi) - np.median(lo) for *_, lo, hi, _ in panels]

    fig = plt.figure(figsize=(FP.FIGSIZE_FIG3[0], 4.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.45)
    ax_a = fig.add_subplot(outer[0])
    gs = outer[1].subgridspec(2, 3, width_ratios=spans_s, wspace=0.3, hspace=0.12)
    w0, w1 = strip_window

    draw_train(ax_a, t_up, t_down, gap_before, strip_window, y0=1.2, h=1.0)
    s = spikes[(spikes > w0) & (spikes < w1)]
    ax_a.vlines(s, 0, 1, color="k", linewidth=FP.LW_THIN)
    ax_a.set_xlim(w0, w1)
    ax_a.set_ylim(-0.1, 2.9)
    ax_a.set_yticks([0.5, 1.7], [f"unit {cluster_id}", "screen"])
    ax_a.tick_params(axis="y", length=0)
    ax_a.set_xticks([w0, w0 + 5], ["0", "5 s"])
    ax_a.spines[["top", "right", "left"]].set_visible(False)

    # Durations of one standard cycle.
    i = np.flatnonzero((t_up > w0 + 0.5) & (t_down - t_up < LONG_MIN_S))[0]
    std_on, std_off = t_down[i] - t_up[i], t_up[i + 1] - t_down[i]
    ax_a.annotate(f"{std_on:.3f} s flash, {std_off:.3f} s gap", (t_up[i], 2.55), xytext=(0, 4),
                  textcoords="offset points", fontsize=FP.FS_LEGEND, va="bottom")
    ax_a.set_title(f"pigeon {exp}, unit {cluster_id}, standard-flash p = {pval:.1g}",
                   fontsize=FP.FS_TITLE, loc="right")
    ax_a.text(-0.02, 1.02, "A", transform=ax_a.transAxes, fontsize=FP.FS_PANEL, ha="right", va="bottom")

    psth_axes, peak = [], 0
    for col, (letter, title, color, idx, lo, hi, bin_s) in enumerate(panels):
        ax_r = fig.add_subplot(gs[0, col])
        ax_p = fig.add_subplot(gs[1, col], sharex=ax_r,
                               sharey=psth_axes[0] if psth_axes else None)
        peak = max(peak, aligned_panel(ax_r, ax_p, spikes, t_up[idx], lo, hi, on[idx], bin_s))
        psth_axes.append(ax_p)
        ax_r.set_title(f"{title} (n = {len(idx)})", fontsize=FP.FS_TITLE, color=color, pad=10)
        ax_p.set_xticks(np.arange(np.ceil(np.median(lo) / 0.5) * 0.5, np.median(hi) + 1e-9, 0.5))
        ax_p.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x + 0:g}"))
        ax_r.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-22 if col else -32, 16),
                      textcoords="offset points", fontsize=FP.FS_PANEL, ha="right", va="bottom")
        if col == 0:
            ax_r.set_ylabel("Trial")
            ax_p.set_ylabel("Rate (Hz)")
        else:
            ax_p.tick_params(labelleft=False)
    psth_axes[0].set_ylim(0, peak * 1.1)
    # Where each deviant fell among B's (time-ordered) standard flashes, as a
    # tick left of B's raster: does B's response come from the flashes next
    # to a deviant?
    ax_b = fig.axes[1]
    side = matplotlib.transforms.blended_transform_factory(ax_b.transAxes, ax_b.transData)
    for idx, color in ((long_flash, COLOR_LONG_FLASH), (long_gap, COLOR_LONG_GAP)):
        ax_b.hlines(np.searchsorted(std, idx), -0.1, -0.02, color=color, linewidth=1.5,
                    transform=side, clip_on=False)
    psth_axes[1].set_xlabel("Time from flash onset (s)")
    return fig


def make_video(t_up, t_down, gap_before, spikes_by_unit, exp, video_window, out_path):
    """spikes_by_unit: {cluster_id: spike times}, one timeline row each,
    in order from the top, all from session `exp`."""
    plt.rcParams["animation.ffmpeg_path"] = str(FFMPEG)
    v0, v1 = video_window
    n = len(spikes_by_unit)
    fig = plt.figure(figsize=(6.4, 6.4), facecolor="white")
    ax_screen = fig.add_axes((0.25, 0.55, 0.5, 0.39))
    ax_t = fig.add_axes((0.16, 0.08, 0.8, 0.36))

    ax_screen.set_xticks([])
    ax_screen.set_yticks([])
    screen = Rectangle((0, 0), 1, 1, transform=ax_screen.transAxes, facecolor=DARK)
    ax_screen.add_patch(screen)
    label = ax_screen.text(0.5, -0.02, "", transform=ax_screen.transAxes, ha="center", va="top",
                           fontsize=FP.FS_BODY + 2)
    clock = ax_screen.text(1.0, 1.02, "", transform=ax_screen.transAxes, ha="right", va="bottom",
                           fontsize=FP.FS_BODY)
    ax_screen.set_title(f"Screen (oddball, pigeon {exp})", fontsize=FP.FS_BODY + 1, loc="left")

    draw_train(ax_t, t_up, t_down, gap_before, (v0 - VIDEO_SPAN_S, v1 + VIDEO_SPAN_S), y0=n + 0.2, h=1.0)
    for row, spikes in enumerate(spikes_by_unit.values()):
        s = spikes[(spikes > v0 - VIDEO_SPAN_S) & (spikes < v1 + VIDEO_SPAN_S)]
        ax_t.vlines(s, n - 1 - row + 0.1, n - row - 0.1, color="k", linewidth=FP.LW_THIN)
    playhead = ax_t.axvline(v0, color="tab:blue", linewidth=FP.LW_TRACE)
    ax_t.set_ylim(-0.1, n + 1.6)
    ax_t.set_yticks([n - 0.5 - row for row in range(n)] + [n + 0.7],
                    [f"unit {c}" for c in spikes_by_unit] + ["screen"])
    ax_t.tick_params(axis="y", length=0)
    ax_t.set_xlabel("Time (s)")
    # Same zero as the clock above the screen.
    ax_t.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x - v0:g}"))
    ax_t.spines[["top", "right", "left"]].set_visible(False)

    frames = np.arange(v0, v1, 1 / VIDEO_FPS)

    def update(t):
        k = np.searchsorted(t_up, t, side="right") - 1
        lit = k >= 0 and t < t_down[k]
        screen.set_facecolor(LIT if lit else DARK)
        if lit:
            text, color = (("long flash", COLOR_LONG_FLASH) if t_down[k] - t_up[k] >= LONG_MIN_S
                           else ("flash", "k"))
        else:
            gap = t_up[k + 1] - t_down[k] if k + 1 < len(t_up) else 0
            text, color = (("long gap", COLOR_LONG_GAP) if gap >= LONG_MIN_S else ("gap", "k"))
        label.set_text(text)
        label.set_color(color)
        clock.set_text(f"{t - v0:5.1f} s")
        playhead.set_xdata([t, t])
        ax_t.set_xlim(t - VIDEO_SPAN_S / 2, t + VIDEO_SPAN_S / 2)
        return screen, label, clock, playhead

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(out_path, writer=animation.FFMpegWriter(fps=VIDEO_FPS, bitrate=2000), dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    out_dir = Path(FP.SUPPLEMENT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FP.PARQUET_PATH)
    strip_path = out_dir / "oddball_stimulus_strip.pdf"
    with PdfPages(strip_path) as pdf:
        for exp, cluster_id in UNITS:
            t_up, t_down, gap_before, rec_start_s = load_flashes(exp)
            spikes = load_spikes(exp, cluster_id, rec_start_s)
            fig = plot_strip(t_up, t_down, gap_before, spikes, exp, cluster_id,
                             unit_p_value(df, exp, cluster_id), SESSIONS[exp]["strip"])
            pdf.savefig(fig, bbox_inches="tight", dpi=FP.DPI)
            plt.close(fig)
    print(f"Saved {strip_path}")

    # One video per session (the sessions saw different flash trains), all
    # of its units in UNITS stacked.
    for exp, session in SESSIONS.items():
        ids = [c for e, c in UNITS if e == exp]
        t_up, t_down, gap_before, rec_start_s = load_flashes(exp)
        spikes = {c: load_spikes(exp, c, rec_start_s) for c in ids}
        make_video(t_up, t_down, gap_before, spikes, exp, session["video"],
                   out_dir / f"oddball_stimulus_{exp}.mp4")


if __name__ == "__main__":
    main()
