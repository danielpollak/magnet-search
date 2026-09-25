"""Fig 3 supplement — detection sensitivity vs. measured modulation depth.

Fig3.pdf's exact layout (four quadrants A-D, each split into a "Magnetic"
(negative-result) and a "Visual/Audio" (positive-control) sub-axes, populations
built by `statistics.get_poscontrols_negresults`), but every one of the eight
axes is a SCATTERPLOT: per-unit detection sensitivity `sens` on x, measured
modulation depth on y, and every point coloured by its own 1F p-value on a
log scale -- NOT by occurrence wave as Fig3's own panels are. See PVAL_CMAP /
PVAL_FLOOR for the colour scale and why the wave split still happens anyway.

Modulation depth is measured exactly as asked for: the amplitude of a sinusoid
fitted to the unit's own raw PSTH at that unit's own stimulation frequency,
with amplitude and phase as the only free parameters (the frequency is known,
it is the `freq` the unit's p-value was computed at). Concretely, per
(rec, freq, unit):

    phase_k  = (spk_k mod 1/freq) / (1/freq) * 2*pi        # nominal-period fold
    psth_b   = count_in_bin_b * PSTH_N_BINS / T            # spikes/s, PSTH_N_BINS bins
    psth_b  ~= m0 + a*cos(theta_b) + b*sin(theta_b)        # ordinary least squares
    amp_hz   = hypot(a, b)                                 # page 2's y
    depth    = amp_hz / m0                                 # page 1's y

`T` is read from the NWB `fourier_group_results` table, so it is bit-for-bit
the duration `fit_fourier_sig` used for this group -- which is what makes
`amp_hz` directly comparable to the persisted `sens` on the x axis.

Four pages. The first two are sensitivity vs. the two useful
normalisations of "modulation depth":
  page 1  relative depth `amp_hz / mean_rate` -- dimensionless, 0-1, the same
          quantity Fig4's simulation sweeps as its synthetic amplitude A, and
          comparable across units of wildly different firing rate.
  page 2  absolute amplitude `amp_hz` in spikes/s -- raw effect size, but
          dominated by each unit's own firing rate.
The last two put firing rate (`mean_rate_hz`, = spk_count/T) on x, to show
how much of each of those quantities firing rate alone accounts for:
  page 3  sensitivity vs. firing rate.
  page 4  relative modulation depth vs. firing rate.

Two versions of the same four pages are written, differing only in what the
point colour encodes (see COLOR_SPECS):
  Fig3_modulation_depth.pdf           the unit's own 1F p-value.
  Fig3_modulation_depth_duration.pdf  the total duration T of the spike train
                                      the p-value was computed from.

Only EPHYS species appear (pigeon, zebra finch, quail). A spike PSTH does not
exist for the rest: mouse/owl are precomputed analysis frames with no raw data
in the repo at all, and zebrafish/medaka are GCaMP dF/F traces, whose
sinusoid amplitude is in fluorescence units and does not share a y axis with
spikes/s. So quadrant A is "all ephys species", not literally all species --
its title says so.

Why a least-squares fit rather than reusing the persisted Fourier
coefficient: for uniform phase bins the two-parameter fit above IS the
one-term DFT of the histogram, so `amp_hz` reproduces `2*|fou0|` (the
on-frequency coefficient already in every NWB) up to the bin-width sinc
attenuation, sinc(pi/PSTH_N_BINS) ~ 0.9997 at 72 bins. `--validate` prints
that comparison per experiment; it is a check on this file, not the source of
the plotted numbers, which really do come from the PSTH.

Requires:
  data/manuscript/all_fourier_df.parquet  (run python pipeline/aggregate.py first)
  data/{experiment}.nwb for every ephys experiment

The per-unit PSTH fits are cached to data/manuscript/ on first run (~20 NWBs,
a few minutes) and reused afterwards; pass --recompute to force a refit.

Usage:
    python pipeline/manuscript/fig3_modulation_depth.py
    python pipeline/manuscript/fig3_modulation_depth.py --recompute --validate
    python pipeline/manuscript/fig3_modulation_depth.py --workers 8
    python pipeline/manuscript/fig3_modulation_depth.py --color duration
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

# Detect if running in Jupyter notebook (must do this before matplotlib.use)
try:
    from IPython import get_ipython
    in_notebook = get_ipython() is not None
except ImportError:
    in_notebook = False

import matplotlib
if not in_notebook:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from magpyneto2 import statistics
from pipeline import nwb_io
from pipeline.schema import load_experiment

import format_parameters as FP
from fig3 import _split_into_waves

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EXPERIMENTS_DIR = _REPO_ROOT / "experiments"

# Paradigms that produce a `Units` table, i.e. the ones a spike PSTH exists
# for. "engert"/"medaka" are imaging (see the module docstring); "manual" never
# writes an NWB at all.
_EPHYS_PARADIGMS = {"openephys", "openephys_multistim", "gutfreund", "spikeglx_direct"}

# Phase bins in the fitted PSTH. 72 is fig1.py's/fig4.py's own PSTH
# resolution (their PSTH_N_BINS), so the histogram being fitted here is the
# same one those figures draw for their exemplar units. It also keeps the
# bin-width sinc attenuation on the fitted amplitude at 0.03%
# (sinc(pi/72) = 0.99968) -- small enough that it is reported by --validate
# rather than corrected for.
PSTH_N_BINS = 72

# Stamped with the bin count: the fitted amplitude is a function of it (via
# the sinc attenuation above), so a cache built at a different resolution must
# not be silently reloaded as if it matched.
MOD_DEPTH_CACHE = (_REPO_ROOT / "data" / "manuscript"
                   / f"psth_modulation_depth_{PSTH_N_BINS}bins.parquet")

# Null-threshold reference line on page 1 (see _add_threshold_line): the
# modulation depth at which a unit of a given sensitivity would reach the
# NFC the uncorrected/eps-corrected null distribution puts this much mass
# below. Same percentile Fig2's excess-count statistic counts crossings at.
THRESHOLD_PERCENTILE = 0.99

# Every point is coloured by its own 1F p-value on a log scale, rather than by
# which occurrence wave (1st/2nd/... magnetic trial) it came from as Fig3's
# own panels are. That makes hue a quantitative encoding shared by all eight
# axes, so the same colour means the same p-value everywhere -- including
# across the Magnetic/Visual-Audio pair, where an occurrence-wave colour
# deliberately could NOT be shared (wave index means different things in the
# two populations, see fig3.plot_uniform_p's docstring). The wave SPLIT is
# still performed, because quadrant D's top-decile sensitivity filter is
# applied per wave in Fig3 and that threshold definition is kept identical
# here; the waves are simply concatenated before being drawn.
#
# 1e-3 floor (was 1e-8): below this the colour scale was spending most of its
# dynamic range on distinctions that are not interpretable. Two things happen
# down there. The extreme tail is dominated by the float64 precision floor
# (1-CDF underflows for the largest NFC -- 724 rows come back as exactly 0.0
# and cannot be placed on a log scale at all). And well before that, a p-value
# of 1e-5 versus 1e-7 carries no usable meaning here: both say "far past any
# threshold we apply", and the eps-corrected null's own grid resolution does
# not support separating them. Flooring at 1e-3 spends the colormap on the
# 1e-3..1 range where the population actually lives and where the p<0.01
# suspect threshold sits.
#
# 5.15% of units now fall at or below the floor and share the colormap's end
# colour (it was 2.16% at 1e-8).
PVAL_FLOOR = 1e-3

# Truncated at 0.88 rather than the full 0..1 range: viridis' top end is a
# very pale yellow that all but disappears against white at this marker size,
# and that end is where the DENSE bulk of the population lives (p near 1).
# Clipping it back to a mid-green keeps the null cloud visible as a cloud
# while still letting the rare small-p points -- dark purple -- read as
# outliers against it.
PVAL_CMAP = matplotlib.colors.ListedColormap(
    matplotlib.colormaps["viridis"](np.linspace(0.0, 0.88, 256)),
    name="viridis_truncated")

# Half of FP.MS_SMALL (3 pt), which is what these scatters used before.
# `ax.scatter` takes an AREA in pt^2, so halving the marker's diameter means
# squaring the halved diameter here, not halving MS_SMALL**2.
MS_SCATTER = (FP.MS_SMALL / 2) ** 2

# Duration colormap: plasma, not viridis, so the duration-coloured PDF can't be
# mistaken for the p-value one at a glance. Truncated at 0.9 for the same
# reason PVAL_CMAP is -- plasma's top end is a pale yellow that vanishes
# against white at this marker size.
DURATION_CMAP = matplotlib.colors.ListedColormap(
    matplotlib.colormaps["plasma"](np.linspace(0.0, 0.9, 256)),
    name="plasma_truncated")

# What the point colour encodes, one entry per output PDF. `small_on_top` sets
# which points are drawn on top where they overlap: smallest value last for
# p-values (see plot_scatter); otherwise (duration, which has no end that
# matters more than the other) a fixed-seed shuffle, so no one duration
# systematically hides the rest. vmin/vmax of None are taken from the data
# (the full joined frame, so every axes shares one scale).
COLOR_SPECS = {
    "pvalue": dict(
        col="p_value", cmap=PVAL_CMAP, vmin=PVAL_FLOOR, vmax=1.0,
        label="p-value (1F)", small_on_top=True,
        # The bottom end is a clip, not a value (see PVAL_FLOOR) -- say so on
        # the tick rather than letting it read as an exact p of PVAL_FLOOR.
        # Derived from PVAL_FLOOR so the ticks can't drift out of range again
        # (they were still the 1e-8 floor's, and matplotlib silently drops
        # out-of-range ticks, so the clip label never showed).
        ticks=[PVAL_FLOOR, 1e-2, 0.1, 1.0],
        ticklabels=[rf"$\leq${PVAL_FLOOR:g}", "0.01", "0.1", "1"],
        out_name="Fig3_modulation_depth.pdf"),
    "duration": dict(
        # T: the (rec, freq) group's analysis duration, read from the NWB
        # fourier_group_results table -- the same T fit_fourier_sig used.
        col="T", cmap=DURATION_CMAP, vmin=None, vmax=None,
        label="Spike-train duration T (s)", small_on_top=False,
        ticks=None, ticklabels=None,
        out_name="Fig3_modulation_depth_duration.pdf"),
}
DRAW_ORDER_SEED = 0


def _resolve_color_spec(name, df):
    """COLOR_SPECS[name] with any data-derived vmin/vmax filled in from `df`,
    plus the LogNorm every scatter and the colorbar share."""
    spec = dict(COLOR_SPECS[name])
    vals = df[spec["col"]].values
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if spec["vmin"] is None:
        spec["vmin"] = float(vals.min())
    if spec["vmax"] is None:
        spec["vmax"] = float(vals.max())
    spec["norm"] = matplotlib.colors.LogNorm(vmin=spec["vmin"], vmax=spec["vmax"])
    return spec


def resolve_yaml_path(name: str, experiments_dir: Path = None):
    """`experiments/{name}.yml` when that exists and really declares `name`,
    otherwise the YAML found by scanning for a matching `name:` field.

    A handful of YAMLs' filenames don't match their own internal `name:` --
    `experiments/Q146_20230815.yml` declares `name: "Q146_20230815_g0"`, which
    is what `data/Q146_20230815_g0.nwb` and every parquet row are keyed by.
    Same two-step resolution `verify_outputs._load_cfg_by_name` does, for the
    same reason.
    """
    experiments_dir = experiments_dir or _EXPERIMENTS_DIR
    direct = experiments_dir / f"{name}.yml"
    if direct.exists() and load_experiment(direct).name == name:
        return direct
    for yml_path in sorted(experiments_dir.glob("*.yml")):
        if load_experiment(yml_path).name == name:
            return yml_path
    raise FileNotFoundError(f"no experiment YAML with name '{name}' in {experiments_dir}")


def discover_ephys_experiments(experiments_dir: Path):
    """Every experiment whose paradigm writes a `Units` table and that has an
    NWB on disk, as a list of `cfg.name`. Same YAML-scan-by-predicate pattern
    as fig4.discover_pigeon_hp_experiments, rather than a hardcoded list, so
    this tracks `experiments/` automatically.
    """
    found = []
    for yml_path in sorted(experiments_dir.glob("*.yml")):
        cfg = load_experiment(yml_path)
        if cfg.paradigm not in _EPHYS_PARADIGMS:
            continue
        nwb_path = _REPO_ROOT / "data" / f"{cfg.name}.nwb"
        if not nwb_path.exists():
            print(f"  WARNING: {cfg.name} has no {nwb_path.name} -- skipping "
                  f"(run pipeline/processing.py + analysis.py for it)")
            continue
        found.append(cfg.name)
    if not found:
        raise ValueError(f"No ephys experiment YAMLs with an NWB found in {experiments_dir}")
    return found


def fit_psth_sinusoids(phase, unit_index, n_units, T, n_bins=PSTH_N_BINS):
    """Least-squares fit of `m0 + a*cos(theta) + b*sin(theta)` to each unit's
    phase PSTH, for every unit of one (rec, freq) group at once.

    `phase` (radians, 0..2*pi) and `unit_index` (0..n_units-1) are flat
    per-spike arrays; `T` is the group's analysis duration in seconds.

    Returns `(mean_rate, amp)`, both length-`n_units` arrays in spikes/s:
    the fitted constant term (which for a complete set of uniform bins is
    exactly the unit's mean rate, spk_count/T) and the fitted sinusoid
    amplitude `hypot(a, b)`.

    Done as one matrix product over a (n_units, n_bins) PSTH rather than a
    per-unit curve fit: the model is linear in (m0, a, b) once the frequency
    is fixed, so the normal equations have a single closed-form solution
    shared by every unit, and the design matrix's pseudo-inverse can be built
    once for the whole group.
    """
    edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    # np.clip, not a modulo: a phase of exactly 2*pi (possible when a spike
    # lands precisely on a period boundary) would otherwise index bin n_bins
    # and fall off the end of the histogram.
    bin_index = np.clip((phase / (2 * np.pi) * n_bins).astype(np.int64), 0, n_bins - 1)
    counts = np.zeros((n_units, n_bins))
    np.add.at(counts, (unit_index, bin_index), 1.0)
    # Counts -> spikes/s: each bin covers (1/freq)/n_bins seconds of each of
    # the T*freq cycles, i.e. T/n_bins seconds of observation in total.
    psth = counts * (n_bins / T)

    centers = (edges[:-1] + edges[1:]) / 2
    X = np.column_stack([np.ones(n_bins), np.cos(centers), np.sin(centers)])
    coef = psth @ np.linalg.pinv(X).T          # (n_units, 3)
    return coef[:, 0], np.hypot(coef[:, 1], coef[:, 2])


def compute_experiment_mod_depth(experiment: str, data_dir=None, validate=False):
    """Fit a PSTH sinusoid to every (rec, freq, unit) of one ephys experiment.

    Returns a DataFrame with one row per (rec, freq, unit) and columns
    experiment / rec / freq / id / spk_count / T / mean_rate_hz / amp_hz /
    mod_depth (+ amp_fou_hz when `validate`).

    The unit population is whatever `build_modulation_frame(good_only=cfg.good)`
    yields -- the same call, with the same default `min_spikes`, that
    `analysis_stages/simple.py` and `analysis_stages/multistim.py` make -- so
    these rows line up 1:1 with the experiment's rows in all_fourier_df.parquet
    instead of being a differently-filtered population that merely resembles
    it.

    Groups with no `fourier_group_results` 1F row are skipped rather than
    fitted on a locally-derived duration: those are exactly the categories the
    analysis stage deliberately does not Fourier-analyse (oddball's
    long_on/long_off/long_both, see analysis_stages/multistim.py), so they have
    no `sens` to be plotted against either.
    """
    data_dir = Path(data_dir) if data_dir is not None else _REPO_ROOT / "data"
    cfg = load_experiment(resolve_yaml_path(experiment))
    io_r, nwbfile = nwb_io.read_nwbfile(str(data_dir / f"{experiment}.nwb"))
    modulation_df = nwb_io.build_modulation_frame(nwbfile, good_only=cfg.good)
    group_df, unit_df = nwb_io.read_fourier_group_and_unit_tables(nwbfile)
    io_r.close()

    onef = group_df.loc[group_df["harmonic"] == "1F"]
    T_by_group = {(rec, float(frq)): float(T)
                  for rec, frq, T in zip(onef["rec"], onef["frequency"], onef["T"])}
    # Keyed by the group table's own LABEL, not by position within `onef`:
    # unit_df.group_1f_index refers to a row of the full group_df (1F and 2F
    # interleaved), so an enumerate() over the 1F-only view would point at the
    # wrong group.
    group_index = {(rec, float(frq)): label
                   for label, (rec, frq) in zip(onef.index, zip(onef["rec"], onef["frequency"]))}

    rows = []
    for (rec, frq), sub in modulation_df.groupby(["rec", "freq"], sort=False):
        frq = float(frq)
        T = T_by_group.get((rec, frq))
        if T is None or frq <= 0 or T <= 0:
            continue
        uids, unit_index = np.unique(sub["id"].values, return_inverse=True)
        period_s = 1.0 / frq
        # Fold on the NOMINAL period rather than reusing the frame's own
        # `phase` column: `phase`'s meaning varies by the epoch's
        # phase_method (interpolated between real Schmitt crossings for
        # "crossings"/"gratings", pinned to 0.0 outside the crossing span,
        # and left UNNORMALISED for oddball's
        # "stitched_crossings_unnorm"), whereas fit_fourier_sig -- which
        # produced the `sens` on the x axis -- always evaluates its
        # coefficient at exactly this nominal `freq` off the `spk` times. One
        # uniform rule across paradigms, and the one that matches the x axis.
        phase = np.mod(sub["spk"].values, period_s) / period_s * 2 * np.pi
        mean_rate, amp = fit_psth_sinusoids(phase, unit_index, len(uids), T)
        spk_count = np.bincount(unit_index, minlength=len(uids)).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(mean_rate > 0, amp / mean_rate, np.nan)
        row = pd.DataFrame({
            "experiment": experiment, "rec": rec, "freq": frq, "id": uids,
            "spk_count": spk_count, "T": T,
            "mean_rate_hz": mean_rate, "amp_hz": amp, "mod_depth": depth,
        })
        if validate:
            # 2*|fou0| is the amplitude of the same sinusoid computed from the
            # exact spike times instead of the binned PSTH (fou0 is already
            # divided by T, so it is a rate amplitude in spikes/s).
            gi = group_index[(rec, frq)]
            u = unit_df.loc[unit_df["group_1f_index"] == gi].set_index("unit_id")
            fou = np.full(len(uids), np.nan)
            present = np.array([uid in u.index for uid in uids])
            if present.any():
                sel = u.loc[uids[present]]
                fou[present] = 2 * np.abs(sel["fou0_real"].values
                                          + 1j * sel["fou0_imag"].values)
            row["amp_fou_hz"] = fou
        rows.append(row)

    if not rows:
        print(f"  {experiment}: no analysed (rec, freq) groups -- nothing fitted")
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    msg = f"  {experiment}: {len(out)} (rec, freq, unit) PSTH fits"
    if validate:
        ok = np.isfinite(out["amp_fou_hz"]) & (out["amp_fou_hz"] > 0)
        if ok.any():
            ratio = (out.loc[ok, "amp_hz"] / out.loc[ok, "amp_fou_hz"]).values
            msg += (f" | amp_psth/2|fou0|: median {np.median(ratio):.5f}, "
                    f"5-95% [{np.percentile(ratio, 5):.4f}, {np.percentile(ratio, 95):.4f}]"
                    f" (expected ~{np.sinc(1 / PSTH_N_BINS):.5f})")
    print(msg)
    return out


def _mod_depth_worker(task):
    experiment, data_dir, validate = task
    return compute_experiment_mod_depth(experiment, data_dir=data_dir, validate=validate)


def build_mod_depth_table(experiments, data_dir=None, workers=1, validate=False):
    """`compute_experiment_mod_depth` over every ephys experiment, concatenated.

    Parallel across experiments (each opens its own NWB and is independent),
    same `--workers` convention as pipeline/processing.py and fig4.py.
    """
    tasks = [(e, str(data_dir) if data_dir else None, validate) for e in experiments]
    if workers > 1:
        with Pool(min(workers, len(tasks))) as pool:
            frames = pool.map(_mod_depth_worker, tasks)
    else:
        frames = [_mod_depth_worker(t) for t in tasks]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise ValueError("No PSTH fits produced from any ephys experiment")
    table = pd.concat(frames, ignore_index=True)
    dup = table.duplicated(subset=["rec", "freq", "id"]).sum()
    if dup:
        # The merge below keys on (rec, freq, id) without the experiment name,
        # because all_fourier_df.parquet carries no experiment column. Rec
        # names embed a date and a wall-clock timestamp so they are unique in
        # practice -- but a collision would silently duplicate plotted points,
        # so it is checked rather than assumed.
        raise ValueError(
            f"{dup} (rec, freq, id) keys occur in more than one experiment -- "
            f"the parquet merge key is ambiguous and needs the experiment name "
            f"threading through aggregate.py first."
        )
    return table


def load_mod_depth_table(experiments, data_dir=None, workers=1, recompute=False,
                          validate=False):
    if MOD_DEPTH_CACHE.exists() and not recompute:
        print(f"Loading cached PSTH fits from {MOD_DEPTH_CACHE}")
        return pd.read_parquet(MOD_DEPTH_CACHE)
    print(f"Fitting PSTH sinusoids across {len(experiments)} ephys experiments "
          f"({PSTH_N_BINS} phase bins) ...")
    table = build_mod_depth_table(experiments, data_dir=data_dir, workers=workers,
                                   validate=validate)
    MOD_DEPTH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(MOD_DEPTH_CACHE, index=False)
    print(f"Saved {MOD_DEPTH_CACHE} ({len(table)} rows)")
    return table


def attach_mod_depth(all_fourier_df, mod_depth_table):
    """Inner-join the PSTH fits onto all_fourier_df on (rec, freq, id).

    An inner join is what restricts the figure to ephys species: mouse/owl and
    zebrafish/medaka rows have no PSTH fit to join to and drop out here (see
    the module docstring), and so does any ephys row whose experiment has no
    NWB on disk.
    """
    before = len(all_fourier_df)
    merged = all_fourier_df.merge(
        mod_depth_table[["experiment", "rec", "freq", "id", "mean_rate_hz",
                          "amp_hz", "mod_depth", "T"]],
        on=["rec", "freq", "id"], how="inner", validate="one_to_one")
    kept_species = sorted(merged["species"].unique())
    lost = (all_fourier_df.groupby("species").size()
            - merged.groupby("species").size().reindex(
                all_fourier_df.groupby("species").size().index).fillna(0)).astype(int)
    print(f"Joined PSTH fits: {before} parquet rows -> {len(merged)} "
          f"({', '.join(kept_species)})")
    print("  rows without a PSTH fit, by species:")
    for sp, n in lost.sort_values(ascending=False).items():
        if n:
            print(f"    {sp}: {n}")
    if merged.empty:
        raise ValueError("No parquet row joined to a PSTH fit -- check the cache")
    return merged


def _add_threshold_line(ax, plotted, x_col, y_col):
    """Slope -1 reference line: the modulation depth a unit of a given `sens`
    would need to reach the THRESHOLD_PERCENTILE point of its own null
    distribution.

    The identity is exact and is why this is a straight line on log-log axes.
    With `fou0` the on-frequency coefficient and `sigma` the off-frequency
    noise scale, NFC = |fou0|/sigma and sens = spk_count/T/(2*sigma), while
    the fitted amplitude is amp = 2*|fou0| and the mean rate is spk_count/T.
    So depth = amp/mean_rate = NFC/sens, and NFC = NFC_crit maps to
    depth = NFC_crit / sens.

    Only drawn on the sensitivity-vs-relative-depth page: on the absolute
    page the same threshold is amp = NFC_crit * mean_rate / sens, a different
    line per firing rate rather than one curve, and the firing-rate pages
    have no sensitivity axis for it to be a function of.

    `eps` is taken as the median over the plotted rows' own `Q` (via
    `statistics.eps_from_Q`, the same conversion Fig2 uses), since Q -- and
    hence the finite-sample correction to the null -- varies by recording.
    """
    if (x_col, y_col) != ("sens", "mod_depth") or plotted is None or plotted.empty:
        return
    qs = plotted["Q"]
    if not np.isfinite(qs).any():
        return
    eps = statistics.eps_from_Q(float(np.nanmedian(qs)))
    nfc_crit = statistics.inverse_Rayleigh_CDF(THRESHOLD_PERCENTILE, eps=eps)
    xlim = ax.get_xlim()
    xx = np.geomspace(*xlim)
    ax.plot(xx, nfc_crit / xx, "--", color=FP.COLOR_NULL, linewidth=FP.LW_REFERENCE,
            zorder=0, label=f"{THRESHOLD_PERCENTILE:.0%} null")
    ax.set_xlim(xlim)


def plot_scatter(waves, ax, x_col, y_col, color_spec, percentile=None,
                  sens_col="sens", pval_col="p_value"):
    """One axes: `x_col` vs `y_col`, every point coloured by
    `color_spec["col"]` on a log scale (see COLOR_SPECS).

    Rows with a non-finite `pval_col` are dropped whatever the colouring, so
    the p-value and duration versions of a panel draw exactly the same points.

    `percentile`, when given, is applied PER WAVE before the waves are
    concatenated -- quadrant D's "top 10% most sensitive" threshold is
    computed within each occurrence wave in Fig3, and that definition is kept
    bit-for-bit here even though wave identity no longer affects the drawing.

    Returns the single concatenated frame actually plotted (or None if
    nothing survived filtering), so the caller can derive the threshold
    line's `eps` from exactly those rows.
    """
    kept = []
    for wave_df in waves:
        if len(wave_df) == 0:
            continue
        if percentile is not None:
            pct = np.percentile(wave_df[sens_col], percentile)
            wave_df = wave_df.loc[wave_df[sens_col] > pct]
        if len(wave_df):
            kept.append(wave_df)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if not kept:
        return None
    df = pd.concat(kept, ignore_index=True)
    # Both axes are log-scaled, so a non-positive value has no place on them.
    # These are genuine degenerate units (a flat PSTH fits amplitude 0), not
    # missing data, and matplotlib would silently drop them anyway --
    # filtered explicitly so the counts match what is drawn.
    color_col = color_spec["col"]
    df = df.loc[(df[x_col] > 0) & (df[y_col] > 0)
                & np.isfinite(df[pval_col]) & np.isfinite(df[color_col])]
    if df.empty:
        return None
    if color_spec["small_on_top"]:
        # Smallest p drawn LAST, hence on top. Without this the handful of
        # genuinely significant units would be buried under the tens of
        # thousands of near-p=1 points they overlap, which is precisely the
        # comparison the colour encoding exists to show.
        df = df.sort_values(color_col, ascending=False)
    else:
        rng = np.random.default_rng(DRAW_ORDER_SEED)
        df = df.iloc[rng.permutation(len(df))]
    ax.scatter(df[x_col].values, df[y_col].values,
               c=np.clip(df[color_col].values, color_spec["vmin"], color_spec["vmax"]),
               cmap=color_spec["cmap"],
               norm=color_spec["norm"],
               s=MS_SCATTER, alpha=FP.ALPHA_TRACE, linewidths=0,
               rasterized=True)
    # Deliberately NOT statistics.nestle_labels, which fig3.plot_uniform_p
    # uses to tuck the x label in at the axis's left end: that works there
    # because the axis carries exactly two ticks ("0" and the neuron count),
    # but a log-scaled sensitivity axis has a decade label at the left edge
    # for the nestled label to collide with.
    return df


# Same four conditions, quadrant slots and panel letters as fig3.plot_fig3 --
# only the first title differs, because a spike PSTH does not exist for every
# species (see the module docstring).
CONDITIONS = [
    ("All ephys-species neurons", None, None),
    ("All pigeon neurons", lambda df: df.species == "Pigeon", None),
    ("Pigeon hippocampus neurons", lambda df: (df.area == "HP") & (df.species == "Pigeon"), None),
    ("Top 10% most sensitive neurons in pigeon HP",
     lambda df: (df.area == "HP") & (df.species == "Pigeon"), 90),
]
QUADRANT_SLOTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
PANEL_LETTERS = "ABCD"

AXIS_LABELS = {
    "sens": "Sensitivity",
    "mod_depth": "Modulation depth (amplitude / mean rate)",
    "amp_hz": "Modulation amplitude (spikes/s)",
    "mean_rate_hz": "Firing rate (spikes/s)",
}


def plot_page(waves, pos_control_waves, x_col, y_col, suptitle, color_spec):
    """One page: Fig3's 2x2 quadrant grid, each quadrant a Magnetic /
    Visual-Audio pair of `x_col`-vs-`y_col` scatter axes. `waves` /
    `pos_control_waves` are the two populations' `_split_into_waves` output.
    """
    xlabel, ylabel = AXIS_LABELS[x_col], AXIS_LABELS[y_col]

    font = {"family": FP.FONT_FAMILY, "size": FP.FS_BODY}
    matplotlib.rc("font", **font)

    fig = plt.figure(figsize=(FP.FIGSIZE_FIG3[0] * 1.4, FP.FIGSIZE_FIG3[1] * 1.8))
    # right=0.93, not Fig3's 0.98: the remaining strip holds the shared
    # p-value colorbar added at the end of this function.
    # top=0.86 / hspace=0.45 leave room for the quadrant headers (placed 0.04
    # above each quadrant) to clear both the suptitle and the row above's
    # x labels.
    outer = fig.add_gridspec(2, 2, wspace=0.45, hspace=0.45,
                              left=0.08, right=0.93, top=0.86, bottom=0.07)

    for (title, cond_filter, percentile), (orow, ocol), letter in \
            zip(CONDITIONS, QUADRANT_SLOTS, PANEL_LETTERS):
        # 1x2 rather than Fig3's 2x2: the p-value/q-value row split has no
        # analogue here, there is one scatter per population.
        inner = outer[orow, ocol].subgridspec(1, 2, wspace=0.08)
        ax_neg = fig.add_subplot(inner[0, 0])
        ax_pos = fig.add_subplot(inner[0, 1])
        # Both axes share x AND y within the quadrant (Fig3 shares y only
        # within a row, and x only within a column): here the two populations
        # are measured in the same two units on both axes, so putting them on
        # one scale is what makes the magnetic cloud readable against the
        # positive-control one. Sharing stays scoped to the quadrant, as in
        # Fig3 -- the four conditions have very different populations and
        # shouldn't force each other's ranges.
        ax_neg.sharex(ax_pos)
        ax_neg.sharey(ax_pos)
        ax_pos.tick_params(labelleft=False)

        neg_subset = waves if cond_filter is None else [w.loc[cond_filter(w)] for w in waves]
        pos_subset = pos_control_waves if cond_filter is None else \
            [w.loc[cond_filter(w)] for w in pos_control_waves]

        neg_plotted = plot_scatter(neg_subset, ax_neg, x_col, y_col, color_spec,
                                   percentile=percentile)
        pos_plotted = plot_scatter(pos_subset, ax_pos, x_col, y_col, color_spec,
                                   percentile=percentile)
        # After both scatters, so the shared x limits are final.
        _add_threshold_line(ax_neg, neg_plotted, x_col, y_col)
        _add_threshold_line(ax_pos, pos_plotted, x_col, y_col)

        ax_neg.set_title("Magnetic", fontsize=FP.FS_TITLE)
        ax_pos.set_title("Visual/Audio", fontsize=FP.FS_TITLE)
        ax_neg.set_ylabel(ylabel)
        ax_neg.set_xlabel(xlabel)
        ax_pos.set_xlabel(xlabel)
        if ax_pos.get_legend_handles_labels()[0]:
            ax_pos.legend(fontsize=FP.FS_LEGEND, loc="upper right", frameon=False)

        quad_pos = outer[orow, ocol].get_position(fig)
        fig.text((quad_pos.x0 + quad_pos.x1) / 2, quad_pos.y1 + 0.04, title,
                  ha="center", va="bottom", fontsize=FP.FS_BODY + 1, fontweight="bold")
        fig.text(quad_pos.x0, quad_pos.y1 + 0.04, letter,
                  ha="left", va="bottom", fontfamily="arial", fontsize=12, fontweight="bold")

    # One colorbar for all eight axes: hue is the same quantitative encoding
    # everywhere (see COLOR_SPECS), so a per-axes bar would repeat itself
    # eight times. Built from a standalone ScalarMappable rather than from one
    # of the scatters, so it doesn't matter which axes happened to be drawn
    # last or whether any given one ended up empty.
    cax = fig.add_axes([0.955, 0.33, 0.013, 0.34])
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=color_spec["norm"], cmap=color_spec["cmap"]),
        cax=cax)
    cbar.set_label(color_spec["label"], fontsize=FP.FS_TITLE)
    cbar.ax.tick_params(labelsize=FP.FS_LEGEND)
    if color_spec["ticks"] is not None:
        cbar.set_ticks(color_spec["ticks"])
        cbar.set_ticklabels(color_spec["ticklabels"])

    fig.suptitle(suptitle, fontsize=FP.FS_BODY + 2, fontweight="bold", y=0.975)
    return fig


def plot_fig3_modulation_depth(all_fourier_df, out_dir: Path, color="pvalue"):
    """All four pages into one PDF, points coloured per COLOR_SPECS[color]
    (see the module docstring)."""
    color_spec = _resolve_color_spec(color, all_fourier_df)
    all_neg_res, all_pos_control, _ = statistics.get_poscontrols_negresults(all_fourier_df)
    waves = _split_into_waves(all_neg_res)
    pos_control_waves = _split_into_waves(all_pos_control)
    pages = [
        ("sens", "mod_depth", "Detection sensitivity vs. relative modulation depth"),
        ("sens", "amp_hz", "Detection sensitivity vs. absolute modulation amplitude"),
        ("mean_rate_hz", "sens", "Detection sensitivity vs. firing rate"),
        ("mean_rate_hz", "mod_depth", "Relative modulation depth vs. firing rate"),
    ]
    out_path = out_dir / color_spec["out_name"]
    with PdfPages(out_path) as pdf:
        for x_col, y_col, suptitle in pages:
            fig = plot_page(waves, pos_control_waves, x_col, y_col, suptitle,
                            color_spec)
            pdf.savefig(fig, bbox_inches="tight", dpi=FP.DPI)
            if not in_notebook:
                plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate the Fig3-layout sensitivity vs modulation-depth scatter figure")
    parser.add_argument("--out-dir", default=FP.OUT_DIR, help="Output directory for PDFs")
    parser.add_argument("--parquet", default=FP.PARQUET_PATH,
                        help=f"Path to all_fourier_df.parquet (default: {FP.PARQUET_PATH})")
    parser.add_argument("--data-dir", default=FP.DATA_DIR,
                        help="Directory holding the per-experiment .nwb files")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Explicit experiment names (default: auto-discover every ephys YAML)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for the per-experiment PSTH fits")
    parser.add_argument("--recompute", action="store_true",
                        help="Refit the PSTH sinusoids instead of reusing the cache")
    parser.add_argument("--validate", action="store_true",
                        help="Also compare each fitted amplitude against 2*|fou0| from the NWB")
    parser.add_argument("--color", choices=[*COLOR_SPECS, "both"], default="both",
                        help="What the point colour encodes; one PDF per choice (default: both)")
    args = parser.parse_args([] if in_notebook else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = args.experiments or discover_ephys_experiments(_EXPERIMENTS_DIR)
    mod_depth_table = load_mod_depth_table(experiments, data_dir=args.data_dir,
                                            workers=args.workers,
                                            recompute=args.recompute,
                                            validate=args.validate)

    print(f"Loading {args.parquet} ...")
    all_fourier_df = pd.read_parquet(args.parquet)
    all_fourier_df = attach_mod_depth(all_fourier_df, mod_depth_table)
    colors = list(COLOR_SPECS) if args.color == "both" else [args.color]
    for color in colors:
        plot_fig3_modulation_depth(all_fourier_df, out_dir, color=color)


if __name__ == "__main__":
    if in_notebook:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    main()
