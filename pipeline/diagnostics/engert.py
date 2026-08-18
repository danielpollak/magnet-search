"""
Engert GCaMP diagnostic plots.

plot_engert_diagnostics  — single multi-page PDF:
  Page 1: c-hat histogram vs Rayleigh null + inset (1F)
  Page 2: Fourier coefficient spectrum (real + imaginary vs frequency, subsampled cells) (1F)
  Page 3: 2F c-hat histogram (skipped for medaka, which has no 1F/2F harmonic
          pairing, and for 1F frequencies above half-Nyquist, where
          analysis_stages/engert.py skips 2F entirely)
  Page 4: 2F Fourier coefficient spectrum (same skip conditions as Page 3)
  Page 5: Fluorescence heatmap (cells sorted by c-hat descending, rasterized),
          with a period-duration scale bar so cycles can be counted by eye
  Page 6: Cell mask FOV — excluded cells gray, included cells colored by c-hat
  Page 7: P(iscell) x npix joint histogram with ECDF marginals
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from magpyneto2.statistics import draw_hist, inset_hist, get_epsilon
from pipeline.diagnostics.analysis import _subsample_units, _rasterize_ax


def plot_engert_diagnostics(cfg, F, fourier_df, freq_win,
                             onfreq_pow_l, offfreq_pow_l, save_dir,
                             roi_df=None, included_mask=None, imaging_dims=None,
                             freq_win_2f=None, onfreq_pow_2f=None,
                             offfreq_pow_2f=None, Q_2f=None):
    """Write a multi-page PDF of GCaMP analysis diagnostics.

    Parameters
    ----------
    cfg            : ExperimentConfig
    F              : np.ndarray  (n_cells, n_frames) fluorescence traces
                     -- already filtered to this analysis run's population
                     (iscell/npix threshold + flatline removal)
    fourier_df     : pd.DataFrame  with columns [NFC, freq, rec, ...]
    freq_win       : np.ndarray  off-frequencies (Hz), 1F
    onfreq_pow_l   : np.ndarray  (n_cells,) complex on-frequency coefficients, 1F
    offfreq_pow_l  : list of np.ndarray  per-cell off-frequency complex coefficients, 1F
    save_dir       : Path-like
    roi_df         : pd.DataFrame, optional -- ALL suite2p ROIs (unfiltered),
                     from `nwb_io.read_roi_data` (columns p_iscell, npix, x,
                     y, pixel_mask), in original suite2p row order. Pages 5/6
                     (cell-mask FOV + P(iscell)xnpix ECDF) are skipped if not
                     given.
    included_mask  : np.ndarray of bool, optional, same length as `roi_df` --
                     True for ROIs that survived BOTH the iscell_threshold/
                     npix_threshold mask AND flatline removal (i.e. the exact
                     population behind `F`/`fourier_df` in THIS run). Replaces
                     the old fragile centroid-matching workaround (matching
                     ROIs across a filtered/unfiltered pair by hashing
                     `stat["med"]`, which breaks silently on centroid
                     collisions) with direct index alignment -- also unifies
                     what were two inconsistent thresholds (page 4 implicitly
                     used the flatline-filtered set; page 5 recomputed a
                     separate iscell/npix-only mask with `>=` instead of the
                     analysis stage's own `>`).
    imaging_dims   : (Ly, Lx) tuple, optional, from `nwb_io.get_imaging_dims`.
    freq_win_2f, onfreq_pow_2f, offfreq_pow_2f, Q_2f : optional, the SAME as
                     freq_win/onfreq_pow_l/offfreq_pow_l/(bin count) but for
                     the 2F harmonic -- only meaningful for engert (medaka's
                     magnetic/visual legs are independent groups, not 1F/2F
                     harmonics of each other, so its caller never passes
                     these). Page 3 (2F diagnostics) is skipped if
                     `onfreq_pow_2f` is None.
    """
    save_dir = Path(save_dir)
    out_path = save_dir / f"{cfg.name}_analysis_diagnostics.pdf"
    freq = cfg.analysis.f
    NFC = fourier_df["NFC"].values

    with PdfPages(out_path) as pdf:

        # ── Page 1: c-hat histogram ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle(f"{cfg.name}  |  {freq} Hz  —  c-hat distribution", fontsize=9)
        if len(NFC) >= 2:
            Q = fourier_df["Q"].iloc[0] if "Q" in fourier_df.columns else None
            eps = get_epsilon(Q) if Q else None
            vals, bins = draw_hist(NFC, ax, xlim=9, inset=False, eps=eps)
            inset_hist(ax, vals, bins, eps=eps)
            ax.set_title(f"N = {len(NFC)} cells")
        else:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
        _rasterize_ax(ax)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 2: Fourier coefficient spectrum ─────────────────────────────
        fou_alt = np.array(offfreq_pow_l)   # (C, 2Q-1) complex
        kk = _subsample_units(len(fou_alt), max_units=30)
        fou_alt_sub = fou_alt[kk]
        onfreq_sub  = np.array(onfreq_pow_l)[kk]

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle(f"{cfg.name}  |  {freq} Hz  —  Fourier spectrum (ΔF/F)", fontsize=9)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(kk)))
        for i, (row, c_on) in enumerate(zip(fou_alt_sub, onfreq_sub)):
            axes[0].plot(freq_win, np.real(row), ".", color=colors[i],
                         markersize=0.8, alpha=0.6, rasterized=True)
            axes[1].plot(freq_win, np.imag(row), ".", color=colors[i],
                         markersize=0.8, alpha=0.6, rasterized=True)
        axes[0].plot([freq] * len(onfreq_sub), np.real(onfreq_sub),
                     "r.", markersize=3, label=f"{freq} Hz")
        axes[1].plot([freq] * len(onfreq_sub), np.imag(onfreq_sub),
                     "r.", markersize=3)
        axes[0].set_ylabel("real  component")
        axes[1].set_ylabel("imaginary  component")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[0].legend(fontsize=7)
        for ax in axes:
            _rasterize_ax(ax)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Pages 3 & 4: 2F diagnostics (skipped if 2F wasn't computed -- see
        #    analysis_stages/engert.py's Nyquist check, and medaka, which
        #    never passes these at all) ────────────────────────────────────
        if onfreq_pow_2f is not None:
            NFC_2f_all = fourier_df["2f_NFC"].values if "2f_NFC" in fourier_df.columns else np.array([])
            NFC_2f = NFC_2f_all[~np.isnan(NFC_2f_all)] if len(NFC_2f_all) else NFC_2f_all

            if len(NFC_2f) >= 2:
                fig, ax = plt.subplots(figsize=(7, 5))
                fig.suptitle(f"{cfg.name}  |  {freq * 2} Hz (2F)  —  c-hat distribution",
                             fontsize=9)
                eps_2f = get_epsilon(Q_2f) if Q_2f else None
                vals, bins = draw_hist(NFC_2f, ax, xlim=9, inset=False, eps=eps_2f)
                inset_hist(ax, vals, bins, eps=eps_2f)
                ax.set_title(f"N = {len(NFC_2f)} cells")
                _rasterize_ax(ax)
                fig.tight_layout()
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

            fou_alt_2f = np.array(offfreq_pow_2f)
            kk2 = _subsample_units(len(fou_alt_2f), max_units=30)
            fou_alt_2f_sub = fou_alt_2f[kk2]
            onfreq_2f_sub = np.array(onfreq_pow_2f)[kk2]

            fig, axes = plt.subplots(2, 1, figsize=(10, 6))
            fig.suptitle(f"{cfg.name}  |  {freq * 2} Hz (2F)  —  Fourier spectrum (ΔF/F)",
                         fontsize=9)
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(kk2)))
            for i, (row, c_on) in enumerate(zip(fou_alt_2f_sub, onfreq_2f_sub)):
                axes[0].plot(freq_win_2f, np.real(row), ".", color=colors[i],
                             markersize=0.8, alpha=0.6, rasterized=True)
                axes[1].plot(freq_win_2f, np.imag(row), ".", color=colors[i],
                             markersize=0.8, alpha=0.6, rasterized=True)
            axes[0].plot([freq * 2] * len(onfreq_2f_sub), np.real(onfreq_2f_sub),
                         "r.", markersize=3, label=f"{freq * 2} Hz")
            axes[1].plot([freq * 2] * len(onfreq_2f_sub), np.imag(onfreq_2f_sub),
                         "r.", markersize=3)
            axes[0].set_ylabel("real  component")
            axes[1].set_ylabel("imaginary  component")
            axes[1].set_xlabel("Frequency (Hz)")
            axes[0].legend(fontsize=7)
            for ax in axes:
                _rasterize_ax(ax)
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        # ── Page 5: Fluorescence heatmap sorted by c-hat ─────────────────────
        sort_idx = np.argsort(NFC)[::-1]
        F_sorted = F[sort_idx]
        N_frames = int(120 * (F.shape[1] // 60))
        F_display = F_sorted[:, :N_frames]

        # Normalize each row to [0, 1] for display
        row_min = F_display.min(axis=1, keepdims=True)
        row_max = F_display.max(axis=1, keepdims=True)
        denom = np.where(row_max > row_min, row_max - row_min, 1)
        F_norm = (F_display - row_min) / denom

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f"{cfg.name}  |  {freq} Hz  —  ΔF/F sorted by c-hat (high → low)",
                     fontsize=9)
        im = ax.imshow(F_norm, aspect="auto", cmap="viridis",
                       interpolation="none", rasterized=True)
        ax.set_xlabel("Frame")
        ax.set_ylabel(f"Cell (N={len(F_norm)}, sorted)")
        ax.set_yticks([0, len(F_norm) - 1])
        plt.colorbar(im, ax=ax, label="normalized ΔF/F", shrink=0.6)
        _draw_period_scalebar(ax, freq, cfg.sample_period)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Pages 6 & 7: cell mask FOV + P(iscell) ECDF ─────────────────────
        # Both pages now read directly from the NWB-backed roi_df +
        # included_mask (one consistent population: iscell/npix threshold
        # AND flatline removal, both using the analysis stage's own strict
        # `>` -- previously page 4 used a fragile stat["med"]-centroid-hash
        # match against a freshly re-read stat.npy, and page 5 recomputed a
        # SEPARATE, inconsistent `>=` mask that also didn't account for
        # flatline removal at all) -- see this function's docstring.
        try:
            if roi_df is None or included_mask is None or imaging_dims is None:
                raise ValueError("roi_df/included_mask/imaging_dims not provided")
            Ly, Lx = imaging_dims
            included_mask = np.asarray(included_mask, dtype=bool)
            n_included = int(included_mask.sum())
            n_excl = len(roi_df) - n_included

            NFC_norm = (NFC - NFC.min()) / max(NFC.max() - NFC.min(), 1e-10)
            cmap_cells = plt.cm.plasma

            img = np.zeros((Ly, Lx, 3), dtype=np.float32)

            def _pixels(mask_entries):
                # NWB pixel_mask convention: (x, y, weight) triples -- but
                # pynwb reads each ROI's mask back as a 1-D STRUCTURED
                # (record) array with named fields (x, y, weight), not a
                # plain (n_pixels, 3) array. Fancy column indexing
                # (arr[:, 1]) raises "too many indices for array: array is
                # 1-dimensional, but 2 were indexed" on that structured
                # form -- confirmed on real data. Field-name indexing works
                # for the structured form; fall back to column indexing in
                # case a caller ever passes a plain 2D array instead.
                arr = np.asarray(mask_entries)
                if arr.dtype.names is not None:
                    return arr["y"].astype(int), arr["x"].astype(int)  # (ypix, xpix)
                return arr[:, 1].astype(int), arr[:, 0].astype(int)  # (ypix, xpix)

            # All excluded ROIs → dim gray
            for pixel_mask in roi_df.loc[~included_mask, "pixel_mask"]:
                ypix, xpix = _pixels(pixel_mask)
                img[ypix, xpix, :] = 0.25

            # Included ROIs colored by c-hat -- roi_df[included_mask]'s row
            # order matches NFC's order exactly: both the iscell/npix mask
            # and remove_flatlines's inclusion_inds preserve relative order,
            # and included_mask was built by composing the two the same way
            # the analysis stage itself did.
            for rank, pixel_mask in enumerate(roi_df.loc[included_mask, "pixel_mask"]):
                ypix, xpix = _pixels(pixel_mask)
                color = np.array(cmap_cells(NFC_norm[rank])[:3], dtype=np.float32)
                img[ypix, xpix, :] = color

            fig, ax = plt.subplots(figsize=(8, 8 * Ly / Lx))
            fig.suptitle(
                f"{cfg.name}  |  cell masks  —  {n_included} included, "
                f"{n_excl} excluded (gray)  |  iscell_threshold={cfg.iscell_threshold}",
                fontsize=8)
            ax.imshow(img, aspect="equal", interpolation="none", rasterized=True)
            ax.axis("off")
            sm = plt.cm.ScalarMappable(
                cmap=cmap_cells,
                norm=plt.Normalize(vmin=NFC.min(), vmax=NFC.max()))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="c-hat", shrink=0.6, pad=0.02)
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            # ── Page 7: P(iscell) × npix joint histogram with ECDF marginals ──
            p_iscell  = roi_df["p_iscell"].values
            npix_vals = roi_df["npix"].values

            fig = plt.figure(figsize=(8, 7))
            gs  = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                                   hspace=0.05, wspace=0.05)
            ax_main  = fig.add_subplot(gs[1, 0])
            ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
            fig.add_subplot(gs[0, 1]).set_visible(False)   # empty corner

            fig.suptitle(
                f"{cfg.name}  |  P(iscell) × npix  —  "
                f"{n_included} / {len(p_iscell)} ROIs included "
                f"(thresholds + flatline removal)",
                fontsize=9)

            # Main scatter
            ax_main.scatter(npix_vals[~included_mask], p_iscell[~included_mask],
                            s=3, alpha=0.35, color="gray", rasterized=True, label="excluded")
            ax_main.scatter(npix_vals[included_mask], p_iscell[included_mask],
                            s=3, alpha=0.6, color="steelblue", rasterized=True, label="included")
            ax_main.axhline(cfg.iscell_threshold, color="crimson",   lw=1.2, ls="--",
                            label=f"iscell > {cfg.iscell_threshold}")
            ax_main.axvline(cfg.npix_threshold,   color="darkorange", lw=1.2, ls="--",
                            label=f"npix > {cfg.npix_threshold}")
            ax_main.set_xlabel("npix")
            ax_main.set_ylabel("P(iscell)")
            ax_main.legend(fontsize=7, markerscale=2)

            # Top marginal: npix ECDF
            sorted_npix = np.sort(npix_vals)
            ecdf_npix   = np.arange(1, len(sorted_npix) + 1) / len(sorted_npix)
            ax_top.plot(sorted_npix, ecdf_npix, color="darkorange", lw=1.2, rasterized=True)
            ax_top.axvline(cfg.npix_threshold, color="darkorange", lw=1.2, ls="--")
            ax_top.set_ylabel("ECDF")
            ax_top.tick_params(labelbottom=False)

            # Right marginal: P(iscell) ECDF (rotated — fraction on x, value on y)
            sorted_p = np.sort(p_iscell)
            ecdf_p   = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
            ax_right.plot(ecdf_p, sorted_p, color="steelblue", lw=1.2, rasterized=True)
            ax_right.axhline(cfg.iscell_threshold, color="crimson", lw=1.2, ls="--")
            ax_right.set_xlabel("ECDF")
            ax_right.tick_params(labelleft=False)

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        except Exception as exc:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"cell mask / iscell error:\n{exc}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)


def _draw_period_scalebar(ax, freq, sample_period, loc=(0.03, 0.93)):
    """Draw a labeled horizontal bracket spanning one stimulus period, in the
    same x-units (frames) as the plot it's overlaid on -- lets a reader
    visually count periods against the x-axis without doing arithmetic on
    non-round tick spacing (e.g. counting a non-integer number of 60 s
    periods across 200-frame ticks).

    `loc`: (x0, y) in axes-fraction-for-y / data-for-x coordinates (see
    `ax.get_xaxis_transform()`) -- x0 is where the bar starts, as a fraction
    of the current x-axis range; y is the vertical placement, in axes
    fraction (0 = bottom, 1 = top), so it stays inside the plot regardless
    of the underlying image's data range.
    """
    period_frames = (1.0 / freq) / sample_period
    period_s = 1.0 / freq

    trans = ax.get_xaxis_transform()  # x: data coords, y: axes-fraction coords
    x0 = ax.get_xlim()[0] + loc[0] * (ax.get_xlim()[1] - ax.get_xlim()[0])
    y = loc[1]
    tick = 0.02  # half-height of the end-ticks, in axes fraction

    stroke = [pe.withStroke(linewidth=2.5, foreground="black")]
    for xa, xb in ((x0, x0 + period_frames),):
        ax.plot([xa, xb], [y, y], color="white", lw=1.5, solid_capstyle="butt",
                 transform=trans, path_effects=stroke, zorder=10)
    for xt in (x0, x0 + period_frames):
        ax.plot([xt, xt], [y - tick, y + tick], color="white", lw=1.5,
                 transform=trans, path_effects=stroke, zorder=10)
    ax.text(x0 + period_frames / 2, y + tick + 0.015,
            f"1 period = {period_frames:.1f} frames ({period_s:.1f} s)",
            transform=trans, ha="center", va="bottom", fontsize=7,
            color="white", path_effects=stroke, zorder=10)

