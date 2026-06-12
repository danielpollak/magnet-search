"""
Engert GCaMP diagnostic plots.

plot_engert_diagnostics  — single multi-page PDF:
  Page 1: c-hat histogram vs Rayleigh null + inset
  Page 2: Fourier coefficient spectrum (real + imaginary vs frequency, subsampled cells)
  Page 3: Fluorescence heatmap (cells sorted by c-hat descending, rasterized)
  Page 4: Cell mask FOV — excluded cells gray, included cells colored by c-hat
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from magpyneto2.statistics import draw_hist, inset_hist
from pipeline.diagnostics.analysis import _subsample_units, _rasterize_ax


def plot_engert_diagnostics(cfg, F, stat, fourier_df, freq_win,
                             onfreq_pow_l, offfreq_pow_l, save_dir):
    """Write a multi-page PDF of GCaMP analysis diagnostics.

    Parameters
    ----------
    cfg            : ExperimentConfig
    F              : np.ndarray  (n_cells, n_frames) fluorescence traces
    stat           : np.ndarray  suite2p stat array (filtered)
    fourier_df     : pd.DataFrame  with columns [rr, freq, rec, ...]
    freq_win       : np.ndarray  off-frequencies (Hz)
    onfreq_pow_l   : np.ndarray  (n_cells,) complex on-frequency coefficients
    offfreq_pow_l  : list of np.ndarray  per-cell off-frequency complex coefficients
    log_dict       : dict  {(rec, freq): {...}}  (same format as NPIX)
    save_dir       : Path-like
    """
    save_dir = Path(save_dir)
    out_path = save_dir / f"{cfg.name}_analysis_diagnostics.pdf"
    freq = cfg.analysis.f
    rr = fourier_df["rr"].values

    with PdfPages(out_path) as pdf:

        # ── Page 1: c-hat histogram ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle(f"{cfg.name}  |  {freq} Hz  —  c-hat distribution", fontsize=9)
        if len(rr) >= 2:
            vals, bins = draw_hist(rr, ax, xlim=9, inset=False)
            inset_hist(ax, vals, bins)
            ax.set_title(f"N = {len(rr)} cells")
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

        # ── Page 3: Fluorescence heatmap sorted by c-hat ─────────────────────
        sort_idx = np.argsort(rr)[::-1]
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
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Pages 4 & 5: cell mask FOV + P(iscell) ECDF ─────────────────────
        try:
            import os
            suite2p_dir = os.path.join(cfg.session_path, "suite2p", "plane0")
            stat_all   = np.load(os.path.join(suite2p_dir, "stat.npy"),   allow_pickle=True)
            iscell_all = np.load(os.path.join(suite2p_dir, "iscell.npy"), allow_pickle=True)
            ops        = np.load(os.path.join(suite2p_dir, "ops.npy"),    allow_pickle=True).item()
            Ly, Lx     = ops["Ly"], ops["Lx"]

            # stat (filtered + flatline-removed) matches rr 1:1.
            # Match back to stat_all by centroid to identify excluded cells.
            included_meds = {tuple(s["med"]) for s in stat}
            rr_norm    = (rr - rr.min()) / max(rr.max() - rr.min(), 1e-10)
            cmap_cells = plt.cm.plasma

            img = np.zeros((Ly, Lx, 3), dtype=np.float32)

            # All cells not in the final included set → dim gray
            for s in stat_all:
                if tuple(s["med"]) not in included_meds:
                    img[s["ypix"], s["xpix"], :] = 0.25

            # Included cells colored by c-hat (stat order == rr order)
            for rank, s in enumerate(stat):
                color = np.array(cmap_cells(rr_norm[rank])[:3], dtype=np.float32)
                img[s["ypix"], s["xpix"], :] = color

            fig, ax = plt.subplots(figsize=(8, 8 * Ly / Lx))
            n_excl = len(stat_all) - len(stat)
            fig.suptitle(
                f"{cfg.name}  |  cell masks  —  {len(stat)} included, "
                f"{n_excl} excluded (gray)  |  iscell_threshold={cfg.iscell_threshold}",
                fontsize=8)
            ax.imshow(img, aspect="equal", interpolation="none", rasterized=True)
            ax.axis("off")
            sm = plt.cm.ScalarMappable(
                cmap=cmap_cells,
                norm=plt.Normalize(vmin=rr.min(), vmax=rr.max()))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="c-hat", shrink=0.6, pad=0.02)
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            # ── Page 5: P(iscell) × npix joint histogram with ECDF marginals ──
            p_iscell  = iscell_all[:, 1]
            npix_vals = np.array([s["npix"] for s in stat_all])
            included_mask = (p_iscell >= cfg.iscell_threshold) & (npix_vals >= cfg.npix_threshold)
            n_included = included_mask.sum()

            fig = plt.figure(figsize=(8, 7))
            gs  = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                                   hspace=0.05, wspace=0.05)
            ax_main  = fig.add_subplot(gs[1, 0])
            ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
            fig.add_subplot(gs[0, 1]).set_visible(False)   # empty corner

            fig.suptitle(
                f"{cfg.name}  |  P(iscell) × npix  —  "
                f"{n_included} / {len(p_iscell)} ROIs pass both thresholds",
                fontsize=9)

            # Main scatter
            ax_main.scatter(npix_vals[~included_mask], p_iscell[~included_mask],
                            s=3, alpha=0.35, color="gray", rasterized=True, label="excluded")
            ax_main.scatter(npix_vals[included_mask], p_iscell[included_mask],
                            s=3, alpha=0.6, color="steelblue", rasterized=True, label="included")
            ax_main.axhline(cfg.iscell_threshold, color="crimson",   lw=1.2, ls="--",
                            label=f"iscell ≥ {cfg.iscell_threshold}")
            ax_main.axvline(cfg.npix_threshold,   color="darkorange", lw=1.2, ls="--",
                            label=f"npix ≥ {cfg.npix_threshold}")
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

