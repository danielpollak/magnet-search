import os
import glob
import sys
# Local imports

from .statistics import *
from .utils import *
from scipy.fft import fft, fftfreq

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tifffile
import matplotlib.cm as cm

import tqdm.auto as tqdm

from sklearn.cluster import KMeans




def min_subtract(F):
    return (F.T - np.min(F, axis=1)).T


def normalize_F(F, stat):
    """Maps F to [0,1] for each unit"""
    F_minned = min_subtract(F)
    maxes = np.max(F_minned, axis=1)
    maxes[maxes == 0] = 1
    F_normed = np.zeros_like(F_minned)
    for i in range(len(F)):
        F_normed[i] = F_minned[i] / maxes[i]

    return F_normed, stat


def functional_cluster(tiff, F, stat, n_clusters=3):
    """
    """
    variance_proj = np.std(tiff[0:20,:,:], axis=0)

    stacked_img = np.stack((variance_proj,)*3, axis=-1)
    # Normalize F
    F_normed, stat = normalize_F(F, stat)


    # get kmeans
    kmeans_kwargs = {"init": "random","n_init": 10, "max_iter": 300, "random_state": 42}

    kmeans = KMeans(n_clusters=n_clusters, **kmeans_kwargs)
    corr_mat = np.corrcoef(F)

    # Clean up nans
    if np.sum(np.isnan(corr_mat)) > 0:
        nanind_x, nanind_y = np.where(np.isnan(corr_mat))
        corr_mat[nanind_x, nanind_y] = 0

    kmeans.fit(corr_mat)

    # Negative image for contrast (lighter is better)
    stacked_img = 255 - normalize_image_values(stacked_img).astype(int)

    for i, cell in enumerate(stat):
        # Edit the cell to a color, eventually label by colormap
        label = kmeans.labels_[i]
        color = 255*np.array(cm.Dark2(label / n_clusters)[:-1])

        stacked_img[cell["ypix"], cell["xpix"], :] = color

    anatomy_fig, anatomy_ax = plt.subplots()
    anatomy_ax.imshow(stacked_img, cmap="viridis")

    """Show raw traces sorted"""
    trace_fig, trace_ax = plt.subplots(figsize=(10,3))
    output_l = []
    for label in np.unique(kmeans.labels_):
        F_l = F_normed[kmeans.labels_ == label]
        output_l.append(F_l)

    # Procedural reversal of mutable object
    # As a result of this reversal, a lot of things become a lot more complicated.
    output_l.reverse()

    # Show all clusters
    im = trace_ax.imshow(np.concatenate(output_l), cmap='viridis', aspect="auto")
    plt.colorbar(im, ax=trace_ax)
    trace_ax.set_ylabel("cell")
    trace_ax.set_xlabel("minutes")

    # Get positions of all yticks, which are at the end of the cluster
    yticks = np.cumsum([len(l) for l in output_l])

    # Last element of yticks, we are going from the bottom up instead of top down
    prev = yticks[::-1][0]
    for cluster_i, ytick in enumerate(np.hstack([yticks[::-1], np.array([0])])):
        color = cm.nipy_spectral(float(cluster_i-1) / n_clusters)
        plt.vlines(5, prev, ytick, color=color, linewidth=10)
        # Decrementing the prev
        prev = ytick

    trace_ax.set_yticks(yticks)
    trace_ax.set_yticklabels(np.arange(len(output_l)))

    trace_ax.set_xticks(trace_ax.get_xticks())
    trace_ax.set_xticklabels(np.round(trace_ax.get_xticks() / 60, 2))

    # ax.tight_layout()
    return anatomy_fig, anatomy_ax, trace_fig, trace_ax


# Unpack exemplar GCaMP data
def load_GEVI(path, tiffpath, sr = 1, length=20):
    tiff = np.empty((20, 700, 700))
    with tifffile.TiffFile(tiffpath) as tiffile:
        for frame_ind in tqdm.tqdm(np.arange(length)):
            tiff[frame_ind] = tiffile.pages[frame_ind].asarray()

    F = np.load(path + r"\F.npy", allow_pickle=True)
    stat = np.load(path + r'\stat.npy', allow_pickle=True)
    iscell = np.load(path + r"\iscell.npy", allow_pickle=True)

    # Filter out neuropil
    stat = stat[iscell[:,0].astype(bool)]
    F = F[iscell[:,0].astype(bool),:]

    return tiff, F, stat


def compute_NFC(on_freq, off_freqs):
    """Normalized Fourier Coefficient: |on-freq coefficient| / noise-floor sigma,
    where sigma is derived from the off-frequency coefficients."""
    return np.abs(on_freq) / np.sqrt(0.5 * np.mean(np.abs(off_freqs)**2))


def fit_Fourier(F, T=1, f=0.4, Q_frac=0.1):
    """
    F: (2d arr) fluorescence traces
    T: (int) sample spacing (inverse of sampling rate)
    f: (float) stim frequency
    Q_frac: (float) off-frequency half-window width, as a fraction of `f`
        (half-width_Hz = Q_frac * f). Converted to an integer bin count `M`
        via `bins_for_fraction`, using this call's own real-FFT bin
        resolution `1/(N*T)` (N = frame count after the 120-frame-multiple
        truncation below) -- see that function's docstring for the exact
        formula and error conditions.

    Returns NFC_l, onfreq_coef_l, offfreq_coef_l, xf[freq_win], M, avg_signal_l
    -- avg_signal_l is each cell's mean RAW fluorescence, over the same
    N-frame window the FFT itself analyzes. This is the numerator of the
    `sens` statistic (avg_signal / (2*sigma)), analogous to the spiking-side
    `spk_count / (2*sigma)` where `spk_count` is spike count.
    """

    onfreq_coef_l = np.zeros(len(F),dtype="complex")
    offfreq_coef_l = [None] * len(F)

    # N/xf/f0/M don't depend on cell_ind (only on F.shape[1] and T) -- hoisted
    # out of the per-cell loop instead of being recomputed every iteration.
    # `120*(F.shape[1]//60)` is a NOMINAL "multiple of 120 frames" target,
    # but it routinely EXCEEDS the actual frame count (e.g. 2280 vs an
    # actual 1194) -- the per-cell slice below then silently clips to
    # F.shape[1] elements. N must reflect that ACTUAL (possibly clipped)
    # length, exactly like the original per-cell `N = len(y)` (computed
    # AFTER slicing) did, or fftfreq/f0/freq_win end up built on a
    # frequency grid that doesn't match the real FFT output length at all --
    # confirmed on real data: silently wrong with no crash, pulling the
    # coefficient from the wrong bin entirely (e.g. requesting 0.4 Hz on a
    # grid sized for 2280 samples, but actually getting a 1194-sample FFT,
    # ends up reading the bin for ~0.76 Hz instead).
    N = min(int(120 * (F.shape[1] // 60)), F.shape[1])
    avg_signal_l = np.mean(F[:, :N], axis=1)

    xf = fftfreq(N, T)[:N // 2]
    f0 = np.argmin(np.abs(f - xf))
    # max_bins: the tighter of the two sides actually available in this
    # real-FFT grid (only N//2 bins wide) -- without this, a high base
    # frequency close to Nyquist combined with a wide Q_frac can request a
    # window that runs off the edge of the spectrum, silently corrupting
    # freq_win rather than erroring at the point of misconfiguration.
    max_bins = min(f0, len(xf) - 1 - f0)
    M = bins_for_fraction(f, Q_frac, resolution=1.0 / (N * T), max_bins=max_bins,
                           context=f"fit_Fourier @ {f}Hz ")
    freq_win = np.concatenate([np.arange(f0 - M, f0), np.arange(f0 + 1, f0 + M + 1)])

    for cell_ind in range(len(F)):
        # 120 is the lowest common multiple of the periods here.
        y = F[cell_ind, :N].copy()
        y -= np.mean(y)
        yf = fft(y)[:N//2]

        offfreq_coef_l[cell_ind] = yf[freq_win]
        onfreq_coef_l[cell_ind] = yf[f0]

    NFC_l = [compute_NFC(c_on, c_off) for c_on, c_off in zip(onfreq_coef_l, offfreq_coef_l)]
    return NFC_l, onfreq_coef_l, offfreq_coef_l, xf[freq_win], M, avg_signal_l



# After rethinking how I fit things, I want to sort everything together
def get_len_df(path):
    """"
    Get all tifs in path. Contains:

    2022_09_15_fish1_magneto_0/
    2022_09_15_fish1_magneto_1/
    2022_09_15_fish1_magneto_2/
    2022_09_15_fish1_no_magneto_0/
    2022_09_15_fish1_no_magneto_1/
    2022_09_15_fish1_no_magneto_2/
    magneto/
    no-magneto/
    suite2p/
    2022_09_15_fish1_magneto_0.tif
    2022_09_15_fish1_magneto_1.tif
    2022_09_15_fish1_magneto_2.tif
    2022_09_15_fish1_no_magneto_0.tif
    2022_09_15_fish1_no_magneto_1.tif
    2022_09_15_fish1_no_magneto_2.tif

    Now, you can run suite2p. It'll automatically concatenate tiffs for you.
    Then you have a length df ready to go.
    """

    tif_paths = glob.glob(path+r"/*.tif")
    len_df_l = []
    for tif_path in tqdm.tqdm(tif_paths):
        try:
            tiff = tifffile.memmap(tif_path)
        except:
            print(f"Error reading {tif_path}")
            tiff = tifffile.imread(tif_path)

        # Add to list
        len_df_l.append(pd.DataFrame({"path":tif_path, "length":tiff.shape[0]}, index=[0]))


    len_df = pd.concat(len_df_l)

    # Get cumulative sums
    len_df["start"] = np.cumsum(len_df.length.values) - len_df.length.values[0]
    len_df["end"] = np.cumsum(len_df.length.values)
    return len_df


def dataIO(path, tiffname, len_df=None, sr=1, iscell_thres=0.7, npix_thres=20):
    """

    Parameters
    ----------
    path : str
        path to suite2p directory
    tiffname : str

    """
    if len_df is not None:
        tiffname = os.path.join(path, tiffname)
        start = len_df.loc[len_df["path"] == tiffname, "start"].values[0]
        end = len_df.loc[len_df["path"] == tiffname, "end"].values[0]
        # tiff = tifffile.memmap(tiffname);
    else:
        # None so that it indexes the entire array
        start, end = None, None
        # tiff = tifffile.memmap(tiffname);

    try:
        tiff = tifffile.memmap(tiffname)
    except:
        tiff = tifffile.imread(tiffname)

    F = np.load(path + r"\suite2p\plane0\F.npy", allow_pickle=True)
    spks = np.load(path + r"\suite2p\plane0\spks.npy", allow_pickle=True)
    stat = np.load(path + r'\suite2p\plane0\stat.npy', allow_pickle=True)
    iscell = np.load(path + r"\suite2p\plane0\iscell.npy", allow_pickle=True)

    # filter out neuropil using: classifier values, mask size
    thresholded_cell_inds = np.logical_and(
        (iscell[:,1] > iscell_thres),
        np.array([s["npix"] > npix_thres for s in stat])
    )

    spks = spks[thresholded_cell_inds, start:end]
    stat = stat[thresholded_cell_inds]
    F = F[thresholded_cell_inds, start:end]

    duration = F.shape[0] * sr

    # Return spoils
    return tiff, F, spks, stat, duration


def remove_flatlines(F, spks=None, stat=None, rtol=0.01, f=0.2):
    """In the noisier recordings, some traces are just
    delta functions, which result in sqrt(2) values which
    must be cleaned from the raw data. This strategy identies
    these traces by running a fourier transform at a separate frequency,
    and removes traces with an NFC value of sqrt(2).

    `spks`/`stat` are optional (NWB-backed callers don't load `spks.npy` at
    all -- it's never used analytically downstream -- and may not have
    `stat` in the original suite2p object-array form). Also returns
    `inclusion_inds`, the original row positions that survived, so callers
    can map back to their own full-ROI-set index (e.g. a PlaneSegmentation
    row) without the fragile centroid-matching this used to require."""

    """Note: if there are nans, that means that the trace is all zeros"""
    # Q_frac here is an artifact-detection window, not stimulus-meaningful --
    # only needs to clear MIN_FOURIER_BINS at this f=0.2 default.
    NFC_l, onfreq_coef_l, offfreq_coef_l, xf, _M, _avg_signal_l = fit_Fourier(F, T=1, f=f, Q_frac=0.15)
    inclusion_inds = np.where(
        np.logical_not(np.isclose(NFC_l, np.sqrt(2), rtol=rtol))
        & np.logical_not(np.isnan(NFC_l)))[0]
    spks_out = spks[inclusion_inds] if spks is not None else None
    stat_out = stat[inclusion_inds] if stat is not None else None
    return F[inclusion_inds, :], spks_out, stat_out, inclusion_inds
