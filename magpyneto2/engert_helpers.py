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


def visualize_fourier(v_chat, b_chat):
    fig, ax = plt.subplots()

    generate_hist(ax, b_chat, "Magnetic", color="#C00000")
    generate_hist(ax, v_chat, "Visual")

    # Pretty up
    ax.legend()
    ax.set_xlabel(r"$|\hat{c}_f|$")
    ax.set_ylabel("count")
    
    return fig, ax


def generate_hist(ax, arr, label, color=None):
    vals, bins = np.histogram(arr, bins=np.arange(0, 12, 0.2), density=True)
    ax.bar(bins[:-1], vals, width=np.diff(bins)[0], align="edge", label=label, color=color, alpha=0.5)
    XX, YY = normalized_Fourier_PDF()
    ax.plot(XX, YY, "k", linewidth=1, label="theoretical")


def normalize_chat(on_freq, off_freqs):
    """"""
    return np.abs(on_freq) / np.sqrt(0.5 * np.mean(np.abs(off_freqs)**2))
    

def fit_Fourier(F, T=1, f=0.4, Q=100):
    """
    F: (2d arr) fluorescence traces
    T: (int) sample spacing (inverse of sampling rate) 
    f: (float) stim frequency
    Q: (int): frequency window size
    """

    onfreq_pow_l = np.zeros(len(F),dtype="complex")
    offfreq_pow_l = [None] * len(F)

    F = (F.T - np.min(F, axis=1)).T
    
    for cell_ind in range(len(F)):
        # 120 is the lowest common multiple of the periods here.
        y = F[cell_ind,:int(120*(F.shape[1]//60))].copy()
        y -= np.mean(y)
        N = len(y)
        yf = fft(y)[:N//2]
        xf = fftfreq(N, T)[:N//2]

        f0 = np.argmin(np.abs(f-xf))
        freq_win = np.concatenate([np.arange(f0-(Q-1), f0), np.arange(f0+1, f0+Q)])
        offfreq_pow_l[cell_ind] = yf[freq_win]
        onfreq_pow_l[cell_ind] = yf[f0]

    chat_l = [normalize_chat(c_on, c_off) for c_on, c_off in zip(onfreq_pow_l, offfreq_pow_l)]
    return chat_l, onfreq_pow_l, offfreq_pow_l, xf[freq_win]


def fit_Fourier_deprecated(F, T=1, f_v=1/60, f_b=0.4, Q_v=6, Q_b=100):
    """
    F: (2d arr) fluorescence traces
    T=1: (int) sample spacing (inverse of sampling rate) 
    f_v: (float) stim freq for visual stim
    f_b: (float) stim freq for magnetic stim
    Q_v=6: (int): window for visual stim
    Q_b=50: (int): window for magnetic stim
    """
    F = min_subtract(F)

    v_onfreq_pow_l, b_onfreq_pow_l = np.zeros(len(F), dtype="complex"), np.zeros(len(F),dtype="complex")
    v_offfreq_pow_l, b_offfreq_pow_l = [None] * len(F), [None] * len(F)

    for cell_ind in range(len(F)):
        # 120 is the lowest common multiple of the periods here.
        y = F[cell_ind,:int(120*(F.shape[1]//60))].copy()
        y -= np.mean(y)
        N = len(y)
        yf = fft(y)[:N//2]
        xf = fftfreq(N, T)[:N//2]


        f0_v = np.argmin(np.abs(f_v-xf))
        v_freq_win = np.concatenate([np.arange(f0_v-(Q_v-1), f0_v), np.arange(f0_v+1, f0_v+Q_v)])
        v_offfreq_pow_l[cell_ind] = yf[v_freq_win]
        v_fffreq_l = xf[v_freq_win]
        v_onfreq_pow_l[cell_ind] = yf[f0_v]

        f0_b = np.argmin(np.abs(f_b-xf))
        b_freq_win = np.concatenate([np.arange(f0_b-(Q_b-1), f0_b),np.arange(f0_b+1, f0_b+Q_b)])

        b_offfreq_pow_l[cell_ind] = yf[b_freq_win]
        b_fffreq_l = xf[b_freq_win]
        b_onfreq_pow_l[cell_ind] = yf[f0_b]

    v_chat_l = [normalize_chat(c_on, c_off) for c_on, c_off in zip(v_onfreq_pow_l, v_offfreq_pow_l)]
    b_chat_l = [normalize_chat(c_on, c_off) for c_on, c_off in zip(b_onfreq_pow_l, b_offfreq_pow_l)]
    
    return v_chat_l, b_chat_l, v_onfreq_pow_l, b_onfreq_pow_l, v_offfreq_pow_l, b_offfreq_pow_l, v_fffreq_l, b_fffreq_l



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


def save_GCaMP_diagnostics(tiff, F, stat, fourier_results, tiff_name, close=False, f_b=0.4, f_v=1/60):
    """
    Parameters
    ----------
    tiff : np.array
        tiff file contents
    F : np.array
        fluorescence traces
    stat : np.array
        suite2p statistics
    fourier_results : tuple
        tuple of fourier results from fit_Fourier_deprecated
    tiff_name : str
        name of tiff file
    close : bool
        close figures
    f_b : float
        magnetic frequency
    f_v : float
        visual frequency
    """
    
    (
        v_chat_l, b_chat_l, v_onfreq_pow_l,
        b_onfreq_pow_l, v_offfreq_pow_l, b_offfreq_pow_l,
        v_fffreq_l, b_fffreq_l
    ) = fourier_results

    anatomy_fig, anatomy_ax, trace_fig, trace_ax = functional_cluster(tiff, F, stat);
    
    anatomy_ax.set_title(tiff_name)
    trace_ax.set_title(tiff_name)
    
    anatomy_fig.savefig(r"C:\Users\dan\Documents\MagnetSearch\figs\diagnostics"+f"\\{tiff_name}_anatomy.png")
    trace_fig.savefig(r"C:\Users\dan\Documents\MagnetSearch\figs\diagnostics"+f"\\{tiff_name}_functional_cluster.png")
    
    fourier_fig, fourier_ax = visualize_fourier(v_chat_l, b_chat_l);
    fourier_ax.set_title(tiff_name)
    fourier_fig.savefig(r"C:\Users\dan\Documents\MagnetSearch\figs\diagnostics"+f"\\{tiff_name}_power_spectrum.png")    
    
    # Visual spectrum
    v_spectrum_fig, v_spectrum_ax = GCaMP_power_spectrum_diagnostic(
        v_fffreq_l, v_offfreq_pow_l, v_onfreq_pow_l, onfreq=f_v)
    
    v_spectrum_ax[0].set_title("visual_" + tiff_name)
    v_spectrum_fig.savefig(r"C:\Users\dan\Documents\MagnetSearch\figs\diagnostics"+f"\\{tiff_name}_visual_power_spectrum_diagnostic.png")
    
    # Magnetic spectrum
    b_spectrum_fig, b_spectrum_ax = GCaMP_power_spectrum_diagnostic(
        b_fffreq_l, b_offfreq_pow_l, b_onfreq_pow_l, onfreq=f_b) 
    
    b_spectrum_ax[0].set_title("magnetic_" + tiff_name)
    b_spectrum_fig.savefig(r"C:\Users\dan\Documents\MagnetSearch\figs\diagnostics"+f"\\{tiff_name}_magnetic_power_spectrum_diagnostic.png")
    

    if close:
        plt.close(anatomy_fig)
        plt.close(trace_fig)
        plt.close(fourier_fig)
        plt.close(v_spectrum_fig)
        plt.close(b_spectrum_fig)
    else:
        return anatomy_ax, trace_ax, fourier_ax, v_spectrum_ax, b_spectrum_ax


def GCaMP_power_spectrum_diagnostic(
        offfreq_l, offfreq_pow_l, onfreq_pow_l, onfreq=1/60):
    """
    Parameters
    ----------
    offfreq_l : list
        list of frequencies
    offfreq_pow_l : list
        list of power spectra
    onfreq_pow_l : list
        list of power spectra
    onfreq : float
        frequency of interest
    """
    fig, axes = plt.subplots(2,1)
    
    # real
    # Off freq
    [axes[0].plot(offfreq_l, offfreq_pow_l[i].real ,"k.") for i in range(len(offfreq_pow_l))]

    # On freq
    axes[0].plot([onfreq] * len(onfreq_pow_l), onfreq_pow_l.real, ".", alpha=.5)

    # imag
    # Off freq
    [axes[1].plot(offfreq_l, offfreq_pow_l[i].imag ,"k.") for i in range(len(offfreq_pow_l))]

    # On freq
    axes[1].plot([onfreq] * len(onfreq_pow_l), onfreq_pow_l.imag, ".")
    axes[1].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("real")
    axes[1].set_ylabel("imaginary")

    return fig, axes


def remove_flatlines(F,spks, stat, rtol=0.01, f=0.2):
    """In the noisier recordings, some traces are just
    delta functions, which result in sqrt(2) values which 
    must be cleaned from the raw data. This strategy identies
    these traces by running a fourier transform at a separate frequency,
    and removes traces with a c_hat value of sqrt(2)."""
    
    """Note: if there are nans, that means that the trace is all zeros"""
    chat_l, onfreq_pow_l, offfreq_pow_l, xf = fit_Fourier(F, T=1, f=f, Q=100)
    inclusion_inds = np.where(
        np.logical_not(np.isclose(chat_l, np.sqrt(2), rtol=rtol))
        & np.logical_not(np.isnan(chat_l)))[0]
    return F[inclusion_inds,:], spks[inclusion_inds], stat[inclusion_inds]
