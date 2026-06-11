import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm.auto as tqdm
import scipy.io
import json

import glob
import os
from ephysio.kilosortIO import Reader
from peakx import schmitt


def get_condition_array(t, p):
    """
    Parameters
    ----------
    t: array of spikes
    p: array of period markers
    """
    # If there are ps that are greater than last t, get rid of it
    p = p[(p<t[-1]) & (p>t[0])]
    
    c = np.zeros_like(t)
    switchpoint_indices = np.searchsorted(t, p)
    c[switchpoint_indices] = 1

    return np.cumsum(c)


def get_concat_spks_consistent_period(spks, ons, offs, period):
    """
    Starts spiketrain at zero and concatenates consecutive trials
    """
    last_off = 0
    
    spk_l = []
    for on, off in zip(ons, offs):
        BL = (period-(off-on)) / 2 
        spk_l.append(spks[(spks > on - BL) & (spks < off + BL)] - (on-BL) + last_off)
        last_off = off + BL
    
    return np.concatenate(spk_l)


def get_concat_spks(spks, ons, offs, BL=0*30_000):
    """
    Starts spiketrain at zero and concatenates consecutive trials
    """
    last_off = 0
    
    spk_l = []
    for on, off in zip(ons, offs):
        
        spk_l.append(spks[(spks > on - BL) & (spks < off + BL)] - (on-BL) + last_off)
        last_off = off + BL
    
    return np.concatenate(spk_l)


def get_cluster_info(data_path):
    """Loads cluster information"""
    # Load the sorting data
    unit_df = pd.read_csv(data_path + "\\cluster_info.tsv", sep='\t')
    if "channel" in unit_df.columns:
        unit_df["ch"] = unit_df["channel"]
    unit_df = unit_df.sort_values("ch")

    # Load channelmap data
    ChanMap = scipy.io.loadmat(r"C:\Users\dan\Documents\MATLAB\Kilosort-2.5\configFiles\neuropixPhase3B2_kilosortChanMap.mat")
    del ChanMap["__header__"], ChanMap["__version__"]
    del ChanMap["__globals__"], ChanMap["name"], ChanMap["chanMap"], ChanMap["connected"]
    probe_df = pd.DataFrame({key: val.flatten() for key, val in ChanMap.items()})
    probe_df.columns=['ch', 'shankInd', 'xcoords', 'ycoords']
    
    # Annotate sorting data with channelmap data
    udf = pd.merge(unit_df, probe_df, how="left", on="ch")

    # Account for different versions of phy/Kilosort
    if "cluster_id" in udf.columns:
        udf["id"] = udf["cluster_id"]

    return udf


def get_biggest_wf(template):
    """
    Input: 
    `template = we.get_template(unit_id=unit_id, mode='median')`
    """
    
    min_vals = np.min(template, axis=0)
    min_ind = np.where(np.min(min_vals) == min_vals)[0][0]
    return template[:,min_ind]


def smooth(data, window_width=500):
    """Edited from 
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    in order for the output to be the same length as the input. The naive implementation cuts the first window_width-1
    from the output array in terms of length. I add it back in by copying the first value window_width-1 times
    """
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    ma_vec_evened = np.concatenate(
        (
            np.ones(
                (len(data)-len(ma_vec))//2
            ) * ma_vec[0],
            ma_vec,
            np.ones(
                (len(data)-len(ma_vec))//2 + 1
            ) * ma_vec[-1],
        )
    )
    return ma_vec_evened


def window_rms(a, window_size=500):
    """
    https://stackoverflow.com/questions/8245687/numpy-root-mean-squared-rms-smoothing-of-a-signal
    """
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'valid'))


def mean_center(data, threshold=None, NPIX_THRESHOLD=600):
    # Centers based on the mean of the last half of the data
    if threshold is not None:
        thres=threshold
    else:
        thres=NPIX_THRESHOLD
    return data-np.mean(data[data>thres])


def get_theta(analog_dat_path, threshold=None):
    """
    Unpacks your magnetic stimulus data and takes the phases of it
    
    Parameters
    ----------
    analog_dat_path: str/Path
        Path to binary file for analog inputs to Intan
    
    Returns
    -------
    front: np.array
        Trace of y component of magnetic field
    side: np.array
        Trace of x component of magnetic field
    θ: np.array
        Instantaneous phase of magnetic field
    """

    import spikeinterface
    import spikeinterface.extractors as se
    import spikeinterface.toolkit as st
    si_version = spikeinterface.__version__.split(".")
    
    assert si_version[0] == "0", "Spikeinterface version is more advanced than expected (<1.0)"

    if int(si_version[1]) >= 90:
        ADC = se.BinaryRecordingExtractor(
            analog_dat_path, sampling_frequency=30_000, num_chan=8, dtype="int16")

        ADC_trace = ADC.get_traces()

        front = mean_center(ADC_trace[:,0], threshold)
        side  = mean_center(ADC_trace[:,1], threshold)
    else: 
        ADC = se.BinDatRecordingExtractor(
            folder_location + "/analogin.dat",
            sampling_frequency=30_000, numchan=8, dtype="int16")

        ADC_trace = ADC.get_traces()

        front = mean_center(ADC_trace[0,:], threshold)
        side  = mean_center(ADC_trace[1,:], threshold)

    # Now it's in the right range
    return front, side, np.arctan2(front, side) + np.pi



def get_crossings(θ, offset, window_size=1000):
    """Get upward zero crossings from the recording
    Ensure you pass θ once the stimulus has started
    
    Parameters
    ----------
    θ: np.array
        Array of empirical instantaneous phases
    window_size: int
        Window size for smoothing operation. Empirical noodling around suggests that 1000 is reasonable.
    
    Returns
    -------
    crossings: np.array
        
    Usage
    -----
    DEPRICATED, THE CURRENT APPROACH IS SIMPLER:
    ```
    window_size = 1_000
    _, idown = schmitt(smooth(θ, window_size), thr_on=2, thr_off=0, starttype=0, endtype=0)
    ```
    
    Old useage:
    ```
    front, side, θ = get_theta(folder_location + "/analogin.dat")
    θ = θ[front > thres]
    ```
    """
    # Window size is ~length of aberration
    smoothed_theta = smooth(θ, window_size)

    # Make all values below zero equal to negative 100.
    smoothed_theta[smoothed_theta < 0] = -100

    # Find where the value changes a lot
    crossings = np.where(np.diff(smoothed_theta) > 50)[0]

    # Offset the crossings
    return crossings + offset


def between_periods(signal, idown):
    return signal[(signal > idown[0]) & (signal < idown[-1])]

    
def get_whole_integer_periods(θ, sts, skips, window_size=1_000):
    """
    Parameters
    θ: np.array
        Array of instantaneous phases, from 0 to 2pi, which meansyou need to take the output of np.arctan2 and add 2 pi to it if you haven't already.
    sts: array of spiketrains
    
    Returns
    -------
    Cropped phase traces, spiketrain data, and period delineations for a whole number of periods"""
    # Schmitt triggers on SMOOTHED theta to get crossings for periods
    
    _, idown = schmitt(smooth(θ, window_size), thr_on=np.pi+2, thr_off=np.pi, starttype=0, endtype=0)

    # We will only take whole periods, i.e., from idown[0] to idown[1]
    period_crossings = idown

    # Further cleaning step: I can't turn on both signals at the same time so I need to skip a certain number of them
    period_crossings = period_crossings[skips:]
    
    relevant_θ = θ[idown[0]:idown[-1]]
#     relevant_period_crossings = between_periods(period_crossings, idown)
    relevant_sts = [between_periods(st, idown) for st in sts]
    
    return relevant_θ, period_crossings, relevant_sts
    
    
def skip_bad_periods(period_crossings, skips):
    """Removes a user-identified number of bad periods
    `# plt.plot(θ[period_crossings[skips]: period_crossings[skips+1]])`
    """
    return period_crossings[skips:]


def beep(frequency=2500, duration=1000):
    """https://stackoverflow.com/questions/6537481/python-making-a-beep-noise"""
    import winsound
      # Set Frequency To 2500 Hertz
      # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    
    
def process_raw_data(folder_locations_freq_skips, THRES=-14_000, good_units_only=True):
    """
    This is essentially only for the UCLA probes.

    Parameters
    ----------
    folder_locations_freq_skips: list(tuple)
        List of tuples comprising (absolute path to data, frequency)
    THRES: int
        User-defined threshold for separating out baseline and stimulus, since before stimulus, the ADC level is quite low, below this threshold.
    good_units_only: bool
        Whether to filter out Kilosort-labeled noise and MUA units. Note: This function does not use user-curated input on sorting.
    
    Returns
    -------
    modulation_df: pd.DataFrame
        You know
    data: list(list(np.ndarray))
        frequency[recording[unit spike trains]]
    """
    # Initialize output structures
    data, pandas_l, crossing_d = {}, [], {}

    for folder_location, freq, skips in folder_locations_freq_skips:   
        # Initialize output structures
        front, side, θ = get_theta(folder_location + "/analogin.dat")

        # Get sorting object and labels
        # sorting = se.KiloSortSortingExtractor(folder_location)
        ksr = Reader(folder_location)
        sampling_rate = 30_000

        # Get unit labels
        label_df = pd.read_csv(folder_location + r"/cluster_group.tsv", sep="\t")
        if good_units_only:
            label_df = label_df.loc[label_df.KSLabel=="good",:]

        # For each unit...
        unfiltered_sts = [ksr.spikesforcluster(row.cluster_id) for _, row in label_df.iterrows()]

        # Get whole integer periods
        _, period_crossings, relevant_sts = get_whole_integer_periods(θ, unfiltered_sts, skips)
        
        # Further cleaning step: sometimes 
        period_crossings = period_crossings[skips:]
        
        crossing_d[(folder_location, freq)] = period_crossings
        
        # Sanity check raw data
        # I basically don't use relevant_θ
        fig, axes = sanity_check_raw_data(
            θ, period_crossings,
            relevant_sts,
            n_representative_units=12,
            sampling_rate=sampling_rate)
        
        axes[0].set_title(f"{freq} Hz, {folder_location}")
        # Save figure
        save_and_close(fig, folder_location, "Raw", freq, save_path=r"C:\Users\dan\Documents\MagnetSearch\figs")
        spks = []
        for st_ind, st in enumerate(relevant_sts):
            # If there are zero spikes, skip this unit.
            if len(st) == 0:
                continue
            
            row = label_df.iloc[st_ind, :]
            # 
            pandas_l.append(
                pd.DataFrame({
                    "period": [np.sum(spk > period_crossings) for spk in st],
                    "spk": st / sampling_rate, # In seconds
                    # Do not use relevant_θ here because st is not offset.
                    "phase": θ[st], # add pi to bring it into range 0, 2π
                    "freq": freq,
                    "id": row.cluster_id,
                    "rec": folder_location.split("\\")[-1].split("/")[-1]
                })
            )

            # Append modulated spikes to the list and convert to seconds
            spks.append(st / sampling_rate)

        # If `freq` is not a key, make it one and dd the spikes to the dict
        if freq not in data:
            data[freq] = {}

        data[freq][folder_location]=unfiltered_sts, period_crossings
    
    modulation_df = pd.concat(pandas_l)
    return modulation_df, data, period_crossings

"""
***Neuropixels-specific utilities***
"""

def get_theta_NPIX(ADC_trace):
    """
    Unpacks your magnetic stimulus data and takes the phases of it
    
    Parameters
    ----------
    analog_dat_path: str/Path
        Path to binary file for analog inputs to Intan
    
    Returns
    -------
    front: np.array
        Trace of y component of magnetic field
    side: np.array
        Trace of x component of magnetic field
    θ: np.array
        Instantaneous phase of magnetic field
    """
    
    # 
    front = mean_center(ADC_trace[:,0])
    side  = mean_center(ADC_trace[:,1])

    # Now it's in the right range
    return front, side, np.arctan2(front, side) + np.pi


def get_whole_integer_periods_NPIX(θ, sts, ldr, skips, window_size=1_000):
    """
    Gets whole integer periods, as well as takes into account differences in NPIX and NIDAQ sampling.
    
    It translate period crossings in NIDAQ into NPIX time, makes an idealized sawtooth, and uses
    spikes in NPIX time.
    
    Parameters
    θ: np.array
        Array of instantaneous phases, from 0 to 2pi, which means you need to take the output of np.arctan2 and add 2 pi to it if you haven't already.
    sts: array of spiketrains
    
    Returns
    -------
    Cropped phase traces, spiketrain data, and period delineations for a whole number of periods"""
    # Schmitt triggers on SMOOTHED theta to get crossings for periods
    _, period_crossings = schmitt(
        smooth(θ, window_size), thr_on=np.pi+2,
        thr_off=np.pi, starttype=0, endtype=0
    )
    
    # Further cleaning step: I can't turn on both signals at the same time so I need to skip a certain number of them
    period_crossings = period_crossings[skips:]
        
    # Translate period crossings to NPIX time
    translated_period_crossings = ldr.shifttime(
        period_crossings, sourcestream=ldr.nidaqstream(),
        deststream=ldr.spikestream(),
        sourcebarcode="A2"
    ).astype(int)


    # Get total length of NPIX recording, effectively
    length = ldr.shifttime(
        np.array([len(θ)]),
        sourcestream=ldr.nidaqstreams()[0],
        deststream=ldr.spikestreams()[0],
        sourcebarcode="A2"
    ).astype(int)

    # Correct theta using translated period crossings
    corrected_θ = correct_theta(length, translated_period_crossings)
    
    # Take between first and last
    relevant_θ = corrected_θ[translated_period_crossings[0]:translated_period_crossings[-1]]
    
    # 
    relevant_sts = [between_periods(st, translated_period_crossings) for st in sts]
    
    return corrected_θ, relevant_θ, translated_period_crossings, relevant_sts


def correct_theta(length, period_crossings):
    """θ is liable to not be straight, leading to all sorts of mishaps.
    This function can correct that by taking the period crossings (when 6 pi radians become zero),
    and turn it into an idealized straight sawtooth shape."""
    # Generate list of indices the same length as θ. Then 
    θ_corrected = np.zeros(length)
    for crossing_ind in range(len(period_crossings)):
        if crossing_ind > 0:
            last = period_crossings[crossing_ind-1]
            curr = period_crossings[crossing_ind]
            θ_corrected[last:curr] = np.linspace(0, 2*np.pi, num=(curr-last))
    return θ_corrected


def save_and_close(fig, rec, label, freq, save_path=r'C:\Users\dan\Documents\MagnetSearch\figs'):
    """
    """
    recname = rec.split("/")[-1].split("\\")[-1]
    # print(f"{save_path}\\{recname}_{label}_{freq}.png")
    fig.savefig(f"{save_path}\\{recname}_{label}_{freq}.png")
    plt.close(fig)


def process_raw_data_NPIX(
    folder_locations_freq_skips, THRES=300, good_units_only=True,
    sampling_rate=30_000):
    """
    Parameters
    ----------
    folder_locations_freq_skips: list(tuple)
        List of tuples comprising (absolute path to data, frequency)
    THRES: int
        User-defined threshold for separating out baseline and stimulus, since before stimulus, the ADC level is quite low, below this threshold.
    good_units_only: bool
        Whether to filter out Kilosort-labeled noise and MUA units. Note: This function does not use user-curated input on sorting.
    
    Returns
    -------
    modulation_df: pd.DataFrame
        You know
    data: list(list(np.ndarray))
        frequency[recording[unit spike trains]]
    """
    
    # Initialize output structures
    data = {}
    pandas_l = []

    for contents in folder_locations_freq_skips: # folder_location, freq, skips, st_d, recording, ldr
        if len(contents) == 5:
            folder_location, freq, skips, st_d, ldr = contents
        elif len(contents) == 6:
            # Sometimes these contents size six in the non-NWB version. I want to preserve functionality without a major refactor. 
            folder_location, freq, skips, st_d, recording, ldr = contents
        # Initialize output structures
        _, _, θ = get_theta_NPIX(ldr.data(ldr.nidaqstream()))

        # For each unit...
        unfiltered_sts = [st for st in st_d.values()]
        
        # Get whole integer periods
        # Correct the relevant sts and exclude incomplete periods
        corrected_θ, relevant_θ, period_crossings, relevant_sts = get_whole_integer_periods_NPIX(
            θ, unfiltered_sts, ldr, skips, window_size=10)
        
        # Sanity check raw data
        # I basically don't use relevant_θ
        fig, axes = sanity_check_raw_data(
            corrected_θ, period_crossings, relevant_sts,
            n_representative_units=12, sampling_rate=30e3
        )
        axes[0].set_title(f"{freq} Hz, {folder_location}")
        # Save figure
        save_and_close(fig, folder_location, "Raw", freq)
        spks = []
        for st_ind, st in enumerate(relevant_sts):
            # If there are fewer than 50 spikes, skip this unit.
            if len(st) < 50:
                continue
            # 
            pandas_l.append(
                pd.DataFrame({
                    "period": [np.sum(spk > period_crossings) for spk in st],
                    "spk": st / sampling_rate, # In seconds
                    # Do not use relevant_θ here because st is not offset.
                    "phase": corrected_θ[st.astype(int)], # add pi to bring it into range 0, 2π
                    "freq": freq,
                    "id": list(st_d.keys())[st_ind],
                    "rec": folder_location.split("\\")[-1].split("/")[-1]
                })
            )
            
            # Append modulated spikes to the list and convert to seconds
            spks.append(st / sampling_rate)
        
        # If `freq` is not a key, make it one and dd the spikes to the dict
        if freq not in data:
            data[freq] = {}

        data[freq][folder_location]=st_d, period_crossings
    
    modulation_df = pd.concat(pandas_l)
    return modulation_df, data, period_crossings

def get_MM_offset(cat_df, recname):
    """
    Gets offset for a given recording name from the catalog dataframe.
    Get offset, using the recname to find the right row in cat_df
    """
    row = cat_df.loc[[elem.split("/")[-1]==recname for elem in cat_df.recname]]
    offset = (row.cumulate - row.nframes).iloc[0] # get vals
    return offset

def update_MM_d_mag(MM_d, data, cat_df):
    """
    Docstring for update_MM_d
    
    :param MM_d: Description
    :param data: Description
    :param cat_df: Description
    """
    
    # Get relevant keys
    for freq in data.keys():
        for folder in data[freq].keys():
            recname = os.path.basename(folder)

            offset = get_MM_offset(cat_df, recname)
            _, period_crossings = data[freq][folder]
                
            mag_period = period_crossings[[0, -1]] + offset
            MM_d["aux"][recname] = (mag_period, freq)

    return MM_d


def save_MM_d_pickle(MM_d, filename, sr=30_000):
    """Saves data to a pickle file.
    
    Parameters
    ----------
    data: dict
        The data to save.
    filename: str
        The name of the file to save to.
    """
    # First, convert everything to seconds
    # Spikes
    spks_int32 = MM_d["spikes"] # has np.int32 for keys instead of int
    spks = {} # New spks dict
    for cluster, st in spks_int32.items():
        spks[int(cluster)] = st / sr 
    MM_d["spikes"] = spks

    # Aux
    for recname, (period, freq) in MM_d["aux"].items():
        MM_d["aux"][recname] = (period / sr, freq)

    # save    
    with open(filename, 'wb') as f:
        pickle.dump(MM_d, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_MM_d_pickle(filename):
    """Loads data from a pickle file.
    
    Parameters
    ----------
    filename: str
        The name of the file to load from.
    """
    with open(filename, 'rb') as f:
        MM_d = pickle.load(f)
    return MM_d

def sanity_check_raw_data(θ, period_crossings, sts, n_representative_units=7, sampling_rate=30_000):
    """
    Usage:
    ```
    relevant_θ, relevant_period_crossings, relevant_sts = get_whole_integer_periods(θ, sorting)
    fig, ax = sanity_check_raw_data(relevant_θ, relevant_period_crossings, relevant_sts, n_representative_units=7)
        
    # Save figure
    save_and_close(fig, folder_location.split('\\')[-1].split["/"][-1], "Raw", freq, save_path="figs/diagnostics/")
    
    # Further analysis...
    ```
    """
    
    # Figsize depends on length of signal.
    fig, axes = plt.subplots(2, 1, figsize=(25, 4))

    # Plot theta...
    axes[0].plot(θ, color="g", linewidth=0.4, label="ϕ") # r"$\vec{B}$"
    axes[0].set_ylabel("ϕ") # r"|$\vec{B}$| (a.u.)"
    axes[0].set_xlabel("time (s)")

    # Plot period crossings
    axtwin = axes[0].twinx()
    diffs = np.diff(period_crossings)
    diffs = np.concatenate([[np.mean(diffs)], diffs])
    axtwin.plot(
        period_crossings,
        diffs, # np.ones(len(period_crossings)) * np.pi,
        "r|"
    )
    axtwin.set_ylabel("inter-period-interval")
    
    # Plot representative firing
    [axes[1].plot(st, np.ones(len(st)) * (row_ind + np.pi + 1), r"k|") for row_ind, st in enumerate(sts[:n_representative_units])]
    
    # match xlim
    axes[1].set_xlim((axes[0].get_xlim()))
    
    # Set xticklabels
    [ax.set_xticks(ax.get_xticks()) for ax in axes]
    [ax.set_xticklabels(ax.get_xticks() // sampling_rate) for ax in axes]

    plt.tight_layout()
    return fig, axes

'''Sanity check end'''


def save_aggregated_data(quick_debug=False):
    """Annotates region, species, and date for zebra finches, pigeons, and owls.
    Zebrafish and medaka already have that data available.
    
    Parameters
    ----------
    quick_debug: bool
        If True, skips loading modulation and fourier data from pickles."""
    
    # Get spiking data    
    if not quick_debug:
        modulation_df_l = []
        modulation_df_files = glob.glob("./data/*True_modulation_df.pickle")

        for modulation_df_file in modulation_df_files:
            modulation_df_l.append(pd.read_pickle(modulation_df_file))
        all_modulation_df = pd.concat(modulation_df_l).reset_index(drop=True)
    

    # Fourier data
    full_fourier_df_files = glob.glob("./data/*full_fourier_df.pickle")
    # print("zebrafish pickles being excluded from all_fourier_df")
    f_df_l = []
    for ff_df_file in full_fourier_df_files:
        # if 'zebrafish' not in full_fourier_df_file:
        f_df_l.append(pd.read_pickle(ff_df_file))
    all_fourier_df = pd.concat(f_df_l)

    annot_d = {
    'mag10Hz_2022-04-21_18-17-43':  ['20220421', 'S37♀', 'cerebellum', 'zebra finch', "mag"],
    'mag1Hz_2022-04-21_18-06-30':   ['20220421', 'S37♀', 'cerebellum', 'zebra finch', "mag"], 
    'mag3Hz_2022-04-21_18-19-59':   ['20220421', 'S37♀', 'cerebellum', 'zebra finch', "mag"], 
    'mag5Hz_2022-04-21_18-12-27':   ['20220421', 'S37♀', 'cerebellum', 'zebra finch', "mag"], 
    'mag8Hz_2022-04-21_18-14-58':   ['20220421', 'S37♀', 'cerebellum', 'zebra finch', "mag"], 
    '3HZ_2022-06-21_15-01-35':      ['20220621', 'Or237♀', 'cerebellum', 'zebra finch', "mag"], 
    '4HZ_2022-06-21_15-03-26':      ['20220621', 'Or237♀', 'cerebellum', 'zebra finch', "mag"], 
    '5HZ_2022-06-21_15-06-07':      ['20220621', 'Or237♀', 'cerebellum', 'zebra finch', "mag"], 
    '7HZ_2022-06-21_15-07-27':      ['20220621', 'Or237♀', 'cerebellum', 'zebra finch', "mag"], 
    'mag10hz_2022-04-08_15-26-22':  ['20220408', 'V649♀', 'NCM', 'zebra finch', "mag"], 
    'mag3hz_2022-04-08_15-31-11':   ['20220408', 'V649♀', 'NCM', 'zebra finch', "mag"], 
    'mag5hz_2022-04-08_15-28-47':   ['20220408', 'V649♀', 'NCM', 'zebra finch', "mag"], 
    '2nd site 1 Hz_2022-02-28_17-54-46':    ['20220228_secondsite', 'V649♀', 'NCM', 'zebra finch', "mag"], 
    '2nd site 10 Hz_2022-02-28_17-53-18':   ['20220228_secondsite', 'V649♀', 'NCM', 'zebra finch', "mag"], 
    '2nd site 2 Hz_2022-02-28_17-48-17':    ['20220228_secondsite', 'V649♀', 'NCM', 'zebra finch', "mag"], 
    '2nd site 5 Hz_2022-02-28_17-50-52':    ['20220228_secondsite', 'V649♀', 'NCM', 'zebra finch', "mag"], 
    'mag5hz_2022-03-14_18-15-42':   ['20220314', 'G122♀', 'arcopallium', 'zebra finch', "mag"], 
    '10 Hz_2022-02-28_16-21-40':    ['20220228', 'V649♀', 'NCM', 'zebra finch', "mag"], 
    '2 Hz_2022-02-28_16-25-47':     ['20220228', 'V649♀', 'NCM', 'zebra finch', "mag"], 
    '5Hz_2022-02-28_16-06-03':      ['20220228', 'V649♀', 'NCM', 'zebra finch', "mag"], 
    'mag1_10hz_211203_180333':      ['20211203', 'S14♀', 'NCM', 'zebra finch', "mag"], 
    'mag1_1hz_211203_175013':       ['20211203', 'S14♀', 'NCM', 'zebra finch', "mag"], 
    'mag1_211207_001111':           ['2021123', 'S14♀', 'NCM', 'zebra finch', "mag"],
    'mag1_3hz_211203_175843':       ['20211203', 'S14♀', 'NCM', 'zebra finch', "mag"], 
    'mag1_half_taketwo_211203_173755': ['20211203', 'S14♀', 'NCM', 'zebra finch', "mag"], 
    'mag1_halfhz_211203_173141':    ['20211203', 'S14♀', 'NCM', 'zebra finch', "mag"], 
    'mag2_10hz_211203_185724':      ['20211203', 'S14♀', 'NCM', 'zebra finch', "mag"], 
    'mag2_2hz_211126_175222':       ['20211203', 'S14♀', 'NCM', 'zebra finch', "mag"],
    'mag2_3hz_211203_185443':       ['20211203', 'S14♀', 'NCM', 'zebra finch', "mag"], 
    'mag2_halfhz_211126_194937':    ['20211203', 'S14♀', 'NCM', 'zebra finch', "mag"],
    'mag2_halfhz_211203_185153':    ['20211203', 'S14♀', 'NCM', 'zebra finch', "mag"], 
    'mag3_2hz_211206_224245':       ['20211203', 'S14♀', 'NCM', 'zebra finch', "mag"],
    'mag3_2hz_211207_102113':       ['20211203', 'S14♀', 'NCM', 'zebra finch', "mag"],
    'mag3_5hz_211206_225809':       ['20211203', 'S14♀', 'NCM', 'zebra finch', "mag"],
    'mag3hz_2023-02-16_11-43-42':   ['20230216', 'Pk12L', 'above OT', 'Pigeon', "mag"], 
    'mag8hz2ndtry_2023-02-16_11-49-12': ['20230216', 'Pk12L', 'above OT', 'Pigeon', "mag"], 
    'pigeon_CB_Mag1_inclined_2023-02-21_12-12-00': ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"], 
    'pigeon_CB_Mag1_inclined_lightson_2023-02-21_12-19-00': ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"], 
    'pigeon_CB_Mag2_inclined_2023-02-21_12-21-00': ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"], 
    'pigeon_CB_Mag5_inclined_2023-02-21_12-22-00': ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"], 
    'pigeon_CB_Mag7_inclined_2023-02-21_12-24-00': ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"], 
    'pigeon_CB_Mag7_upright_2023-02-21_12-26-02': ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"], 
    'pigeon_CB_mag1_upright_2023-02-21_12-33-18': ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"], 
    'pigeon_CB_mag2_upright_2023-02-21_12-30-43': ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "mag"], 
    'pigeon_CB_oddball_5percent_2023-02-21_13-36-00': ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "positive control"],
    'pigeon_CB_WN3D_2023-02-21_12-35-40': ['20230221', 'Pk12L', 'cerebellum', 'Pigeon', "positive control"],
    'mag2_inclined_2023-02-28_17-12-50': ['20230228', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'mag2_inclined_lightson_2023-02-28_17-14-38': ['20230228', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'mag2_lightson_2023-02-28_17-18-29': ['20230228', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'mag2real_lightson_2023-02-28_17-20-26': ['20230228', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'mag5_inclined_lightson_2023-02-28_17-16-04': ['20230228', 'Pk12L', 'cerebellum', 'Pigeon', "mag"],
    'oddball_2023-02-28_17-25-32': ['20230228', 'Pk12L', 'cerebellum', 'Pigeon', "positive control"],
    'WN_2023-02-28_17-39-32': ['20230228', 'Pk12L', 'cerebellum', 'Pigeon', "positive control"],
    '2023-04-13_15-08-42_W25R_Mag2': ['20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-13_15-11-11_W25R_Mag3': ['20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-13_15-13-11_W25R_Mag8': ['20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-13_15-15-40_W25R_Mag5': ['20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-13_15-19-07_W25R_Mag2_inclined': [
        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-13_15-22-55_W25R_Mag3_inclined': [
        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-13_15-25-47_W25R_Mag3_inclined_repositioned': [
        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-13_15-27-50_W25R_Mag5_inclined_repositioned_redo': [
        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-13_15-29-41_W25R_Mag8_inclined_repositioned': [
        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-13_15-49-48_W25R_visual_3Hz_0': [        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_15-49-48_W25R_visual_3Hz_135': [        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_15-49-48_W25R_visual_3Hz_180': [        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_15-49-48_W25R_visual_3Hz_225': [        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_15-49-48_W25R_visual_3Hz_270': [        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_15-49-48_W25R_visual_3Hz_315': [        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_15-49-48_W25R_visual_3Hz_45': [        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_15-49-48_W25R_visual_3Hz_90': [        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_16-00-46_W25R_visual_2Hz_0': [        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_16-00-46_W25R_visual_2Hz_135': [        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_16-00-46_W25R_visual_2Hz_180': [        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_16-00-46_W25R_visual_2Hz_225': [        '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_16-00-46_W25R_visual_2Hz_270': [       '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_16-00-46_W25R_visual_2Hz_315': [    '20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_16-00-46_W25R_visual_2Hz_45': ['20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_16-00-46_W25R_visual_2Hz_90': ['20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_15-34-21_W25R_WN_IndepCh_redo': ['20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_15-39-50_W25R_WN_SameCh': ['20230413_firstsite', 'W25R (Rocco)', 'centrolateral part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_17-04-34_W25R_second_site_mag2_inclined': ['20230413', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "mag"],
    '2023-04-13_17-06-27_W25R_second_site_mag3_inclined': ['20230413', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "mag"],
    '2023-04-13_17-10-21_W25R_second_site_mag8_inclined': ['20230413', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "mag"],
    '2023-04-13_17-13-18_W25R_second_site_mag8': ['20230413', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "mag"],
    '2023-04-13_17-15-32_W25R_second_site_mag5': ['20230413', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "mag"],
    '2023-04-13_17-17-14_W25R_second_site_mag3': ['20230413', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "mag"],
    '2023-04-13_17-18-59_W25R_second_site_mag2': ['20230413', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "mag"],
    
    # 'second_site3-04-13_17-04-34_W25R_second_site_mag2_inclined': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon'], # ┬é
    # 'second_site3-04-13_17-10-21_W25R_second_site_mag8_inclined': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon'], 
    # 'second_site3-04-13_17-13-18_W25R_second_site_mag8': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon'], 
    # 'second_site3-04-13_17-15-32_W25R_second_site_mag5': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon'], 
    # 'second_site3-04-13_17-17-14_W25R_second_site_mag3': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon'], 
    # 'second_site3-04-13_17-18-59_W25R_second_site_mag2': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon'], 
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_0': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_135': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_180': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_225': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_270': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_315': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_45': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_17-31-40_W25R_second_site_visual_movinggratings3Hz_90': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_17-21-57_W25R_second_site_WN_SamCh': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-13_17-42-25_W25R_second_site_WN_IndepChan': ['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"], 
    
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz45':['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"],
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz0':['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"],
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz315':['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"],
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz225':['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"],
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz180':['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"],
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz90':['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"],
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz135':['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"],
    '2023-04-13_17-48-00_W25R_second_site_movinggratings_2Hz270':['20230413_secondsite', 'W25R (Rocco)', 'centromedial part of craniotomy', 'Pigeon', "positive control"],

    '2023-04-14_14-28-40_W25R_mag2': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-14_14-30-22_W25R_mag3': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-14_14-32-14_W25R_mag5': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-14_14-34-11_W25R_mag8': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-14_14-37-41_W25R_mag8_inclined_lightson': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-14_14-42-29_W25R_mag5_inclined_lightson': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-14_14-44-59_W25R_mag3_inclined_lightson': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-14_14-47-16_W25R_mag2_inclined_lightson': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-14_14-50-42_W25R_mag2_lightson': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-14_14-53-11_W25R_mag3_lightson': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-14_15-10-38_W25R_visual_3Hz_0': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-10-38_W25R_visual_3Hz_135': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-10-38_W25R_visual_3Hz_180': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-10-38_W25R_visual_3Hz_225': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-10-38_W25R_visual_3Hz_270': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-10-38_W25R_visual_3Hz_315': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-10-38_W25R_visual_3Hz_45': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-10-38_W25R_visual_3Hz_90': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-20-44_W25R_visual_2Hz_0': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-20-44_W25R_visual_2Hz_135': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-20-44_W25R_visual_2Hz_180': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-20-44_W25R_visual_2Hz_225': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-20-44_W25R_visual_2Hz_270': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-20-44_W25R_visual_2Hz_315': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-20-44_W25R_visual_2Hz_45': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-20-44_W25R_visual_2Hz_90': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_14-56-27_W25R_WN_IndepCh': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-14_15-01-52_W25R_WN_SameCh': ['20230414_firstsite', 'W25R (Rocco)', 'HP, posteromedial part of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_15-56-12_W1R_mag2': ['20230415', 'W1R (Roca)', 'Centromedial part of craniotomy', 'Pigeon', "mag"], 
    '2023-04-15_15-57-55_W1R_mag3': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "mag"], 
    '2023-04-15_15-59-40_W1R_mag8': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "mag"], 
    '2023-04-15_16-01-59_W1R_mag8_inclined': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "mag"], 
    '2023-04-15_16-03-43_W1R_mag3_inclined': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "mag"], 
    '2023-04-15_16-06-20_W1R_mag2_inclined': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "mag"], 
    '2023-04-15_16-08-19_W1R_visual_3Hz_0' : ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-08-19_W1R_visual_3Hz_135': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-08-19_W1R_visual_3Hz_180': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-08-19_W1R_visual_3Hz_225': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-08-19_W1R_visual_3Hz_270': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-08-19_W1R_visual_3Hz_315': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-08-19_W1R_visual_3Hz_45': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-08-19_W1R_visual_3Hz_90': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-19-13_W1R_visual_2Hz_0': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-19-13_W1R_visual_2Hz_135': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-19-13_W1R_visual_2Hz_180': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-19-13_W1R_visual_2Hz_225': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-19-13_W1R_visual_2Hz_270': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-19-13_W1R_visual_2Hz_315': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-19-13_W1R_visual_2Hz_45': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-19-13_W1R_visual_2Hz_90': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    '2023-04-15_16-37-23_W1R_3D_WN_Samechan': ['20230415', 'W1R (Roca)', 'Approximately Septal HP, posterocentralsection of craniotomy', 'Pigeon', "positive control"], 
    'mag10_2022-09-16_15-53-06': ['20220916', 'B32♀', 'wulst', 'zebra finch', "mag"], 
    'mag3_2022-09-16_15-42-31': ['20220916', 'B32♀', 'wulst', 'zebra finch', "mag"], 
    'mag5_2022-09-16_15-46-10': ['20220916', 'B32♀', 'wulst', 'zebra finch', "mag"], 
    'mag7_2022-09-16_15-48-29': ['20220916', 'B32♀', 'wulst', 'zebra finch', "mag"], 
    'mag9_2022-09-16_15-50-36': ['20220916', 'B32♀', 'wulst', 'zebra finch', "mag"], 
    'bars4_2022-09-16_15-31-32_0':  ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"], 
    'bars4_2022-09-16_15-31-32_135':  ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"], 
    'bars4_2022-09-16_15-31-32_180':  ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"], 
    'bars4_2022-09-16_15-31-32_225': ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"], 
    'bars4_2022-09-16_15-31-32_270': ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"], 
    'bars4_2022-09-16_15-31-32_315': ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"], 
    'bars4_2022-09-16_15-31-32_45': ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"], 
    'bars4_2022-09-16_15-31-32_90': ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"], 
    'WN_2022-09-16_14-58-30': ['20220916', 'B32♀', 'wulst', 'zebra finch', "positive control"],
    'Q117_npxl_2022-12-13_s02_magnetD7':['20221213', 'Q117', 'thalamus', 'Quail', "mag"],
    'Q117_npxl_2022-12-14_s02_magnetD7':["20221213", "Q117", "thalamus", 'Quail', "mag"],
    "Q134_11.01.24_s01":["20240111", "Q134", "Nidopallium", "Quail", "mag"],
    'Q146_npxl_2023-08-15_magnet1_g0':["20230815", "Q146", "Nidopallium", "Quail", "mag"],
    'q148magnet19-12s2_g0':     ["20241219", "Q148", "Nidopallium", "Quail", "mag"],
    'q148magnet19-12s2_g1':     ["20241219", "Q148", "Nidopallium", "Quail", "mag"],
    "magnerNPX2bank24shanks_g0":['20240625', "QNPX2", "Nidopallium", "Quail", "mag"],
    "magnerNPX2_g0":            ["20240625", "QNPX2", "Nidopallium", "Quail", "mag"],

    }

    # for rec, val_l in tqdm.tqdm(annot_d.items()):
    #     date, id_, region, species, contingency = val_l
    #     all_fourier_df.loc[all_fourier_df.rec==rec, "date"] = date
    #     all_fourier_df.loc[all_fourier_df.rec==rec, "ID"] = id_
    #     all_fourier_df.loc[all_fourier_df.rec==rec, "area"] = region
    #     all_fourier_df.loc[all_fourier_df.rec==rec, "species"] = species
    #     all_fourier_df.loc[all_fourier_df.rec==rec, "contingency"] = contingency
        
    #     # Finish up all_modulation_df
    #     if not quick_debug:
    #         all_modulation_df.loc[all_modulation_df.rec==rec, "date"] = date
    #         all_modulation_df.loc[all_modulation_df.rec==rec, "species"] = species
    #         all_modulation_df.loc[all_modulation_df.rec==rec, "ID"] = id_
    #         all_modulation_df.loc[all_modulation_df.rec==rec, "area"] = region
    #         all_modulation_df.loc[all_modulation_df.rec==rec, "contingency"] = contingency
    #         all_modulation_df.to_csv("./data/all_modulation_df.csv", index=False)

    # Build annotation DataFrame once (vectorized)
    annot_df = pd.DataFrame.from_dict(
        annot_d, orient='index',
        columns=['date', 'ID', 'area', 'species', 'contingency']
    ).rename_axis('rec').reset_index()

    # Create fast lookup series for each annotation
    annot_idx = annot_df.set_index('rec')

    # Fill (or add) columns by mapping; don't overwrite existing values
    for col in ['date', 'ID', 'area', 'species', 'contingency']:
        mapped = all_fourier_df['rec'].map(annot_idx[col])
        if col in all_fourier_df.columns:
            all_fourier_df[col] = all_fourier_df[col].fillna(mapped)
        else:
            all_fourier_df[col] = mapped

    if not quick_debug:
        for col in tqdm.tqdm(['date', 'ID', 'area', 'species', 'contingency']):
            mapped = all_modulation_df['rec'].map(annot_idx[col])
            if col in all_modulation_df.columns:
                all_modulation_df[col] = all_modulation_df[col].fillna(mapped)
            else:
                all_modulation_df[col] = mapped
        # write once
        all_modulation_df.to_csv("./data/all_modulation_df.csv", index=False)

    # Save memory / speed: convert string columns with few unique values to categorical (optional)
    for col in ['area', 'species', 'contingency', 'ID']:
        if col in all_fourier_df.columns:
            all_fourier_df[col] = all_fourier_df[col].astype('category')


    # Adding gutfreund area
    # all_fourier_df.loc[all_fourier_df.species=="Owl", "area"] = "pallium"
    
    # Remove lines corresponding to species=="Owl" and rec=="control"
    # This is an experiment without magnetic stimulation
    all_fourier_df = all_fourier_df.loc[~((all_fourier_df.species=="Owl") & (all_fourier_df.rec=="control")), :]

    # Clarify ambiguous labels
    # TODO: fix this so that you don't have to do this filtering.
    # all_fourier_df = all_fourier_df.loc[~pd.isna(all_fourier_df.area)]
    # all_fourier_df.loc[["centr" in elem for elem in  all_fourier_df.area.values], "area"] = "HP"
    # all_fourier_df.loc[["craniotomy" in elem for elem in all_fourier_df.area.values], "area"] = "HP"
    # all_fourier_df.loc[["cerebellum" in elem for elem in all_fourier_df.area.values], "area"] = "CB"
    # all_fourier_df.loc[["OT" in elem for elem in all_fourier_df.area.values], "area"] = "pallium"
    
    # Option A (recommended): operate on strings with regex then convert back to category
    area_str = all_fourier_df['area'].astype(str).str.lower()
    area_str = area_str.replace({
        r'.*centr.*': 'HP',
        r'.*craniotomy.*': 'HP',
        r'.*cerebellum.*': 'CB',
        r'.*ot.*': 'pallium'
    }, regex=True)
    # turn explicit 'nan' back into NA
    area_str = area_str.replace('nan', np.nan)
    all_fourier_df['area'] = area_str
    # optional: convert back to categorical for memory/perf
    all_fourier_df['area'] = all_fourier_df['area'].astype('category')


    # save dataframe as csv
    all_fourier_df.to_csv("./data/all_fourier_df.csv", index=False)
    
def load_aggregated_data():
    """Loads all_fourier_df, all_modulation_df"""
    all_fourier_df = pd.read_csv("./data/all_fourier_df.csv")
    all_modulation_df = pd.read_csv("./data/all_modulation_df.csv")
    return all_fourier_df, all_modulation_df


def get_poscontrols_negresults(all_fourier_df):
    """
    Sorts values from all_fourier_df into positive and negative results
    """
    # Remove something that says no stim
    all_fourier_df = all_fourier_df.loc[np.array(["nostim" not in elem for elem in all_fourier_df.rec.values]), :]

    # Boolean indexing for fish
    fish_inds = np.array([("fish" in elem) or ("medaka" in elem) for elem in all_fourier_df.species.values], dtype=bool)

    # Get rec values
    rec_values = all_fourier_df.rec.values
    
    """POSITIVE CONTROL"""
    not_fish_fourier_df_filtered_pos_control = all_fourier_df.loc[
        (all_fourier_df.nn > 50)
        & np.logical_not(fish_inds)
        & np.array([("visual" in elem) | ("oddball" in elem) 
                    | ("WN" in elem) | ("3D" in elem) for elem in rec_values]), :]

    # Magneto_0 means no visual. That means no positive control.
    fish_fourier_df_filtered_pos_control = all_fourier_df.loc[
        fish_inds & np.array([freq < 0.03 for freq in all_fourier_df.freq.values])
        & np.array(["magneto_0" not in elem for elem in rec_values]), :]
    
    """NEGATIVE RESULTS"""
    not_fish_fourier_df_filtered_neg_res = all_fourier_df.loc[
        ((all_fourier_df.nn > 50) | (all_fourier_df.species == "Owl"))
        & np.logical_not(fish_inds) & (all_fourier_df.freq > 1)
        & np.array(["visual" not in elem for elem in rec_values])
        & np.array(["WN" not in elem for elem in rec_values])
        & np.array(["3D" not in elem for elem in rec_values]), :]

    # Expt 5 had the most units
    owl_fourier_df_filtered_neg_res = all_fourier_df.loc[
        (all_fourier_df.species == "Owl")
        & (all_fourier_df.rec == "exp5"), :]
    
    # Fish negative
    fish_fourier_df_filtered_neg_res = all_fourier_df.loc[
        fish_inds & np.array([freq > 0.02 for freq in all_fourier_df.freq.values])
        & np.array(["no_magnet" not in elem for elem in rec_values])
        & np.array(["no-magnet" not in elem for elem in rec_values]), :]

    # Combine negative results
    all_fourier_df_filtered_neg_res = pd.concat([
        fish_fourier_df_filtered_neg_res,
        not_fish_fourier_df_filtered_neg_res,
        owl_fourier_df_filtered_neg_res])

    # Combine positive controls
    all_fourier_df_filtered_pos_control = pd.concat(
        [fish_fourier_df_filtered_pos_control, not_fish_fourier_df_filtered_pos_control])
    

    return all_fourier_df_filtered_neg_res, all_fourier_df_filtered_pos_control

