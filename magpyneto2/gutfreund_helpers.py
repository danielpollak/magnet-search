import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import get_cluster_info

from peakx import schmitt
from ephysio.kilosortIO import Reader
from scipy.stats import linregress
import scipy.signal as signal

import warnings

# pixels of the magnet per length in cm
conversion_rate = 31 / 2.353  # pixels per cm
import ephysio.vizio as vizio
import ephysio.kilosortIO as ksr

def run_viz(full_fourier_df, chat_thres=3, data_path = r"\\datanas\family\data_aggregated\Gutfreund\Q117_npxl_2022-12-13_s02_magnetD7\Q117_2022-12-13_s02_magnetD7_g0\Q117_2022-12-13_s02_magnetD7_g0_imec0\kilosort4"):
    """
    
    """
    import spikeinterface.extractors as se

    udf = get_cluster_info(data_path)
    ksr = Reader(data_path)
    st_d = ksr.spikesbycluster()
    
    # Assumes Spikeglx with KS4
    recording = se.SpikeGLXRecordingExtractor(os.path.dirname(data_path), stream_id="imec0.ap")
    fs_Hz = recording.get_sampling_frequency()
    
    ch_sp = []
    for _, row in full_fourier_df.loc[full_fourier_df.rr > chat_thres, "id"].iterrows():
        id = row.id
        ch_sp.append((udf.loc[udf.cluster_id==id,"ch"].values[0], st_d[id] / fs_Hz))
     
    dat = recording.get_traces()
    vizio.viz(dat, fs_Hz, spikes=ch_sp)


def smooth(angle_arr, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(angle_arr, box, mode='same')
    return y_smooth


def get_angular_difference(u, v):
    """Parameters
    ----------
    u, v: np.array, phasors or angles"""
    # First, check if it's a phasor
    # If phasor, convert to angles in degrees
    if np.iscomplex(u).any() or np.iscomplex(v).any():
        u = np.angle(u, deg=True)
        v = np.angle(v, deg=True)
    elif not np.iscomplex(u).any() and not np.iscomplex(v).any():
        # If it is an angle, modulus any angles outside [-360, 360]
        u = u % 360
        v = v % 360
    else:
        raise ValueError("One of the inputs is a phasor and the other is an angle. Please convert to the same type.")
    thing = u - v

    clip_inds = np.where(np.abs(thing) > 180)[0]
    if len(clip_inds) > 0:
        thing[clip_inds] = (thing[clip_inds] + 180) % 360 - 180
    return thing


# Generate histogram of speeds
def get_speed_timeseries(arr_x, arr_y, interval=5, fps=30):
    """gets a timeseries of average speed with intervals of `interval`
    Parameters
    ----------
    arr_(x/y): (array) x/y positions
    interval: (int) number of frames to average over
    """

    inter_frame_interval = 1 / fps * interval

    inter_frame_distances = np.array([np.sqrt((arr_x[i] - arr_x[i+interval])**2 + (arr_y[i] - arr_y[i+interval])**2) for i in range(len(arr_x)-interval)])

    speeds = inter_frame_distances / inter_frame_interval # pixels/s
    speeds = speeds / conversion_rate # cm / s
    return speeds


def read_DLC_csv(path):
    """Reads DLC csv and returns a pandas dataframe"""
    # Read csv to dataframe
    df = pd.read_csv(path, header=[0, 1, 2, 3])

    # Columns have only relevant info
    new_cols = [None] * len(df.columns)
    for col_i, column in enumerate(df.columns):
        new_cols[col_i] = tuple(
            [df.columns[col_i][tuple_i] for tuple_i in range(len(df.columns[col_i])) if tuple_i > 0])

    df.columns = new_cols
    return df



def match_avi_dlc(path):
    """
    Match avi files with boris, dlc files
    """
    # Holder var
    matched_file_l = []
    all_files = glob.glob(path)
    for avi_file in glob.glob(path+".avi"):
        # get pattern
        pattern = avi_file.split(".avi")[0]

        # Get DLC csvs
        DLC_csv_files = [file for file in all_files if pattern in file and "el.csv" in file]
        DLC_csv_file = DLC_csv_files[0] if len(DLC_csv_files) > 0 else None

        matched_file_l.append((avi_file, DLC_csv_file))

    return matched_file_l


def get_relevant_measures(dlc_csv, conversion_rate):
    """
    Parameters:
    -----------
    dlc_csv: str, path to dlc output
    conversion_rate: float, pixels per cm

    Returns
    -------
    mag_vector: np.array, LOS to magnet relative to beak
    head_vector: np.array, phasor of head position
    body_vector: np.array, phasor of body position
    ego_theta: np.array, egocentric head angle
    mag_theta: np.array, head-mag angle
    head_theta: np.array, allocentric head angle
    dist: np.array, distance between head and mag
    quail_speeds: np.array, speed of quail
    (neck_x, neck_y): tuple, x and y positions of neck
    (beak_x, beak_y): tuple, x and y positions of beak
    (base_x, base_y): tuple, x and y positions of base
    (mag_x, mag_y): tuple, x and y positions of mag
    df: pandas.DataFrame, whole DLC dataframe
    """

    df = read_DLC_csv(dlc_csv)
    # Get quail and mag positions

    headstage_base_x = df[('quail', 'headstagebase', 'x')]
    headstage_base_y = df[('quail', 'headstagebase', 'y')]

    neck_x = df[('quail', 'spine1', 'x')]
    neck_y = df[('quail', 'spine1', 'y')]

    beak_x = df[('quail', 'beak', 'x')]
    beak_y = df[('quail', 'beak', 'y')]

    base_x = df[('quail', 'tailbase', 'x')]
    base_y = df[('quail', 'tailbase', 'y')]

    mag_x = df[('single', 'mag', 'x')]
    mag_y = df[('single', 'mag', 'y')]

    # suppress warnings using with statement
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Remove mag outliers
        aberrant_inds = np.where(np.sqrt(np.diff(mag_x) ** 2 + np.diff(mag_y)) > 100)[0]
        mag_x[aberrant_inds] = np.nan
        mag_y[aberrant_inds] = np.nan

        # Get distances
        dist = smooth(np.sqrt((neck_x - mag_x) ** 2 + (neck_y - mag_y) ** 2) / conversion_rate, 4)

        # Get speeds
        quail_speeds = get_speed_timeseries(neck_x, neck_y)

        # Get angles!
        mag_vector = (mag_x - neck_x) + (mag_y - neck_y) * 1j
        head_vector = (beak_x - headstage_base_x) + (beak_y - headstage_base_y) * 1j
        body_vector = (neck_x - base_x) + (neck_y - base_y) * 1j

        # Egocentric head angle
        ego_theta = get_angular_difference(head_vector, body_vector)

        # Allocentric head angle
        head_theta = np.angle(head_vector, deg=True)

        # head-mag head angle
        mag_theta = get_angular_difference(mag_vector, head_vector)

        ego_theta = smooth(ego_theta, 4)
        head_theta = smooth(head_theta, 4)
        mag_theta = smooth(mag_theta, 4)

    return (mag_vector, head_vector, body_vector, ego_theta, mag_theta, head_theta, dist,
            quail_speeds, (neck_x, neck_y), (beak_x, beak_y), (base_x, base_y),
            (mag_x, mag_y), df)


def vectorized_schmidt_trigger(input_signal, lower_threshold, upper_threshold, falling=False):
    """
    Vectorized Schmidt Trigger implementation using NumPy.
    Currently only finds falling edges

    :param input_signal: NumPy array of input values.
    :param lower_threshold: Lower threshold for switching.
    :param upper_threshold: Upper threshold for switching.
    :return: NumPy array of output states.
    """
    # Initialize the output array with False
    output = np.full_like(input_signal, False, dtype=bool)

    # Identify indices where input crosses the upper and lower thresholds
    upper_crossings = np.where(input_signal > upper_threshold)[0]
    lower_crossings = np.where(input_signal < lower_threshold)[0]

    # If rising edge, just swap the crossings
    if not falling:
        temp = upper_crossings
        upper_crossings = lower_crossings
        lower_crossings = temp

    # Toggle the state at each crossing
    for uc in upper_crossings:
        output[uc:] = True  # Set to True from this index onwards
        # Find the next lower crossing after this upper crossing
        next_lower_crossing = lower_crossings[lower_crossings > uc]
        if next_lower_crossing.size > 0:
            output[next_lower_crossing[0]:] = False  # Reset to False from the next lower crossing

    return output.astype(int)


def get_date_time(datetime_str):
    """Takes OEGUI datetime string and returns date and time"""
    date = datetime_str[:8]
    time = datetime_str[8:]
    # Get month, day, year and HMS
    month, day, year = date[:2], date[2:4], date[4:]
    hour, minute, second = time[:2], time[2:4], time[4:]
    return np.datetime64(f"{year}-{month}-{day} {hour}:{minute}:{second}")


def pretty_print_datetime(datetime_str):
    """Takes OEGUI datetime string and returns date and time"""
    month, day, year, hour, minute, second = get_date_time(datetime_str)


def get_Gutfreund_files(data_path: str):
    """Get the files for the Gutfreund data
    Parameters
    ----------
    data_path: str

    Returns
    -------
    AP_recording_path: str
        The path to the AP recording
    spikeglx_path: str
        The path to the NIDQ recording
    avi_path: str
        The path to the video
    timestamp_path: str
        The path to the timestamp
    ttl_path: str
        The path to the TTL
    """
    AP_recording_path = glob.glob(data_path + "/**/*.ap.bin", recursive=True)
    assert os.path.isfile(AP_recording_path[0]), "AP recording not found"

    spikeglx_path = glob.glob(data_path + "/**/*.nidq.bin", recursive=True)
    assert os.path.isfile(spikeglx_path[0]), "NIDQ recording not found"
    
    avi_path = glob.glob(os.path.dirname(data_path) + "/**/*.avi", recursive=True)
    assert os.path.isfile(avi_path[0]), "Video not found"

    timestamp_path = glob.glob(data_path + "/../**/cam2_timestamp*.csv", recursive=True)
    assert os.path.isfile(timestamp_path[0]), "Timestamp not found"

    ttl_path = glob.glob(data_path + "/../**/cam2_ttl*.csv", recursive=True)
    assert os.path.isfile(ttl_path[0]), "TTL not found"

    return AP_recording_path[0], spikeglx_path[0], avi_path[0], timestamp_path[0], ttl_path[0]


def get_all_spiketrains(reader, sorting_path, label="good"):
    """"""
    unit_df = get_cluster_info(sorting_path)

    if "cluster_id" in unit_df.columns:
        unit_df["id"] = unit_df["cluster_id"]
        del unit_df["cluster_id"]
        
    unit_df = unit_df.loc[unit_df.group==label, :]
    label_units = unit_df.loc[unit_df.group==label, "id"].values.tolist()
    all_sts_d = {
        cluster_id:sts for cluster_id, sts in reader.spikesbycluster().items() if cluster_id in label_units
    }
    all_sts = all_sts_d.values()

    return all_sts_d, all_sts, unit_df


def unpack_Gutfreund_data(data_path: str, label="good",):
    """
    Parameters
    ----------
    data_path: str
        The path to the data
    
    Returns
    -------
    AP_recording: spikeinterface.extractors.BinaryRecordingExtractor
        The AP recording
    sorting: spikeinterface.extractors.KiloSortSortingExtractor
        The sorting data
    NIDAQ_recording: spikeinterface.extractors.BinaryRecordingExtractor
        The NIDQ recording
    cap: cv2.VideoCapture
        The video
    ttl_df_unfilt: pandas.DataFrame
        The TTL dataframe
    timestamp_df: pandas.DataFrame
        The timestamp dataframe
    label: str
        The label to use for the sorting data (good vs mua)"""
    import spikeinterface.extractors as se

    AP_recording_path, spikeglx_path, avi_path, timestamp_path, ttl_path = get_Gutfreund_files(data_path)
    
    # AP recording
    AP_last_trace_path = os.path.join(data_path, "AP_last_trace.npy")
    if not os.path.isfile(AP_last_trace_path):
        print(f"caching at {AP_last_trace_path}")
        AP_recording = se.BinaryRecordingExtractor(AP_recording_path, 30_000, 385, dtype="int16")
        np.save(AP_last_trace_path, AP_recording.get_traces()[:,-1])
    
    assert os.path.isfile(AP_last_trace_path), "AP_last_trace not found"
    AP_last_trace = np.load(AP_last_trace_path)
    AP_sr = float(get_metadata_d(data_path  + r'/**/*.imec0.ap.meta')['imSampRate'])

    # Read sorting
    if os.path.isdir(os.path.join(os.path.dirname(AP_recording_path), r'kilosort4')):
        sorting_path = os.path.dirname(AP_recording_path) + r"\kilosort4"
    elif os.path.isfile(os.path.join(os.path.dirname(AP_recording_path), 'spike_times.npy')):
        sorting_path = os.path.dirname(AP_recording_path)
    else:
        raise FileNotFoundError("Sorting not found")
    
    # sorting = se.KiloSortSortingExtractor(sorting_path)
    reader = ksr.Reader(sorting_path)
    all_sts_d, all_sts, unit_df = get_all_spiketrains(reader, sorting_path, label=label)
    

    
    # Spikeglx binary
    _, NIDAQ_sr = get_sampling_rates(data_path)
    NIDAQ_recording = se.BinaryRecordingExtractor(spikeglx_path, NIDAQ_sr, 9, dtype="int16") 
    TTL_trace = np.array(NIDAQ_recording.get_traces()[:,-1])

    cap = cv2.VideoCapture(avi_path)
    ttl_df_unfilt = pd.read_csv(ttl_path)
    timestamp_df = pd.read_csv(timestamp_path)
    return TTL_trace, AP_last_trace, AP_sr, all_sts_d, all_sts, unit_df, NIDAQ_recording, cap, ttl_df_unfilt, timestamp_df



def plot_NIDAQ(NIDQ_bin):#: se.BinaryRecordingExtractor):
    """Plots the NIDQ recording
    
    Parameters
    ----------
    NIDQ_bin: spikeinterface.extractors.BinaryRecordingExtractor
        The NIDQ recording
    
    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure"""
    fig, ax = plt.subplots(1, 1)
    plt.plot(np.array(NIDQ_bin.get_traces()[:,-1]))
    ax.set_xticklabels(ax.get_xticks()//NIDQ_bin.get_sampling_frequency())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("TTL")
    return fig


def threshold(frame_trace: np.ndarray, thres: int = 30,):
    """
    Parameters
    ----------
    frame_trace: np.ndarray
        The trace of the frames
    thres: int
        
        """
    high_inds = np.where(frame_trace > thres)[0]
    first_inds = np.where(np.diff(high_inds) > 1)[0]
    inds = high_inds[first_inds]

    return inds


def fit_linear_TLL_timestamp_conversion(pulse_samples, ttl_df_unfilt):
    """
    Parameters
    ----------
    pulse_samples: np.ndarray
        The samples of the TTL pulses
    ttl_df_unfilt: pandas.DataFrame
        The TTL dataframe
    timestamp_df: pandas.DataFrame
        The timestamp dataframe
    
    Returns
    -------
    timestamp_samples: np.ndarray
        The samples of the timestamps
    """
    ttl_df = ttl_df_unfilt.loc[ttl_df_unfilt["Input State"]==1]
    # # Drop first one
    # ttl_df = ttl_df.iloc[1:]

    # Include only up to the number of pulse samples in ttl_df
    ttl_df = ttl_df.iloc[:len(pulse_samples)]

    # Time to sample
    result = linregress(ttl_df[" time.time()"], pulse_samples)
    return result


def convert_timestamp_to_samples(result, timestamps):
    """
    Converts nidaq timestamps to NPIX samples

    Parameters
    ----------
    result: scipy.stats.linregress
        The result of the linear regression
    timestamp_df: _
        _
    Returns
    -------
    timestamp_samples: np.ndarray
        The converted samples of the timestamps
    """
    m = result.slope
    b = result.intercept

    # Convert each element of timestamp_df to samples
    samples = m*(timestamps) + b

    return samples


def get_pulses_and_TTLs(AP_last_trace, ttl_df_unfilt:pd.DataFrame):
    ttl_df = ttl_df_unfilt.loc[ttl_df_unfilt["Input State"]==1]
    # Drop first one
    ttl_df = ttl_df.iloc[1:]

    # Connect ttl_df to the pulses
    pulse_samples = threshold(AP_last_trace, 30) # samples

    # There should be one less pulse than there are TTL pulses
    pulse_samples = pulse_samples[1:]

    # Include only up to the number of pulse samples in ttl_df
    ttl_df = ttl_df.iloc[:len(pulse_samples)]
    if len(pulse_samples > len(ttl_df)):
        print("Warning: More pulses than TTLs")
        pulse_samples = pulse_samples[:len(ttl_df)]
        
    # assert len(ttl_df) == len(pulse_samples)
    return pulse_samples, ttl_df


def get_phase_Quail(TTL_trace, NIDAQ_sr:float, plot=True, last_on=True, freq=5, magnet_off=False):
    """Converts the TTL trace to a phase in TTL time
    Parameters
    ----------
    TTL_trace: np.ndarray
        The TTL trace
    NIDAQ_sr: float
        The sampling rate of the NIDAQ
    plot: bool
        Whether to plot the result
    last_on: bool
        This is tricky. Sometimes the magnet is on and off.
        This option just selects for the last of the on-periods
    magnet_off: bool
        Enables control anlaysis where you look at times where the magnet was off and ask whther the effect is dependent on the presence of a magnet.
    Returns
    -------
    theta: np.ndarray
        The phase in TTL time
    """
    # These conflict. If they are both on, then magnet_off doesn't work.
    assert not (magnet_off and last_on)

    omega = 2 * np.pi * freq
    sine_wave = np.sin(omega * np.arange(len(TTL_trace)) / NIDAQ_sr)
    cosine_wave = np.cos(omega * np.arange(len(TTL_trace)) / NIDAQ_sr)
    
    theta = np.arctan2(sine_wave, cosine_wave)

    
    if magnet_off:
        # Selecting times when magnet is ON to ignore; set to nan
        theta[TTL_trace > 60] = np.nan
    else:
        # Selecting times when magnet is OFF to ignore; set to nan
        theta[TTL_trace < 60] = np.nan

    if last_on:
        # If "last_on", set the values to nan before the last "up time"
        ons, offs = schmitt(TTL_trace.astype("float64"), thr_on=90, thr_off=20)
        theta[:ons[-1]] = np.nan
    
    if plot:
        plt.figure()
        plt.plot(TTL_trace, label="TTL")
        plt.plot(sine_wave, label="sine")
        plt.plot(theta, label="theta")
        plt.legend()
    
    return theta


def get_metadata_d(pattern):
    """Get nidq.meta or imec.meta and turn it into a dict"""
    meta_file = glob.glob(pattern, recursive=True)[0]
    d = {}
    with open(meta_file) as f:
        for line in f:
            (key, val) = line.split("=")
            val = val[:-1]
            if val.isnumeric():
                d[key] = float(val)
            else:
                d[key] = val
    
    return d


def convert_AP_NIDAQ(imec_sr, nidq_sr):
    """Gets real sampling rates for AP and NIDAQ and returns the conversion factors"""
    
    # Spikes to NIDAQ
    AP_to_NIDAQ = float(nidq_sr) / float(imec_sr)
    NIDAQ_to_AP = 1 / AP_to_NIDAQ
    return NIDAQ_to_AP, AP_to_NIDAQ


def get_sampling_rates(data_path):
    """Gets real sampling rates for AP and NIDAQ and returns the conversion factors"""
    
    nidq_d = get_metadata_d(data_path  + r'/**/*.nidq.meta')
    nidq_sr = float(nidq_d['niSampRate'])

    imec_d = get_metadata_d(data_path  + r'/**/*.imec0.ap.meta')
    imec_sr = float(imec_d['imSampRate'])

    return imec_sr, nidq_sr


def Gutfreund_generator(locations_freqs, label):
    """
    Notes
    -----
    Useage example:
    for data_path, freq, gutfreund_files, gutfreund_data, relevant_measures, conversion_rates in Gutfreund_generator(locations_freqs):
        (AP_recording_path, NIDAQ_path, avi_path, timestamp_path, ttl_path) = gutfreund_files

        (TTL_trace, AP_last_trace, AP_sr, all_sts_d, all_sts, unit_df, 
        NIDAQ_recording, cap, ttl_df_unfilt, timestamp_df) = gutfreund_data

        (
            mag_vector, head_vector, body_vector, ego_theta, mag_theta, 
            head_theta, dist, quail_speeds, (neck_x, neck_y),
            (beak_x, beak_y), (base_x, base_y), (mag_x, mag_y), dlc_df
        ) = relevant_measures
        
        (result, bins, fps, NIDAQ_to_AP, AP_to_NIDAQ) = conversion_rates
    """
    for data_path, freq in locations_freqs:
        gutfreund_files = get_Gutfreund_files(data_path)
        gutfreund_data = unpack_Gutfreund_data(data_path, label=label)
        (avi_path, dlc_path) = match_avi_dlc(data_path + "/Videos/*")[0]
        relevant_measures = get_relevant_measures(dlc_path, conversion_rate)

        AP_sr, NIDAQ_sr = get_sampling_rates(data_path)
        NIDAQ_to_AP, AP_to_NIDAQ,  = convert_AP_NIDAQ(AP_sr, NIDAQ_sr) 
        (TTL_trace, AP_last_trace, AP_sr, all_sts_d, all_sts, unit_df, NIDAQ_recording, cap, ttl_df_unfilt, timestamp_df)= gutfreund_data

        pulse_samples, ttl_df = get_pulses_and_TTLs(AP_last_trace, ttl_df_unfilt)
        
        result = fit_linear_TLL_timestamp_conversion(pulse_samples, ttl_df)
        bins = convert_timestamp_to_samples(result, timestamp_df[" time.time()"]) 
        fps = 1 / (np.mean(np.diff(bins)) / AP_sr)
        conversion_rates = (result, bins, fps, NIDAQ_to_AP, AP_to_NIDAQ, AP_sr, NIDAQ_sr)
        
        yield data_path, freq, gutfreund_files, gutfreund_data, relevant_measures, conversion_rates


def plot_arr(arr, fps, frequency_win=None, axes=None, raw=False):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 3))
    
    nanless_arr = arr[np.logical_not(np.isnan(arr))]
    
    fxx, Pxx_den = signal.periodogram(nanless_arr, fps)

    if frequency_win is not None:
        fxx_inds = np.where((fxx > frequency_win[0]) & (fxx < frequency_win[1]))[0]

        # Filter fxx_inds
        fxx = fxx[fxx_inds]
        Pxx_den = Pxx_den[fxx_inds]


    axes[0].semilogy(fxx[1:], Pxx_den[1:], linewidth=0.5, color="k", alpha=0.5)

    axes[0].set_xlabel('frequency [Hz]')
    axes[0].set_ylabel('PSD [V**2/Hz]')

    if raw:
        axes[1].plot(arr)
        axes[1].set_xticklabels(axes[1].get_xticks()//fps)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Angle (rad)")
    return axes


def get_NP1():
    import probeinterface
    total=384 #, 960
    NP1 = probeinterface.generate_multi_columns_probe(
        4, num_contact_per_column=total//4, xpitch=16, ypitch=40,
        y_shift_per_column=[0,20,0,20], contact_shapes="square", contact_shape_params={"width":12},)
    return NP1


def plot_NP1(NP1, clean=True, ax=None):
    from probeinterface.plotting import plot_probe

    if ax is None:
        _, ax = plt.subplots()

    plot_probe(NP1, with_contact_id=True, title="NP1.0", ax=ax)
    
    if clean:
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_ylabel("")

        # removing bounding box on axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    return ax


def plot_units_on_probe(unit_df, NP1, waveform_d, normalize=True, ax=None, color=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,25))

    ax = plot_NP1(NP1, clean=False, ax=ax)

    id_col = "cluster_id" if "cluster_id" in unit_df.columns else "id"
    for id_i, (id, waveforms) in enumerate(waveform_d.items()):
        # Get channel coordinates (jitter them?)
        ch = unit_df.loc[unit_df[id_col]==id, "ch"].values[0]
        xcoords = unit_df.loc[unit_df[id_col]==id, "xcoords"].values[0]
        ycoords = unit_df.loc[unit_df[id_col]==id, "ycoords"].values[0]
        
        # Get the average waveform
        avg_waveform = np.mean(waveforms, axis=0)[10:80]

        # Normalize shape
        avg_waveform = avg_waveform 
        
        if normalize:
            avg_waveform = avg_waveform / np.max(avg_waveform) * 10


        x = np.linspace(0, 20, num=len(avg_waveform))        
        ax.plot(x + xcoords, avg_waveform.T + ycoords, linewidth=1, color=color)
    return ax

def extract_waveforms(recording, sorting, unit_df, pre=0.001, post=0.002, n_waveforms=100, label="good", sr=30_000, sorted=False):
    """
    Extract waveforms from a recording.

    Parameters:
    recording (spikeinterface.BaseRecording): The recording object.
    sorting (object): The sorting object.
    unit_id (int): The ID of the unit.
    pre (float, optional): The duration of the pre-spike window in seconds. Defaults to 0.001.
    post (float, optional): The duration of the post-spike window in seconds. Defaults to 0.002.
    n_waveforms (int, optional): The number of waveforms to extract. Defaults to 100.
    good (bool, optional): Flag to indicate whether to extract waveforms from good units only. Defaults to True.
    sr (int, optional): The sampling rate of the recording in Hz. Defaults to 30_000.

    Returns:
    dict: A dictionary containing the extracted waveforms, where the keys are unit IDs and the values are 2D arrays of waveforms.

    Raises:
    AssertionError: If the recording is not filtered.

    """
    assert recording.is_filtered

    pre, post = pre * sr, post * sr

    id_column = "id" if "id" in unit_df.columns else "cluster_id"

    waveform_d = {row[id_column]: np.zeros((n_waveforms, int(pre+post))) for _, row in unit_df.iterrows() if row.group==label}

    unit_df = unit_df.loc[unit_df.group==label]

    for row_i, row in unit_df.iterrows():
        spiketrain = sorting.get_unit_spike_train(unit_id=row[id_column]).flatten()
        
        # Randomly sample spikes
        if not sorted:
            sub_spiketrain = np.random.choice(spiketrain, n_waveforms, replace=True)
        else:
            sub_spiketrain = spiketrain[:n_waveforms]

        for spike_i, spike in enumerate(sub_spiketrain):  
            waveform_d[row[id_column]][spike_i,:] = recording.get_traces(
                start_frame=int(spike-pre), end_frame=int(spike+post),
                channel_ids=[row.ch]
            ).T
    return waveform_d

def ecdf_wfs(fourier_df, waveform_d):
    fig, ax = plt.subplots()
    argsorted = np.argsort(fourier_df.rr.values)
    n_units = len(fourier_df)

    for w_i, chat in enumerate(fourier_df.rr.values[argsorted]):
        waveforms = waveform_d[fourier_df.id[argsorted][w_i]]
        waveform = np.mean(waveforms, axis=0)
        x = np.linspace(-0.5, 0.5, len(waveform)) + chat
        
        quantile = w_i / n_units
        transformed_waveform = waveform/max(waveform)/n_units + quantile
        transformed_waveforms = waveforms/max(waveform)/n_units + quantile
        plt.plot(x , transformed_waveform, linewidth=0.5, color="k")
        plt.plot(x , transformed_waveforms.T[:,:50], linewidth=0.5, alpha=0.1, color=plt.cm.nipy_spectral(np.random.uniform()) )# viridis(w_i/n_units))

    plt.xlabel(r"$\hat{c}$")
    plt.ylabel(r"ECDF")

    return fig, ax