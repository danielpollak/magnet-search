import numpy as np

import xml.etree.ElementTree as ET
from datetime import datetime
from dateutil.tz import tzlocal
import glob
from peakx import schmitt

from .utils import save_and_close
from .statistics import plot_magnitude_pdf, plot_power_by_freq, plot_coefficient_cdf, Moments_vs_FR, power_spectra
import matplotlib.pyplot as plt

#%% functions for extracting stimulation epochs
def get_stimulation_epoch(front, side, theta, skips, thres=-2500, thr_on=5, thr_off=1, endskips=-1):
    """
    Gets onset and offset of baseline and stimulation periods. Excludes skipped periods,
    which are neither baseline nor admissible stimulation epochs.
    
    Parameters
    front, side: (np.array) signals sent to front and side magnets, respectively
    theta: (np.array) phase of stimulus
    skips, endskips: (int) number of periods to skip - 1, either at the beginning or the end

    Returns
    -------
    (baseline_onset, baseline_offset), (stimulation onset, stimulation offset)"""
    
    assert len(front) == len(side), "Front and side signals must be of the same length"
    
    stimulation_mask = (front > thres) & (side > thres)
    stimulation_indicies = np.where(stimulation_mask)[0]

    _, period_crossings = schmitt(theta, thr_on=thr_on, thr_off=thr_off, starttype=0, endtype=0)
    period_crossings = period_crossings[
        (period_crossings > stimulation_indicies[0]) & 
        (period_crossings < stimulation_indicies[-1])]
    
    stim_onset = period_crossings[skips]
    stim_offset = period_crossings[endskips]
    return (stim_onset, stim_offset), period_crossings



def get_baseline_epoch(front, side, thres=-2500):
    """
    Gets onset and offset of baseline period.
    
    Parameters
    front, side: np.array signals sent to front and side magnets, respectively
    thres: int, threshold for determining baseline period
    Returns
    -------
    (baseline_onset, baseline_offset)"""
    
    assert len(front) == len(side), "Front and side signals must be of the same length"

    baseline_indicies = np.where((front < thres) & (side < thres))[0]
    baseline_onset = baseline_indicies[0] if len(baseline_indicies) > 0 else None
    baseline_offset = baseline_indicies[-1] if len(baseline_indicies) > 0 else None
    
    return (baseline_onset, baseline_offset)

def get_min_setting_time(folder_locations_freq_skips):
    """Estimates the minimum time that the file could have been started by looking at all settings xmls
    for given experiment"""
    setting_l = []
    for folder in folder_locations_freq_skips:
        setting_files = glob.glob(folder[0] + r"\**\settings.xml")
        if len(setting_files) > 0:
            for settingsxml in setting_files:
                setting_l.append(get_openephys_start_time(settingsxml))

    return min(setting_l)

def get_openephys_start_time(settings_path):
    """
    Extract the recording start time from OpenEphys settings.xml file.
    
    Parameters:
    data_path: settings.xml path
    
    Returns:
    datetime object with the recording start time
    """
    
    try:
        # Parse the XML file
        tree = ET.parse(settings_path)
        root = tree.getroot()
        
        # Find the DATE element
        date_element = root.find(".//INFO/DATE")
        
        if date_element is not None:
            date_string = date_element.text  # e.g., "28 Feb 2022 17:54:46"
            
            # Parse the date string
            # Format: "28 Feb 2022 17:54:46"
            recording_datetime = datetime.strptime(date_string, "%d %b %Y %H:%M:%S")
            
            # # Add timezone info (assuming local timezone)
            # recording_datetime = recording_datetime.replace(tzinfo=tzlocal())
            
            return recording_datetime
        else:
            print("DATE element not found in settings.xml")
            return datetime.now(tzlocal())
            
    except FileNotFoundError:
        print(f"settings.xml not found in {settings_path}")
        return None
    except ET.ParseError as e:
        print(f"Error parsing settings.xml: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing date string: {e}")
        return None
    

def process_spikes(st, T, fs):
    """
    Parameters
    ----------
    st: np.array spike times in s
    T: duration of stimulation epoch in s
    fs: stimulation frequency in Hz
    """
    # Reference range bounds
    flo, fhi = fs * 0.9, fs * 1.1  

    # Frequencies in the reference range, all multiples of 1/T
    freq_range_inclusive = np.arange(np.ceil(flo*T), np.ceil(fhi*T))/T 

    # Exclude fs itself from the range
    freq_range = [i for i in freq_range_inclusive if not np.isclose(i, fs, atol=1e-3, rtol=0)] 

    # Sum of phasors at all frequencies in the reference range
    ref_c = [np.sum(np.exp(-1j * 2 * np.pi * f * st)) for f in freq_range] 
    
    # Sum of phasors at stim frequency
    stim_c = np.sum(np.exp(-1j * 2 * np.pi * fs * st)) 
    
    # Sigma estimated from square of coefficients in the reference range
    sigma = np.sqrt(np.mean(np.abs(ref_c)**2)/2) 
    return flo, fhi, freq_range, ref_c, stim_c, sigma


def compute_c_hat(st, T, fs):  
    """
    Generate c_hat value
    Parameters
    ----------
    st: np.array spike times in s
    T: duration of stimulation epoch in s
    fs: stimulation frequency in Hz
    """
    _, _, _, _, stim_c, sigma = process_spikes(st, T, fs)

    c_hat = np.abs(stim_c) / sigma
    return c_hat


def run_diagnostics(
        ff_alt, fou_alt, fou0, frq, rec, Q, nn, 
        fou_alt_c, spks, save_path=r'C:\Users\dan\Documents\MagnetSearch\figs\nwb'):
    """
    Make diagnostic plots

    Parameters
    ----------
    ff_alt: alt freqs
    fou_alt: alt c values
    fou0: stimulus c value
    frq: stimulus freq
    rec: rec name
    Q: number of alt freqs
    nn: n neurons
    fou_alt_c: alt c values for neurons
    spks: spike times
    """
    # Save diagnostic plots
    ax = plot_power_by_freq(ff_alt, fou_alt, fou0, frq, save_path)
    plt.gca().set_title(f"{frq}, {rec}")
    save_and_close(plt.gcf(), rec, f"power_by_freq_Q{Q}", frq, save_path)
    
    ax = plot_coefficient_cdf(ff_alt, fou_alt)
    plt.gca().set_title(f"{frq}, {rec}")
    save_and_close(plt.gcf(), rec, f"coefficient_cdf_Q{Q}", frq, save_path)
    
    ax = plot_magnitude_pdf(ff_alt, fou_alt)
    plt.gca().set_title(f"{frq}, {rec}")
    save_and_close(plt.gcf(), rec, "magnitude_pdf", frq, save_path)

    ax = Moments_vs_FR(nn, fou_alt_c, T)
    plt.gca().set_title(f"{frq}, {rec}")
    save_and_close(plt.gcf(), rec, f"Moments_vs_fr_Q{Q}", save_path)
    
    ax = power_spectra(spks, f0=frq)
    ax.set_title(f"{frq}, {rec}")
    save_and_close(plt.gcf(), rec, f"spectra_Q{Q}", frq, save_path)
