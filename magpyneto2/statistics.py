import numpy as np
import pandas as pd
import scipy.stats
from tqdm.auto import trange
from scipy.integrate import quad
from scipy.stats import norm
from .utils import save_and_close

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_support():
    """Non-uniform support for Fourier coefficient distribution
    with high concentration near zero."""
    # XX = np.concatenate([
    #     # np.arange(0.0005, 0.2, 0.0005),
    #     np.arange(0.001, 6, 0.001)])
    XX = np.arange(0.001, 6, 0.001)
    return XX


def normalized_Fourier_PDF():
    """
    Uncorrected Fourier coefficient distributions
    You can either specify `xx` or `step`.
    `xx` allows you to have the support be non-uniformly spaced.
    Parameters
    ----------
    """
    xx = get_support()
    yy = xx * np.exp(-xx**2 / 2) 
    return xx, yy


def normalized_Fourier_CDF(X):
    return -np.exp(-X**2/2) + 1


"""Corrected Fourier coefficient distributions"""
def modulated_normalized_Fourier_PDF(r, u=0.0, v=1.0):
    """Calculate the probability density function of the normalized Fourier coefficient c-hat.

    Parameters
    ----------
    r : float
        The magnitude of the normalized Fourier coefficient.
    u : float
        The mean of the real part of the Fourier coefficient.
    v : float
        The variance of the real part of the Fourier coefficient.
    Returns
    -------
    float
        The probability density P(c-hat = r).
    """
    if r <= 0:
        return 0.0  # support is r > 0
    prefactor = r / (2 * np.pi * np.sqrt(v))

    def integrand(theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        exponent = -((x - u)**2) / (2 * v) - (y**2) / 2
        return np.exp(exponent)

    integral, _ = quad(integrand, 0, 2 * np.pi, limit=200)
    return prefactor * integral


def modulated_normalized_Fourier_PDF_arr(r_arr, u, v):
    return np.array([modulated_normalized_Fourier_PDF(r, u, v) for r in r_arr])


def get_epsilon(M):
    """
    Get correction factor for M independent samples of neighboring fourier coefficients
    
    Parameters
    ----------
    M: (int) Number of independent samples
    
    Returns
    -------
    eps: (float) Correction factor
    """
    return 1 / (2 * np.sqrt(M))


def normalized_Fourier_PDF_corrected(q_vals, r_vals, p_r_vals, eps):
    """
    Compute p(q) ≈ convolution of p(r) with log-normal noise from s ~ N(1, eps^2)
    
    Parameters:
        q_vals: array of q points (output)
        r_vals: array of r points (support of p_r)
            This is essentially the same as q_vals, but must cover the support of p_r
        p_r_vals: array of p(r) evaluated at r_vals
        eps: standard deviation of log-space Gaussian noise (eps << 1)
        
    Returns:
        p_q_vals: array of p(q) evaluated at q_vals
    """
    # Ensure strictly positive values
    r_vals = np.asarray(r_vals); p_r_vals = np.asarray(p_r_vals); q_vals = np.asarray(q_vals)
    assert np.all(r_vals > 0) and np.all(q_vals > 0)
    
    log_r = np.log(r_vals); log_q = np.log(q_vals)
    dr = np.diff(r_vals).mean()

    # Allocate output
    p_q_vals = np.zeros_like(q_vals)

    for i, lq in enumerate(log_q):
        kernel = norm.pdf(lq - log_r, loc=0, scale=eps) / q_vals[i]
        p_q_vals[i] = np.sum(p_r_vals * kernel) * dr

    return p_q_vals


def normalized_Fourier_CDF_corrected(PDF_vals, r_vals):
    """
    Compute CDF from PDF using numerical integration.
    
    Parameters:
        PDF_vals: array of p(r) evaluated at r_vals
        r_vals: array of r points (support of p_r)
    """
    return np.cumsum(PDF_vals[1:]* np.diff(r_vals))


def interpolate_corrected_CDF(R, CDF, points):
    """
    Evaluate CDF at a specific point (ie, c-hat value)
    """
    return np.interp(points, R[1:], CDF)


def inverse_Rayleigh_CDF(CDF_point:float, eps:float=0.0):
    """in: CDF value (quantile)
    out: distribution value (c-hat)
    Note: This is necessarily quantized. 
    Parameters
    ----------
    CDF_point: (float)  CDF value in (0,1)
    eps: (float) correction factor for dependent samples
    R: (np.array) support points for corrected distribution (only needed if eps>0)
    """
    assert CDF_point > 0 and CDF_point < 1, "CDF must be in (0,1)"
    if eps == 0.0:
        return np.sqrt(-2 * np.log(1-CDF_point)) 
    elif eps > 0.0:
        # Corrected distribution
        R, YY_uncorrected = normalized_Fourier_PDF()

        PDF = normalized_Fourier_PDF_corrected(R, R, YY_uncorrected, eps)
        CDF = normalized_Fourier_CDF_corrected(PDF, R)

        # Find point in CDF function estimate
        changepoint = np.where(np.diff(CDF > CDF_point))[0]

        # Find the R value corresponding to this CDF value
        inv_CDF = R[1:][changepoint]
        
        return inv_CDF


def latesttime(spks):
    """ Returns the latest time in an array of spike trains
    
    Parameters
    ----------
    spks: list(np.ndarray)
        list of spiketrains from a given recording
    
    Returns
    -------
    Output: float
        time of last spike in all the spike trains
    """
    return np.max(np.concatenate(spks))


def earliesttime(spks):
    """"""
    return np.min(np.concatenate(spks))

    
def clip(spks, T0, T1):
    ''' Selects spikes with times T0<=t<T.
    
    Parameters
    ----------
    spks: list(np.ndarray)
        list of spiketrains from a given recording
    T0: float
        Start time
    T1: float
        End time
     
    Returns
    -------
    Output: list(list)
        List of lists of spike times
    '''
    return [ [ t for t in tt if t>=T0 and t<T1 ] for tt in spks ]


# Gently modified from "falsepos.py"
def fourier(ss, freq):
    '''Calculates the complex fourier component of the point process SS at frequency `freq`.
    
    Parameters
    ----------
    ss: list
        Spike times in seconds
    freq: int/float
        Frequency of stimulus in Hz
    
    Returns
    -------
    Output: numpy.complex128
        Fourier coefficient, $\hat{c}$
    
    Notes
    -----
    z = FOURIER(ss, freq)
    Result is _not_ normalized to the number of spikes.
    '''
    return np.sum(np.exp(-2 * np.pi * 1j * np.array(ss) * freq))

def fourier_empirical(phases):
    '''Calculates the complex fourier component of the point process SS at frequency `freq`.
    Must only be used for the on-frequency calculation.
    Parameters
    ----------
    phases: list
        Spiketrain in phases (radians)
    
    Returns
    -------
    Output: numpy.complex128
        Fourier coefficient, $\hat{c}$
    
    Notes
    -----
    z = FOURIER(phases)
    Result is _not_ normalized to the number of spikes.
    '''
    return np.sum( np.exp( -1j * np.array( phases ) ) )


def fouriers(ss, freqs):
    """ 
    Run `fourier()` on collection of spiketrains `ss` for each `freq` in `freqs`
    Parameters
    ----------
    ss: list
        Spike times in seconds
    freqs: int/float
        Stimulation frequency (Hz)
    
    Returns
    -------
    Output: np.array(numpy.complex128)
        Complex fourier coefficients for spiketrain `ss` at each frequency `freqs`
    """
    return np.array([fourier(ss, freq) for freq in freqs])


def allfourier(spks, freq):
    """Returns complex Fourier coefficients at frequency `freq` for each spiketrain `ss` in `spks`
    
    Parameters
    ----------
    spks: list(lists)
        List of spiketrains in seconds
    
    freq: int/float
        Stimulation frequency in Hz
    
    Returns
    -------
    Output: np.array
        Array of complex Fourier coefficients for each spiketrain `ss` at frequency `freq`
    """
    return np.array([fourier(ss, freq) for ss in spks])


def allfouriers(spks, freqs):
    """Returns complex Fourier coefficients at frequency `freq` for each spiketrain `ss` in `spks`
    
    Parameters
    ----------
    spks: list(lists)
        List of spiketrains in seconds
    
    freq: list
        List of frequencies to analyze
    
    Returns
    -------
    Output: np.array(np.array)
        Array of arrays of complex Fourier coefficients for each spiketrain `ss` at each frequency `freq` in `freqs`
    """
    return np.array([fouriers(ss, freqs) for ss in spks])


def allfourier_empirical(phase_l):
    """Returns complex Fourier coefficients at frequency `freq` for each spiketrain `ss` in `spks`
    
    Is at a given frequency, no need to specify it.
    
    Parameters
    ----------
    phase_l: list(lists)
        List of spiketrains in phases (radians)
    
    
    Returns
    -------
    Output: np.array
        Array of complex Fourier coefficients for each spiketrain `ss` at frequency `freq`
    """
    return np.array([fourier_empirical(phases) for phases in phase_l])


def frequencies(T, sr=30_000):
    """
    Returns a list of frequencies that can be analyzed given the length of the recording `T`
    """
    return np.arange(sr)/T


def get_sgm(fou_alt_c):
    complex_fac = fou_alt_c[:,:,0] + fou_alt_c[:,:,1] * 1j
    sgm = np.sqrt(1/2 * np.mean(np.abs(complex_fac)**2, axis=1))
    return sgm


def get_c_hat(fou0, fou_alt_c):
    sgm = get_sgm(fou_alt_c)
    # Normalize the normal distribution
    return np.abs(fou0.flatten()) / sgm


def fourier_analysis(spks, freq, idealized_or_empirical="ideal", phase_l=None, Q=100, sr=30_000, T=None, log=False):
    """Returns a tuple of various combinations of fourier coefficients
    
    Parameters
    ----------
    spks: list(np.array)
        List of spiketrains in seconds
    freq: int/float
        Stimulus frequency
    idealized_or_empirical: ("ideal"/"empiric")
        obvious
    phase_l: (array)
        array of phases
    Q: int
        Determines window size
    
    Returns
    -------
    C: int
        Number of units
    T: int
        Latest spike time of all units, rounded up to nearest second
    nn: np.array
        Number of spikes
    fff: np.array
        Frequencies to analyze
    i0: int
        Index of stimulus frequency
    ff_alt: np.array
        Off-frequencies
    fou0: np.array
        On-frequency complex Fourier coefficients for all neurons
    fou_alt: np.ndarray
        C x len(ff_alt) Off-frequency complex Fourier coefficient magnitudes for off-frequencies for all neurons 
    fou_alt_c: np.ndarray
        For all neurons, all off-frequencies, all components (real and imaginary combined)
        """
    C = len(spks)
    
    # Get latest spike time in seconds, rounded up to nearest s
    if T==None:
        T = np.ceil(latesttime(spks) - earliesttime(spks))
        
    # Get number of spikes of each spike train
    nn = np.array([ len(s) for s in spks])
    if log:
        print(f'{np.mean(nn):.1f} ± {np.std(nn):.1f} spikes in {C} cells. T: {T}')
    
    # Available frequencies
    fff = frequencies(T, sr=sr)
    
    # Index of stimulus frequency
    i0 = np.argmin(np.abs(fff - freq))
    
    # Frequencies for off-frequencies
    ff_alt = np.array([ fff[i] for i in range(i0-Q, i0+Q+1) if (i != i0) and (i >= 0) ])

    # Get Fourier coefficients for the stimulus frequency for each spiketrain
    if idealized_or_empirical == "ideal":
        fou0 = allfourier(spks, freq).reshape(C,1) / T
    elif idealized_or_empirical == "empiric":
        fou0 = allfourier_empirical(phase_l).reshape(C,1) / T
    else:
        raise ValueError("`idealized_or_empirical` must be 'ideal' or 'empiric'")

    # Fourier coefficients for all off-frequencies
    fou_alt = allfouriers(spks, ff_alt) / T
    fou_alt_c = np.dstack((np.real(fou_alt), np.imag(fou_alt)))
    
    c_hat = get_c_hat(fou0, fou_alt_c)

    return (C, T, nn, fff, i0, ff_alt, fou0, fou_alt, fou_alt_c, c_hat)


def get_excess_suspect_stats(rr, crossing_percentile, conf_int_α=0.05):
    """
    Puts confidence bounds on excess counts
    
    Parameters
    ----------
    rr: (np.array) Normalized Fourier coefficient magnitudes
    crossing_percentile: (float) Percentile above which to count crossings empirically and theoretically
    α: (float) significance level for confidence intervals 
        
    Outputs
    -------
    n_empirical: (int) Number of coefficients above the confidence bound
    f_expected: (float) Expected number of coefficients above the confidence bound.
    f_{lo/hi}: (floats)
    
    Notes
    -----
    Compute excess count
    `excess_count = n_empirical - f_expected`
    """
    # Theoretical stats
    K = len(rr)
    f_expected = K * (1-crossing_percentile)

    binom = scipy.stats.binom(K, 1-crossing_percentile)
    f_lo = binom.ppf(conf_int_α/2) - f_expected
    f_hi = binom.ppf(1-conf_int_α/2)  - f_expected
    
    # 
    inverse_cdf_val = inverse_Rayleigh_CDF(crossing_percentile)
    n_empirical = len(rr[rr>inverse_cdf_val])

    return n_empirical, f_expected, f_lo, f_hi


def get_suspect_stats(rr, crossing_percentile:float, conf_int_α:float=0.05, eps:float=0.0):
    """
    Puts confidence bounds on excess counts
    
    Parameters
    ----------
    rr: (np.array) Normalized Fourier coefficient magnitudes
    crossing_percentile: (float) Percentile above which to count crossings empirically and theoretically
    α: (float) significance level for confidence intervals 
        
    Outputs
    -------
    n_empirical: (int) Number of coefficients above the confidence bound
    f_expected: (float) Expected number of coefficients above the confidence bound.
    f_{lo/hi}: (floats)
    eps: default=None (float) Correction factor for dependent samples. If None, no correction is applied.
    """
    # Clean rr of nans
    rr = rr[~np.isnan(rr)]
    # Theoretical stats
    K = len(rr)
    f_expected = K * (1-crossing_percentile)

    binom = scipy.stats.binom(K, 1-crossing_percentile)
    f_lo = binom.ppf(conf_int_α/2) 
    f_hi = binom.ppf(1-conf_int_α/2) 
    
    # 
    inverse_cdf_val = inverse_Rayleigh_CDF(crossing_percentile, eps=eps)
        
    n_empirical = len(rr[rr>inverse_cdf_val])

    return n_empirical, f_expected, f_lo, f_hi


def warp_mod(t,amp,pe,ph):
    '''
    Applies a periodic time warp to a spike train
    t = list of spike times in s
    amp= modulation amplitude, in [0,1]
    pe = modulation period in s
    ph = modulation phase in radians
    returns time-warped spike times
    '''
    from scipy import interpolate
    
    om=2*np.pi/pe # angular frequency
    y=np.linspace(0,pe,100) # new time axis
    x=y+amp/om*np.sin(om*y-ph) # old time axis
    f=interpolate.interp1d(x,y) # function to get new time from old time over one period of modulation
    u=(t//pe)*pe+f(t%pe) # have to apply the time warp only in the last period
    return u # return the warped spike times


def modulate(tt, f0, A, phi=None):
    """"""
    tt1 = np.array(tt).copy()
    if phi is None:
        phi = np.random.random(1)*2*np.pi
    pp = (np.cos(2*np.pi*f0*tt1+phi) + 1)*A/2
    xx = np.random.random(tt1.shape)
    tt1[xx<pp] += .5/f0
    return tt1


def modulate_phase(tt, A):
    """
    Parameters
    ----------
    tt: array-like
        List of instantaneous phases for spikes
    A: float
        Modulation strength

    Returns
    -------
    tt1: array-like
        modulated spike train (in radians)
    """
    tt1=tt.copy()

    # Modulation probabilities
    pp = (np.cos(tt1) + 1) * A / 2

    # Random range
    xx = np.random.random(tt1.shape)

    # Shift over by half a phase probablistically
    tt1[xx<pp] += np.pi

    # Modulus back into the range of radians
    tt1 = tt1 % (np.pi*2)
    return tt1


def gaussianinterp(xx, dat_x, dat_y, smo_x, err=False):
    '''GAUSSIANINTERP - Interpolate data using a Gaussian window
    
    Interpolation through irregularly spaced data
    
    yy = GAUSSIANINTERP(xx, dat_x, dat_y, smo_x) produces a smooth interpolation of the data: y(x) is estimated from all data points, weighing them based on their distance to x:
                                                         
                     -½ (xᵢ ‎- x)² / smo²  
            sumᵢ yᵢ e
    y(x) = ------------------------------                
                      -½ (xᵢ ‎- x)² / smo²  
            sumᵢ    e

    where xᵢ are the elements of DAT_X and yᵢ are the elements of DAT_Y.
    Current implementation assumes data is 1D.
    Algorithm is not fast: Time is O(X*D) where X is the length of XX and D is the length of DAT_X.'''

    N = len(xx)
    yy = np.zeros(xx.shape, xx.dtype)
    for n in range(N):
        wei = np.exp(-.5*np.abs(dat_x - xx[n])**2 / smo_x**2)
        yy[n] = np.sum(dat_y*wei) / np.sum(wei)

    if not err:
        return yy

    sy = np.zeros(xx.shape, xx.dtype)
    y_i = np.interp(xx, yy, dat_x)
    s_i = dat_y - y_i
    bad = np.nonzero(np.isnan(s_i))
    s_i[bad] = 0
    for n in range(N):
        wei = np.exp(-.5*np.abs(dat_x - xx[n])**2 / smo_x**2)
        wei[bad] = 0
        wei /= np.sum(wei)
        eff_n = 1/np.max(wei)
        sy[n] = np.sqrt(np.sum(wei*s_i**2)) / np.sqrt(eff_n)
    return yy, sy

    
def detect_modulation(modulation_df, full_log_dict, key, mods=np.arange(.01, 0.51, 0.01), Z=1000):
    """
    Parameters
    ----------
    modulation_df: pandas.DataFrame
        Output of _____
    full_log_dict: pandas.DataFrame
        Output of ____
    key: (string, int)
        (recording, frequency)
    Z: int
        Number of bootstrap iterations
    mods: array-like
        Modulation levels
    
    Outputs
    -------
    rr: modulated spiking level
    """
    # Patience required, this takes about an hour on my laptop
    theta_spks = [subdf.phase.values for _, subdf in modulation_df.loc[modulation_df.rec==key[0]].groupby("id")]

    sgm = np.std(full_log_dict[key]["fou_alt_c"], (1,2))
    
    M = len(mods)

    # Initialize 3d matrix of data
    C = len(theta_spks)
    T = np.floor(latesttime([modulation_df.spk.values]) - earliesttime([modulation_df.spk.values]))

    rr = np.zeros((C,M,Z))

    # For each modulation level,
    for m in trange(M):
        # For each bootstrap interval,
        for z in trange(Z, leave=False):
            # Modulation each spiketrain at mod level
            theta_spks_mod = [ modulate_phase(sp, mods[m]) for sp in theta_spks]
            
            # Get Fourier coeffient normalized and apparently divided by T...
            rr[:,m,z] = np.abs(allfourier_empirical(theta_spks_mod)) / T / sgm
    return rr


def get_confidence_limits(mod_r, modulation_df, full_log_dict, key, p = 0.05, Z=1000, mods=np.arange(0.01, 0.501, 0.01)):
    """
    Parameters
    ----------
    mod_r: (np.ndarray)
    p: (float): Significance level
    Z: (int) Number of bootstrap iterations
    mods: (np.array) levels of modulation
    
    """
    key_df = modulation_df.loc[modulation_df.rec==key[0],:]
    
    # Normalizing factor
    sgm = np.std(full_log_dict[key]["fou_alt_c"], (1,2))

    # Renormalize to match rr
    r0 = np.abs(full_log_dict[key]["fou0"][:,0])/sgm 


    # Unknown
    K = 10

    # Number of spikes for each unit
    nn = [len(subdf) for id, subdf in key_df.groupby("id")]

    # Number of cells
    C = len(nn)

    # 
    rr1 = mod_r.reshape(C,50,K,Z//K)

    # Get T
    T = key_df.spk.max() - key_df.spk.min()

    # Get array of nans
    mm = np.zeros((C,K)) + np.nan

    for c in range(C):
        for k in range(K):
            if nn[c]>=T * 0.5:
                pp = np.mean(rr1[c,:,k,:] < r0[c], -1)
                where = np.argwhere(pp>p)
                if len(where)==0:
                    mm[c,k] = mods[0]
                else:
                    mmax = np.max(where) 
                    mm[c,k] = mods[mmax] + .01

    m0 = mm.mean(-1) #  + np.random.random(C)*.01-.005
    dm = mm.std(-1)   
    
    return m0, dm, nn, T

"""VIZ"""
def lighten(cc, factor=2):
    """Lower the alpha of a color
    r, g, b = cc
    r = 1 - .5*(1-r)
    g = 1 - .5*(1-g)
    b = 1 - .5*(1-b)
    return r,g,b """
    cc[3] = cc[3] / factor
    return cc
    

def plot_power_by_freq(ff_alt, fou_alt, fou0, frq, kk=None, axes=None):
    """Plot Real component against frequency
    Parameter
    ---------
    ff_alt: list
        List of off-frequency component
    """
    if axes is None:
        fig, axes =plt.subplots(2, 1, figsize=(8,8))
        
    # Plot Fourier coefficients for each off-frequency
    if kk is None:
        kk = np.arange(len(fou_alt))
    
    colors = plt.cm.viridis(np.linspace(0.1,0.9,len(fou_alt)))
    for i, k in enumerate(kk):
        axes[0].plot(ff_alt, np.real(fou_alt[k,:]), '.', color=colors[i], markersize=0.3)
        axes[1].plot(ff_alt, np.imag(fou_alt[k,:]), '.', color=colors[i], markersize=0.3)
    
    axes[0].plot(np.ones(len(fou0)) * frq, np.real(fou0), 'k.', markersize=0.3)
    axes[1].plot(np.ones(len(fou0)) * frq, np.imag(fou0), 'k.', markersize=0.3)
    # 
    axes[1].set_xlabel('Frequency (Hz)')
    
    [ax.set_ylabel(f'{label} component of /c/_/n/') for ax, label in zip(axes, ["real", "imaginary"])]
    return axes 
    
    
def plot_magnitude_cdf(ff_alt, fou_alt, kk=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))    
    
    # Plot Fourier coefficients for each off-frequency
    if kk is None:
        kk = np.arange(len(fou_alt))
        
    colors = plt.cm.viridis(np.linspace(0.1,0.9,len(fou_alt)))

    for i,k in enumerate(kk):
        yy = np.real(fou_alt[k,:])
        zz = np.imag(fou_alt[k,:])
        yy = np.hstack((yy, zz))
        xx = np.sort(yy)
        sgm = np.std(xx)
        yy = np.arange(len(xx))
        yy = yy/yy[-1]
        xxx = np.arange(-1, 1.0001, .001)
        yyy = np.cumsum(np.exp(-.5*xxx**2/sgm**2))
        yyy = yyy / yyy[-1] # Lazy way to normalize

        use = np.abs(xxx)<=.6
        ax.plot(xxx[use], yyy[use], color=lighten(cc[i]), linewidth=3)

        use = np.abs(xx)<=.6
        ax.plot(xx[use], yy[use], '.', markersize=4, color=cc[i])
    return ax
    

def plot_magnitude_pdf(ff_alt, fou_alt, kk=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))    

    # Plot Fourier coefficients for each off-frequency
    if kk is None:
        kk = np.arange(len(fou_alt))
    
    colors = plt.cm.viridis(np.linspace(0.1,0.9,len(fou_alt)))

    # For each k (n-th spiketrain in a length of spiketrains)
    for i,k in enumerate(kk):

        # Real component
        yy = np.real(fou_alt[k,:])

        # Imaginary component
        zz = np.imag(fou_alt[k,:])

        # For some reason we are concatenating the real and imaginary components
        yy = np.hstack((yy, zz))

        # STD
        sgm = np.std(yy)
        dx = .01

        # Plot PDF
        yy, xx = np.histogram(yy, np.arange(-.6,.6001, dx))
        xx = (xx[:-1]+xx[1:])/2
        xxx = np.arange(-1, 1.0001, .001)
        yyy = np.exp(-.5*xxx**2/sgm**2)
        yyy = yyy * np.sum(yy) / np.sum(yyy) * dx/.001

        use = np.abs(xxx)<=.6
        plt.plot(xxx[use], yyy[use], color=colors[i], linewidth=3)

        # Plot histogram
        use = np.abs(xx)<=.6
        plt.bar(xx,yy,dx, facecolor=colors[i], alpha=.5)

    plt.xlabel('Real or imaginary component of c_n')
    plt.ylabel('Number of instances')
    
    return ax
    

def plot_coefficient_cdf(ff_alt, fou_alt, kk=None, ax=None):
    """Plot Real component against frequency
    Parameter
    ---------
    ff_alt: list
        List of off-frequency component
    """
    
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    
    # Plot Fourier coefficients for each off-frequency
    if kk is None:
        kk = np.arange(len(fou_alt))
    
    colors = plt.cm.viridis(np.linspace(0.1,0.9,len(fou_alt)))

    # For each cell of interest
    for i,k in enumerate(kk):
        yy = np.hstack(
            (np.real(fou_alt[k,:]),
             np.imag(fou_alt[k,:]))
        )
        
        xx = np.sort(yy)
        sgm = np.std(xx)
        yy = np.arange(len(xx))
        yy = yy/yy[-1]
        xxx = np.arange(-1, 1.0001, .001)
        yyy = np.cumsum(np.exp(-.5*xxx**2/sgm**2))
        yyy = yyy / yyy[-1] # Lazy way to normalize

        # Filter
        use_xxx = np.abs(xxx)<=.6
        use_xx = np.abs(xx)<=.6
        
        # 
        ax.plot(xxx[use_xxx], yyy[use_xxx], color=lighten(colors[i]), linewidth=3)
        ax.plot(xx[use_xx], yy[use_xx], '.', markersize=4, color=colors[i])
    return ax


def Moments_vs_FR(nn, fou_alt_c, T, ax=None):
    """
    Plots first four moments (mean, STD, skewness, and kurtosis) of fourier coefficients (real and imaginary pushed together) against firing rate
    Figures 5C through 5F
    Parameters
    ----------
    nn: np.ndarray
        Array of firing rates
    fou_alt_c: np.ndarray
        Concatenated list of real and imaginary components of Fourier coefficients for all cells in a recording at a given stimulus frequency
    
    Returns
        ax: plt.axis
            Figure axis
    """
    
    if ax is None:
        _, axes = plt.subplots(4,1, figsize=(5, 10))
    
    # Use (i.e., filter)
    use = nn > 0.5*T
    
    # Shape
    C,N,X=fou_alt_c.shape
    
    # Set up metadata for for loop
    functions = [
        np.mean,
        np.std,
        scipy.stats.skew,
        scipy.stats.kurtosis
    ]
    
    inputs = [
        (fou_alt_c, (1,2)),
        (fou_alt_c, (1,2)),
        (fou_alt_c.reshape(C,N*X), 1),
        (fou_alt_c.reshape(C,N*X), 1)
    ]
    
    xlabels = ["mean", "SD", "Skew", "Kurtosis"]
    
    for ax_ind, ax in enumerate(axes):
        # Current function
        function = functions[ax_ind]
        
        # Current input
        arg = inputs[ax_ind]
        
        # Current label
        label = xlabels[ax_ind]
        
        # Plot
        ax.semilogx(
            nn[use], 
            function(*arg)[use], 
            "o", color=(0,.4,1)
        )
        
        # Label axes 
        ax.set_ylabel(f"{label}", fontsize=15)
        
        # Optional zero line
        if label != "SD":
            ax.plot([.5*T, 100*T], [0,0],'k')
        
    ax.set_xlabel('Firing rate (1/s)', fontsize=15)
    axes[0].set_title("Real and imaginary c_n components")
    return axes


def visualize_fourier_df(fourier_df_arg, XX, YY):
    """Generates diagnostic visualizations of fourier dataframes
    
    Parameters
    ----------
    fourier_df_arg: (pd.DataFrame) Input fourier_df
    XX, YY: (np.array) Data for plotting the Rayleigh distribution
    """
    f_list = []
    for (frq), freq_df in fourier_df_arg.groupby("freq"):       
        for (rec, rec_df) in freq_df.groupby("rec"):
            # Get epsilon
            epsilon = get_epsilon(rec_df.Q.values[0])

            # Get corrected distribution
            YY_corrected = normalized_Fourier_PDF_corrected(XX, XX, YY, epsilon)
            
            fig, ax = plt.subplots()

            # Unpack data relevant to this particular recording
            rr = rec_df.rr.values
            abridged_rec = rec.split('\\')[-1].split('/')[-1]

            n_empirical, f_expected, l_bound, h_bound = get_suspect_stats(rr, .95)
            radius = (h_bound-l_bound) / 2

            plt.hist(rr, bins=np.arange(0, 6, .1), density=True)
            plt.plot(XX, YY, label="Uncorrected PDF")
            plt.plot(XX, YY_corrected, label="Corrected PDF")
            plt.title(rec)
            
            # 
            ax.set_xlabel('$|\hat{c}|$')
            ax.set_ylabel('Proportion of units')
            print(l_bound, h_bound)
            title =  f"{abridged_rec} n={len(rr)}" + "\n"
            title += f"{frq} Hz: {n_empirical} - {np.round(f_expected, 2)} = $"
            excess = n_empirical-f_expected
            title += str(np.round(excess, 2)) + "_{" + str(np.round(excess - radius, 2))
            title += "}^{" + str(np.round(excess + radius, 2)) + "}$"
            
            ax.set_title(title)
            f_list.append((rec, frq, fig))
        
    return f_list
        

def add_legend_label(color, label, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(
        [xlim[0]-10, ylim[0]-10],
        [xlim[0]-10, ylim[0]-10],
        color=color, label=label)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=10)


def power_spectra(spks, f0=None, df=0.3, f_lo=0.3, f_hi=20):
    """Plain FFT power spectrum per spike train, log-log, with stimulus band marked.

    Parameters
    ----------
    spks : list of np.ndarray  spike times in seconds
    f0   : float or None       stimulus frequency to highlight
    df   : float               half-width of highlight band around f0
    f_lo, f_hi : float         frequency display range (Hz)
    """
    dt = 0.02
    T = latesttime(spks) - earliesttime(spks)
    xx = np.arange(0, T + 0.0001, dt)

    yyy = []
    for tt in spks:
        if len(tt) > 1000:
            yy, _ = np.histogram(tt, xx)
            yyy.append(yy - np.mean(yy))
        if len(yyy) > 10:
            break

    fig, ax = plt.subplots()

    if not yyy:
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (a.u.)')
        return ax, np.array([]), np.array([]), None

    N = len(yyy[0])
    ff = np.fft.rfftfreq(N, d=dt)
    # (freq_bins, n_units) power array
    pxx = np.array([np.abs(np.fft.rfft(y)) ** 2 / N for y in yyy]).T

    use = (ff >= f_lo) & (ff <= f_hi)

    if f0 is not None:
        ymin, ymax = pxx[use].min(), pxx[use].max()
        ax.fill_between([f0 - df, f0 + df], [ymin, ymin], [ymax, ymax],
                        facecolor=(0.7, 0.8, 1), zorder=0, label=f'{f0} Hz')

    for k in range(pxx.shape[1]):
        ax.loglog(ff[use], pxx[use, k], alpha=0.6, linewidth=0.8)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (a.u.)')
    return ax, ff, pxx, None


def sanity_check_raw_data(θ, period_crossings, sts, n_representative_units=7, sampling_rate=30_000):
    """
    Usage:
    ------
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
    [ax.set_xticklabels(ax.get_xticks() // sampling_rate) for ax in axes]

    plt.tight_layout()
    return fig, axes


def find_outliers(df, Q=100, sr=30_000, method="ideal", diagnostics=True):
    """Runs fourier_analysis on a dataframe of spiking data

    Parameters
    ----------
    df: (pd.DataFrame) `df.columns` yields `Index(['period', 'spk', 'phase', 'freq', 'id', 'rec'], dtype='object')`
    Q: (int) Determines window size
    sr: (int) sampling rate
    diagnostics: (bool) whether to output diagnostic plots
    
    Returns
    -------
    fourier_df: (pd.DataFrame) Indexes coefficients by unit index and so on
    """
    log_dict = {}
    fourier_l = []
    
    for (rec, frq), subdf in df.groupby(["rec", "freq"]):
        print("analyzing", rec, "at", frq, "Hz")

        # Put it into form fourier_analysis asks
        # Get even trials for test
        spks_and_ids_and_phases = [
            (
                id_, np.sort(id_subdf.spk.values), id_subdf.phase.values
            ) for id_, id_subdf in subdf.groupby("id")]

        # If empty, add a single spike
        spks = [spks if len(spks) > 0  else np.array([0]) for (_, spks, _) in spks_and_ids_and_phases]
        phase_l = [phases if len(phases) > 0 else np.array([0]) for (_, _, phases) in spks_and_ids_and_phases]
        ids = [id_ for (id_, _, _) in spks_and_ids_and_phases]

        # Run Fourier 1F
        (C, T, nn, fff,
         i0, ff_alt, fou0,
         fou_alt, fou_alt_c, c_hat) = fourier_analysis(
            spks, frq, idealized_or_empirical=method,
            phase_l=phase_l, Q=Q, sr=sr, T=(1/frq) * np.max(subdf.period)
        )

        sigma_1F = get_sgm(fou_alt_c)
        
        log_dict[(rec, frq)] = {
            "C": C, "T":T, "nn":nn, "fff":fff,
            "i0":i0, "ff_alt":ff_alt, "fou0":fou0,
            "fou_alt": fou_alt, "fou_alt_c": fou_alt_c, "args":(spks, frq, Q)
        }

        # Run Fourier 2F
        (twoF_C, twoF_T, twoF_nn, twoF_fff, 
         twoF_i0, twoF_ff_alt, twoF_fou0,
         twoF_fou_alt, twoF_fou_alt_c, twoF_c_hat) = fourier_analysis(
            spks, frq * 2, idealized_or_empirical="ideal",
            phase_l=phase_l, Q=Q, sr=sr, T=(1/frq) * np.max(subdf.period)
        )
        sigma_2F = get_sgm(twoF_fou_alt_c)

        
        log_dict[("twoF_" + rec, "twoF_"+str(frq))] = {
            "C": twoF_C, "T":twoF_T, "nn":twoF_nn, "fff":twoF_fff,
            "i0":twoF_i0, "ff_alt":twoF_ff_alt, "fou0":twoF_fou0,
            "fou_alt": twoF_fou_alt, "fou_alt_c": twoF_fou_alt_c,
            "args":(spks, frq * 2, Q)
        }
        
        if diagnostics:
            # Save diagnostic plots
            ax = plot_power_by_freq(ff_alt, fou_alt, fou0, frq)
            plt.gca().set_title(f"{frq}, {rec}")
            save_and_close(plt.gcf(), rec, f"power_by_freq_Q{Q}", frq)
            
            ax = plot_coefficient_cdf(ff_alt, fou_alt)
            plt.gca().set_title(f"{frq}, {rec}")
            save_and_close(plt.gcf(), rec, f"coefficient_cdf_Q{Q}", frq)
            
            ax = plot_magnitude_pdf(ff_alt, fou_alt)
            plt.gca().set_title(f"{frq}, {rec}")
            save_and_close(plt.gcf(), rec, "magnitude_pdf", frq)
            
            ax = Moments_vs_FR(nn, fou_alt_c, T)
            plt.gca().set_title(f"{frq}, {rec}")
            save_and_close(plt.gcf(), rec, f"Moments_vs_fr_Q{Q}", frq)
            
                    
        # Standard deviation of the surrounding frequencies' coefficients
        eps = get_epsilon(Q)
        R, YY_uncorrected = normalized_Fourier_PDF()
        PDF = normalized_Fourier_PDF_corrected(
            R[1:], R[1:], YY_uncorrected[1:], eps)
        CDF = normalized_Fourier_CDF_corrected(PDF, R[1:])
        pp = 1 - np.interp(c_hat, R[2:], CDF)
        twof_pp = 1 - np.interp(twoF_c_hat, R[2:], CDF) 
        
        fourier_l.append(pd.DataFrame({
            "id": ids, "pp":pp, "nn":nn, "rr":c_hat,
            "freq":frq, "rec":rec, "2f_rr":twoF_c_hat, "2f_pp":twof_pp, 
            "sens":nn/2/sigma_1F, "sens_2f":nn/2/sigma_2F}))
    
    fourier_df = pd.concat(fourier_l)
    fourier_df["Q"] = Q # Save number of neighbors used
    return fourier_df, log_dict


def visualize_modulations(mod_rr, m0, dm, nn, T):
    fig, ax = plt.subplots(figsize=(10,7))
    for c in range(len(mod_rr)):
        plt.semilogx(
            nn[c]/T, m0[c],
            np.zeros(2)+nn[c]/T, [m0[c]-dm[c], m0[c]+dm[c]], # Removing confidence intervals
             'k')
    
    plt.semilogx(nn/T, m0, 'k.')
    plt.xlabel('Firing rate (/s)')
    plt.ylabel('95% conf lim on modulation')
    plt.title(f"5% detection level, {np.round(T)} s duration, \n {len(nn)} units")
    ax.hlines(0, *ax.get_xlim())
    return fig, ax


def visualize_detectability(mod_rr, nn, T):
    """
    rr: (array) Fourier coefficients
    nn: (array) spike numbers
    T: (float) duration
    """
    
    mods=np.arange(0.01, 0.501, 0.01)
    pp = 1 - inverse_Rayleigh_CDF(mod_rr)

    fig = plt.figure(figsize=(10,7))
    plt.clf()
    cc = [(0,0,.5), (1,.5,0), (.2,.7,.5)]
    nn = np.array(nn)
    C = len(nn)

    for f, alpha in enumerate([.0001, .001, .01]):
        confid = np.mean(pp[:,:,:] < alpha, -1)
        confid = np.hstack((confid, np.ones((C,1))))
        midx = np.argmax(confid>=.5, -1)
        midx[midx>=len(mods)] = -1
        mm = mods[midx]
        mm[midx<0] = .5

        use = np.logical_and(nn > .5*T, midx>=0)
        plt.semilogx(
            nn[use]/T, mm[use], 'o', 
            color=cc[f], alpha=.4,
             markersize=4)
        xx = 10**np.arange(np.log10(.6), np.log10(90), .02)

        # Ensure accurate interpolation:
        pass # change xx so that the min is >= min FR

        yy = gaussianinterp(np.log10(xx), np.log10(nn[use]/T), mm[use], .1)
        plt.plot(xx, yy, color=cc[f])
        use = np.logical_and(nn > .5*T, midx<0)
        plt.semilogx(nn[use]/T, mm[use], '^', color=cc[f], alpha=.4)
    
    plt.hlines(0, *plt.xlim())
    plt.xlabel('Firing rate (/s)')
    plt.ylabel("Minimum detectable modulation")
    plt.legend(('','α = 0.0001','', '','α = 0.001','', '','α = 0.01',''))
    plt.title(f"{np.round(T)} s duration, {len(nn)} units")
    return fig


def draw_hist(c_hat, ax, xlim=12.5, title=False, inset=True, invert=False):
    # Histogram
    XX, YY = normalized_Fourier_PDF()
    vals, bins = np.histogram(c_hat, bins=np.arange(0, 12, 0.2), density=True)
    if not invert:
        ax.bar(bins[:-1], vals, width=np.diff(bins)[0], align="edge")
        ax.plot(XX, YY, label="theoretical", color="k", linewidth=1)
        ax.axvline(inverse_Rayleigh_CDF(0.99), color="r")
    else:
        ax.barh(bins[:-1], vals, height=np.diff(bins)[0], align="edge")
        ax.plot(YY, XX,  label="theoretical", color="k", linewidth=1)
        ax.axhline(inverse_Rayleigh_CDF(0.99), color="r")

    # Tidy x and y labels
    ax.set_xlabel("$\hat{c}$")
    ax.set_ylabel("PDF")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if xlim is not None:
        ax.set_xlim((0, xlim))

    if title:
        ax.set_title(f"N={len(c_hat)}")

    if inset:
        ax.annotate(
            f"N={len(c_hat)}", xy=(1,1), xycoords="axes fraction",
            xytext=(0,0), textcoords="offset points", ha="right", va="top")

    return vals, bins


def inset_hist(ax, vals, bins):
    x1, x2, y1, y2 = 2.5, 10, 0, .01
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47], xlim=(x1, x2), ylim=(y1, y2))
    ax.indicate_inset_zoom(axins, edgecolor="black")
    
    axins.bar(bins[:-1], vals, width=np.diff(bins)[0], align="edge")
    XX, YY = normalized_Fourier_PDF()
    axins.plot(XX, YY, label="theoretical", color="k", linewidth=1)
    axins.vlines(inverse_Rayleigh_CDF(0.99), *axins.get_ylim(), 'r')
    axins.set_xlim((x1, x2))
    axins.set_ylim((y1, y2))
    axins.set_xticks([x1, x2])
    axins.set_xticklabels([x1, x2])
    return axins


def plot_excess_counts(conf_ax, bigfig_df, area_line_level=-6, species_line_level=-10,  label_recs=False, ylim=(-12, 20)):
    
    jitter = False

    counter, counter_last, species_counter, species_counter_last = 0, 0, 0, 0
    xticks, xticklabels = [], []

    CMAP = cm.tab20b
    unique_freqs = bigfig_df.freq.unique()
    CMAP_colors = CMAP(np.arange(len(unique_freqs)))
    color_d = {freq: color for freq, color in zip(unique_freqs, CMAP_colors)}
    
    desired_order = [
        'Quail', 'Owl','mouse', 'zebra finch','Pigeon',  'zebrafish', 'medaka']

    for species in desired_order:
        species_df = bigfig_df.loc[bigfig_df.species == species, :]

        if len(species_df) == 0:
            continue

        species_counter_last = counter
        for area, area_df in species_df.drop_duplicates().groupby("area"):
            counter_last = counter
            for rec_ind, (rec, recdf) in enumerate(area_df.groupby("rec")):
                
                # raw data
                freq = recdf.freq.unique()[0]
                
                # Confidence bounds
                # Set by rounding up on order of magnitude of number of units in each recording; 
                # less than  one in a thousand.
                sig_thres = 0.99 # Set by the number of recordings; less than one in 100
                
                n_empirical,    _, f_lo, f_hi = get_suspect_stats(recdf["rr"].values,    sig_thres, conf_int_α=0.05)
                conf_ax.hlines(n_empirical,    counter+.25, counter+0.75, "black", zorder=2, linewidth=1, alpha=0.9)
                
                if ~np.all(np.isnan(recdf["2f_rr"].values)):
                    n_empirical_2f, _, _,    _    = get_suspect_stats(recdf["2f_rr"].values, sig_thres, conf_int_α=0.05)
                    conf_ax.hlines(n_empirical_2f, counter+.25, counter+0.75, "red", zorder=2, linewidth=1, alpha=0.9)
                
                conf_ax.plot(
                    [counter+0.5, counter+0.5], [f_lo, f_hi], color=color_d[freq],
                    alpha=.90, linewidth=3, solid_capstyle="butt", zorder=1)
                    
                counter += 1

                if n_empirical > ylim[1]:
                    # conf_ax.arrow(counter - 0.5, ylim[1] - 3, 0, 2, length_includes_head=True, head_width=.1)
                    conf_ax.annotate("", xytext=(counter - 0.5, ylim[1] - 3), xy=(counter - 0.5, ylim[1] - 1), arrowprops=dict(arrowstyle="-|>", color="k", edgecolor=None))
                    
                    jit = 5 if jitter else 5.5
                    jitter = not jitter
                    conf_ax.text(counter - 0.5, ylim[1] - jit, n_empirical, ha="center", rotation=90, fontsize=4)
                
                if label_recs:
                    conf_ax.text(counter, 10, rec.split('\\')[-1].split('/')[-1], ha="center", rotation=90)
            
            # Area lines
            conf_ax.hlines(area_line_level, counter_last+0.25, counter-0.25, "black",zorder=2, linewidth=1)
            
            # Annotate area
            if area in ("wholebrain", "whole brain"):
                area = "WB"
            
            raise_area = ('arcopallium' in area) or (("WB" in area) and (species=='zebrafish')) or ("thalamus" in area) or ("CB" in area)  or ("SC" in area)
            conf_ax.text((counter + counter_last)/2, area_line_level * 0.8 if raise_area else area_line_level * 1.4, area, ha="center", rotation=0)
        
        # Update species counter
        species_counter = counter

        # Add an xtick for the species
        xticks.append((species_counter + species_counter_last) / 2 )

        # xticklabels to lowercase
        if species.lower() == "zebra finch":
            xticklabels.append("zebra\nfinch")
        elif species.lower() == "medaka":
            xticklabels.append("\nmedaka")
        else:
            xticklabels.append(species.lower())

        # Draw species line
        conf_ax.hlines(species_line_level, species_counter_last+0.25, species_counter-0.25, "black",zorder=2, linewidth=1)
        
    # Set xticks and xticklabels for species
    conf_ax.set_xticks(xticks)
    conf_ax.set_xticklabels(xticklabels, rotation=0)

    # Get xlimit before adding data for the legend so you can reset it later
    xlim_conf = conf_ax.get_xlim()

    # Plant labels away from data
    freqs = bigfig_df.freq.unique()
    for freq in np.sort(freqs):
        conf_ax.plot([-10,-9], [-5,-6], color=color_d[freq], label=freq if freq >= 1 else np.round(freq, 4))

    # Set up legend
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    # conf_ax.legend(ncol=len(freqs), title="Hz", bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left')
    conf_ax.legend(ncol=2, title="Hz", bbox_to_anchor=(1,.5), loc="center left") 
    
    # Reset xlim
    conf_ax.set_xlim(xlim_conf)
    conf_ax.set_ylim(ylim)
    conf_ax.set_ylabel("# Suspects");

    # Remove negative xticks
    conf_ax.set_yticks(conf_ax.get_yticks()[conf_ax.get_yticks() >= 0])
    conf_ax.set_yticklabels(conf_ax.get_yticks())
    return conf_ax


def plot_combo_scatterplot(subdf:pd.DataFrame, ax:plt.Axes.axes,
                           legend_size:int=5, CDF_threshold=0.99,):
    """
    Parameters
    ----------
    subdf: pd.DataFrame
        Dataframe of interest
    ax: plt.axis
        Axis to plot on
    legend_size: int
        fontsize for legend
    """
    thres = inverse_Rayleigh_CDF(CDF_threshold)

    # Each combination of frequencies
    freqs = subdf.freq.unique()
    for freq2 in freqs[1:]:
        
        f1_arr, f2_arr = [], []
        for _, id_df in subdf.groupby("id"):
            if (freqs[0] in id_df.freq.values) and (freq2 in id_df.freq.values):
                f1_arr.append(id_df.loc[id_df.freq==freqs[0], "rr"].values[0])
                f2_arr.append(id_df.loc[id_df.freq==freq2, "rr"].values[0])
                    
        values = np.array([f1_arr, f2_arr]).T
        ax.scatter(
            values[values[:,0] > thres, 0], values[values[:,0] > thres, 1],
            s=10, label=f"{freqs[0]} Hz vs {freq2} Hz suspects", alpha=0.5)

        ax.scatter(
            values[values[:,0] < thres, 0], values[values[:,0] < thres, 1],
            s=10, c="grey", alpha=0.5)

        
    ax.set_xlabel(r"Suspect $\hat{c}$")
    ax.set_ylabel(r"Other trial $\hat{c}$")
    ax.set_xlim((0, 7.5))
    ax.set_ylim((0, 4.5))
    ax.set_yticks(ax.get_xticks())
    ax.set_xticks(ax.get_xticks())
    ax.legend(loc="upper right", fontsize=legend_size)
    ax.axvline(thres, color='k')
    ax.axhline(thres, color='k')



def get_poscontrols_negresults(all_fourier_df:pd.DataFrame):
    """
    Sorts values from all_fourier_df into positive and negative results
    
    Parameters
    ----------
    all_fourier_df: pd.DataFrame
        Dataframe of fourier results across all recordings
    
    Returns
    -------
    all_fourier_df_filtered_mag_exp: pd.DataFrame
        Dataframe of negative results
    all_fourier_df_filtered_pos_control: pd.DataFrame
        Dataframe of positive controls
    all_fourier_df_unique_pos_control: pd.DataFrame
        Dataframe of unique positive controls
    """

    # high spike count, permit lower for owl recordings
    nn_filter = ((all_fourier_df.nn.values > 50) | (all_fourier_df.species.values == "Owl"))
    include_fish = np.array([("fish" in elem) or (elem == "medaka") for elem in all_fourier_df.species.values])


    """Positive control"""
    # Without fish
    not_fish_pos_control = all_fourier_df.loc[
        nn_filter & ~include_fish & (
            np.array(["visual" in elem for elem in all_fourier_df.rec.values])
            | np.array(["oddball" in elem for elem in all_fourier_df.rec.values])
            | np.array(["WN" in elem for elem in all_fourier_df.rec.values])
            | np.array(["3D" in elem for elem in all_fourier_df.rec.values])
        ), :]

    # With fish
    fish_pos_control = all_fourier_df.loc[include_fish 
        & np.array(["nostim" not in rec for rec in all_fourier_df.rec.values])
        & (all_fourier_df.freq.values < 0.02) # include 1/60 Hz, 0.016667 Hz
        & ( 
            np.array(["visual" in rec for rec in all_fourier_df.rec.values])
            | np.array([rec.endswith("magneto_1.tif") for rec in all_fourier_df.rec.values])
            | np.array([rec.endswith("magneto_2.tif") for rec in all_fourier_df.rec.values])
        ), :]

    # Combine fish and not fish
    all_pos_control = pd.concat([
        fish_pos_control, not_fish_pos_control])

    # Some neurons will be repeated across experiments. Unique values for neurons, only counted once.
    all_unique_pos_control = all_pos_control.drop_duplicates(
        subset=["species", "date", "id"], keep="first")

    """Magnet experiments"""
    not_fish_mag_exp = all_fourier_df.loc[nn_filter & ~include_fish
        & (all_fourier_df.freq > 1)
        & np.array(["visual" not in elem for elem in all_fourier_df.rec.values])
        & np.array(["WN" not in elem for elem in all_fourier_df.rec.values])
        & np.array(["3D" not in elem for elem in all_fourier_df.rec.values]), :]

    # Expt 5 had the most units
    # What's going on here? isn't owl included above? 
    owl_mag_exp = all_fourier_df.loc[
        (all_fourier_df.species == "Owl")
        & (all_fourier_df.rec == "exp5"), :]

    fish_mag_exp = all_fourier_df.loc[
        include_fish
        & np.array(["nostim" not in rec for rec in all_fourier_df.rec.values])
        & np.array(["visual" not in elem for elem in all_fourier_df.rec.values])
        & (all_fourier_df.freq.values  > 0.02) # exclude 1/60 Hz, 0.016667 Hz
        & (np.array([elem.endswith("magnet") for elem in all_fourier_df.rec.values])
            | np.array([elem.endswith("magneto_0.tif") for elem in all_fourier_df.rec.values])), :]

    # Combine fish and not fish
    all_mag_exp = pd.concat([
        fish_mag_exp, not_fish_mag_exp, owl_mag_exp])
    
    return (all_mag_exp, all_pos_control, all_unique_pos_control)



def boundary_ticks(ax, x=True, y=True, xprec=0, yprec=0):
    if x:
        xlim = ax.get_xlim()
        ax.set_xticks([xlim[0], xlim[1]])
        ax.set_xticklabels([f"{np.round(xlim[0], xprec) if xprec>0 else int(xlim[0])}", f"{np.round(xlim[1], xprec) if xprec>0 else int(xlim[1])}"])
    
    if y:
        ylim = ax.get_ylim()
        ax.set_yticks([ylim[0], ylim[1]])
        ax.set_yticklabels([f"{np.round(ylim[0], yprec) if yprec>0 else int(ylim[0])}", f"{np.round(ylim[1], yprec) if yprec>0 else int(ylim[1])}"])


def nestle_labels(ax, x_offset=0, y_offset=0, y=True, x=True):    
    if y:
        ax.yaxis.set_label_coords(y_offset, 0.5)
    if x:
        ax.xaxis.set_label_coords(0.5, x_offset)
    
    
def boundarize_and_nestle(ax, x=True, y=True, xprec=0, yprec=0, x_offset=0, y_offset=0):
    boundary_ticks(ax, x=x, y=y, xprec=xprec, yprec=yprec)
    nestle_labels(ax, x_offset=x_offset, y_offset=y_offset, x=x, y=y)
    


def plot_spectrum(ax:plt.Axes.axes, fou_alt:np.complex64, win, stimulus_frequency,
    stimulus_frequency_power:np.complex64, legend=False, central_tendency=False):
    """Plots the spectrum of a fourier transformed signal around some frequency of interest
    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to plot on
    fou_alt : np.complex64
        The fourier transform of the signal
    win : np.array
        The window of frequencies
    stimulus_frequency : float
        The frequency of the stimulus
    stimulus_frequency_power : np.complex64
        The power of the stimulus frequency"""
    
    sgm_c = np.sqrt(.5 * np.mean(np.concatenate([fou_alt.real, fou_alt.imag])**2))
    ax.axhline(sgm_c)
    
    ax.plot(win, np.abs(fou_alt.real), ".", color="orange", alpha=0.5, markersize=1, label="real")
    ax.plot(win, np.abs(fou_alt.imag), ".", color="red", alpha=0.5, markersize=1, label="imaginary")

    markerline, stemline, baseline = ax.stem(
        [stimulus_frequency], np.abs(stimulus_frequency_power), "blue", bottom=sgm_c)
    

    ax.plot([stimulus_frequency], np.abs(stimulus_frequency_power), "blue", marker="o", alpha=.8, markersize=4,label=r"$|c_s|$")
    plt.setp(stemline, linewidth=1, color="blue")
    plt.setp(stemline, linewidth=1, color="blue")
    plt.setp(markerline, markersize=2, linewidth=1, color="blue")
    plt.setp(baseline, linewidth=2, color="blue")

    ylim = ax.get_ylim()
    ax.set_ylim((0, ylim[1]))

    if central_tendency:
        ax.axhline(np.median(np.abs(np.concatenate([fou_alt.real, fou_alt.imag]))), color="green", alpha=0.5, markersize=1, label="median")
        ax.axhline(np.mean(np.abs(np.concatenate([fou_alt.real, fou_alt.imag]))), color="yellow", alpha=0.5, markersize=1, label="mean")

    if legend:
        ax.legend()

'''
def plot_spectrum(
        ax:plt.Axes.axes, fou_alt:np.complex64, win, stimulus_frequency,
        stim_frq_pow:np.complex64, legend=False, central_tendency=False):
    """Plots the spectrum of a fourier transformed signal around some frequency of interest
    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to plot on
    fou_alt : np.complex64
        The fourier transform of the signal
    win : np.array
        The window of frequencies
    stimulus_frequency : float
        The frequency of the stimulus
    stim_frq_pow : np.complex64
        The power of the stimulus frequency"""
    
    stim_frq_pow = np.squeeze(stim_frq_pow)
    sgm_c = np.sqrt(.5 * np.mean(np.concatenate([fou_alt.real, fou_alt.imag])**2))
    # Horizonal
    ax.axhline(sgm_c, color="grey", label=r"$\sigma$", zorder=-np.inf)

    # Vertical
    points = [np.abs(stim_frq_pow.imag), np.abs(stim_frq_pow.real), sgm_c]
    ax.plot(2 * [stimulus_frequency],
        [min(points), max(points)],
        color="grey", zorder=-np.inf)
    
    ax.scatter(
        [stimulus_frequency, stimulus_frequency],
        np.abs(np.array([stim_frq_pow.real, stim_frq_pow.imag])),
        c=["orange", "red"], marker="o", s=16, edgecolor="grey",zorder=np.inf)
    
    # Plot spectrum    
    ax.plot(win, np.abs(fou_alt.real), ".", color="orange", alpha=0.5, markersize=1, label="real")
    ax.plot(win, np.abs(fou_alt.imag), ".", color="red",    alpha=0.5, markersize=1, label="imaginary")


    ylim = ax.get_ylim()
    ax.set_ylim((0, ylim[1]))

    if central_tendency:
        ax.axhline(np.median(np.abs(np.concatenate([fou_alt.real, fou_alt.imag]))), color="green", alpha=0.5, markersize=1, label="median")
        ax.axhline(np.mean(np.abs(np.concatenate([fou_alt.real, fou_alt.imag]))), color="yellow", alpha=0.5, markersize=1, label="mean")
    
    if legend:
        ax.legend()
'''

 

def normalize_timeseries(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def raw_GECI(raw_GECI_ax, F, cell_ind):
    """Plots raw timeseries of GCaMP data
    Parameters
    ----------
    raw_GECI_ax : (matplotlib.axes) Axis to plot on
    F : (np.ndarray)     Fluorescence data
    cell_inds : (list) Indices of cells to plot
    """
    raw_GECI_ax.plot(normalize_timeseries(F[cell_ind,:]), "k", linewidth=0.5)
    
    raw_GECI_ax.plot([0, 200], [1, 1], "k")
    raw_GECI_ax.annotate("200 s",(0, .80))
    print(raw_GECI_ax.get_ylim())
    # Set labels
    raw_GECI_ax.set_xlabel("time (s)")
    raw_GECI_ax.set_ylabel(r"$\frac{\Delta F}{F}$ ")
    raw_GECI_ax.axis("off")


def raw_NPIX(raw_NPIX_ax, ldr, spks, unitrow, window, freq, label=0.100, DX=1000, DY=.1):
    """GENERATE RAW DATA VISUALIZATION WITH PERIODS AND PHASORS
    Parameters
    ----------
    raw_NPIX_ax : matplotlib.axes
        Axis to plot on
    ldr : openEphysio.Loader
        loader object for recording
    spks : np.ndarray
        Spike times
    unitrow : pd.Series
        Row of the unit
    window : tuple, optional
        Window to plot, by default """
    # Window
    t_on, t_off = window

    # Plot Timeseries
    
    spike_sr = ldr.samplingrate(ldr.spikestream())
    # trace = spike_recording.get_traces(start_frame=int(t_on * spike_sr), end_frame=int(t_off * spike_sr), channel_ids=[f"AP{unitrow.ch + 1}"])
    trace = ldr.data(ldr.spikestream())[int(t_on * spike_sr):int(t_off * spike_sr), unitrow.ch]
    raw_NPIX_ax.plot(normalize_timeseries(trace), "k")

    # Set linewidths
    [line.set(linewidth=0.5, color="k") for line in raw_NPIX_ax.get_lines()]

    # Spikes in window
    subspks=spks[(spks > t_on) & (spks < t_off)] 

    # Get phases
    phases = ((subspks-subspks.values[0]) % (1/freq)) / (1/freq) * (2*np.pi)

    subspks = subspks - t_on
    subspks = subspks * spike_sr

    # Plot spike rasters
    raw_NPIX_ax.eventplot(subspks, linelengths=.1, color="blue", linewidths=2, lineoffsets= 1.1)
    raw_NPIX_ax.set_xticklabels(np.round(raw_NPIX_ax.get_xticks(),2))
    raw_NPIX_ax.set_xlabel("")

    # make them into phasors
    for spk, phase in zip(subspks, phases):
        dx, dy = DX * np.cos(phase), DY * np.sin(phase), 

        raw_NPIX_ax.annotate("", xy=(spk+dx, -.1+dy), xycoords='data', xytext=(spk-dx, -.1-dy),
                             textcoords='data', arrowprops=dict(facecolor='black',  arrowstyle="->"))
        
    raw_NPIX_ax.axis("off")

    # NPIX Scale bar
    raw_NPIX_ax.annotate(f"{int(label * 1000)} ms",(t_on, -.3+0.05))
    raw_NPIX_ax.hlines(-.3, t_on, t_on + spike_sr * label, "k")



def raw_GECI(raw_GECI_ax, F, cell_ind):
    """Plots raw timeseries of GCaMP data
    Parameters
    ----------
    raw_GECI_ax : (matplotlib.axes) Axis to plot on
    F : (np.ndarray)     Fluorescence data
    cell_inds : (list) Indices of cells to plot
    """
    raw_GECI_ax.plot(normalize_timeseries(F[cell_ind,:]), "k", linewidth=0.5)
    
    raw_GECI_ax.plot([0, 200], [1, 1], "k")
    raw_GECI_ax.annotate("200 s",(0, .80))
    print(raw_GECI_ax.get_ylim())
    # Set labels
    raw_GECI_ax.set_xlabel("time (s)")
    raw_GECI_ax.set_ylabel(r"$\frac{\Delta F}{F}$ ")
    raw_GECI_ax.axis("off")


def normalize_image_values(x, in_min=None, in_max=None, out_min=0, out_max=256):
    if in_min == None:
        in_min = np.min(x)
    if in_max ==None:
        in_max = np.max(x)
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def var_projection_GCaMP(var_projection_ax, tiff1, tiff2, stat, cell_inds):
    """
    Perform variance projection of GCaMP images and display the result.

    Parameters:
    - var_projection_ax: The axis object to display the variance projection image.
    - tiff1: The first GCaMP image.
    - tiff2: The second GCaMP image.
    - stat: A list of dictionaries containing statistical information about cells.
    - cell_inds: The index of the cell to highlight (default: 3).

    Returns:
    None
    """

    projection_l = []
    for tiff_ind, tiff in enumerate([tiff1, tiff2]):
        # Select cell ind 
        cell_ind = cell_inds[tiff_ind]
        
        # Project
        std_proj = np.std(tiff, axis=0)

        # Normalize
        normalized_proj = normalize_image_values(std_proj, out_max=1)
    
        # Stack into RGB channels
        stacked_proj = np.stack((normalized_proj,)*3, axis=-1)

        # Put masks
        for stat_ind, stat_cell in enumerate(stat):
            if stat_cell["npix"] > 35:
                color = np.array([1, 0, 0]) if stat_ind == cell_ind else np.array([0, 1, 0])
                # Fill in masks
                stacked_proj[stat_cell["ypix"], stat_cell["xpix"]] = color
            
        # Normalize
        projection_l.append(stacked_proj)
    
    # Make spacer 
    # Tiff dims were 20 x 700 x 700
    spacer_template = 0*np.ones((int(tiff1.shape[1] * 0.4), tiff1.shape[2]))
    spacer = np.stack((spacer_template,)*3, axis=-1)

    # Concatenate
    concatenated_proj = np.concatenate((projection_l[0],spacer, projection_l[1]), axis=0)

    # Negative image for contrast (lighter is better)
    stacked_img = 1 - concatenated_proj

    var_projection_ax.imshow(stacked_img)
    var_projection_ax.axis("off")
    return stacked_img

from scipy.stats import rice
# compute p-value from c-hat
def compute_p_value(c_hat, u=0):
    """ Compute p-value from c-hat using Rice distribution with parameter u."""
    return 1 - rice.cdf(c_hat, u)

def storey_qvalues(pvals, lambda_=0.5):
    """
    Compute q-values from p-values using Storey's method.
    
    Parameters:
        pvals: array-like of p-values
        lambda_: threshold for estimating pi0 (default 0.5)
        
    Returns:
        qvals: array of q-values
        pi0: estimated proportion of true nulls
    """
    pvals = np.asarray(pvals)
    N = len(pvals)
    
    # Estimate pi0 from the tail of the p-value distribution
    pi0 = np.mean(pvals > lambda_) / (1 - lambda_)
    pi0 = min(pi0, 1.0)  # ensure valid probability

    # Sort p-values and get their order
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    qvals = np.empty(N)

    # Compute initial q-values
    q_raw = pi0 * N * p_sorted / np.arange(1, N + 1)

    # Enforce monotonicity (non-decreasing)
    qvals[order[-1]] = q_raw[-1]
    for i in range(N - 2, -1, -1):
        qvals[order[i]] = min(q_raw[i], qvals[order[i + 1]])

    return qvals, pi0


from .MM_Plot_Utils import plot
def p_and_q_from_chat(c_hat,  u=0):
    """
    Compute and plot p-values and q-values in ascending order
    from c-hat using Storey's method.
    
    Parameters
    ----------
    c_hat : array-like
        Array of c-hat values.
    u : float, optional
        Parameter for the Rice distribution, by default 0.
    """
    
    pval = compute_p_value(c_hat, u)
    qval, pi0 = storey_qvalues(pval, lambda_=0.5)
    print(f"Estimated pi0: {pi0:.4f}")
    
    axes=plot(np.sort(pval), fmts=['b.'], xlabel='Neuron (ranked)', ylabel='p-value', markersize=0.5);
    plot([0, len(qval)], [0, 1], fmts=['k:'], linewidth=0.5, axes=axes);  # Add diagonal line
    
    axes=plot(np.sort(qval), fmts=['g.'], xlabel='Neuron (ranked)', ylabel='q-value', markersize=0.5);
    plot([0, len(qval)], [0, 1], fmts=['k:'], linewidth=0.5, axes=axes);  # Add diagonal line

    return np.array(pval), np.array(qval)



def Fig1_NPIX_data(modulation_df_full, CONTINGENCY, unitrow, freq):
    contingency_path = r"\\datanas\family\data_raw\20230413\first_site" + f"\\{CONTINGENCY}"

    if "visual" in CONTINGENCY:
        modulation_df = modulation_df_full.loc[modulation_df_full.rec==CONTINGENCY + "_90",:]
    else:
        modulation_df = modulation_df_full.loc[modulation_df_full.rec == CONTINGENCY,:]

    # Get number of spikes per unit
    for id, iddf in modulation_df.groupby("id"):
        modulation_df.loc[modulation_df["id"]==id,"nspk"] = len(iddf)

    # Filter by number of spikes per unit
    modulation_df_filt = modulation_df.loc[modulation_df.nspk > 10]

    allspks = {id:id_df.spk for id, id_df in modulation_df_filt.groupby("id")} 

    spks = allspks[unitrow.cluster_id]
    exemplar_fourier = fourier_analysis([spks], freq)

    return allspks, spks, exemplar_fourier, contingency_path  



def save_diagnostics_MM(MM_d, RECORDING_NAME, text=True):
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Plot first ten spiketrains
    [ax.eventplot(st, lineoffsets=st_i) for st_i, st in enumerate(list(MM_d["spikes"].values())[:10])]
    
    # Plot the periods and labels
    for recname, (periods, freq) in MM_d["aux"].items():
        if text:
            ax.text(periods[0] if len(periods.shape) == 1 else periods[0][0], 0, recname, rotation=90)
        for period in periods if len(periods.shape) > 1 else [periods]:
            ax.axvline(period[0], color="green", linestyle="--", alpha=0.5) # starts
            ax.axvline(period[1], color="red", linestyle="--", alpha=0.5) # stops
        
    
    ax.set_xlabel("Time (s)")
    ax.set_ylim((0, 20))
    save_and_close(fig, RECORDING_NAME, "MM_diagnostics", "")


