"""
Author:			Alex Karim El Adl
Project:		Multimodal-BCI-Glasses
File:			/preprocessing/filters.py

Description: implementation of various filters for denoising BCI data
"""

from typing import Dict
import numpy as np
import pandas as pd
import mne
import pywt
from mne.filter import filter_data
from scipy.signal import kaiserord, filtfilt, firwin, butter, iirnotch, savgol_filter

# import spkit as sp
from config import *


# def filter_raw_data(
#     raw: mne.io.Raw,
#     filter_design: Dict,
#     power_freq: float = None,
#     eog_ch=None,
#     plot=False,
#     savefig=False,
#     verbose=True,
# ):


#     if verbose == True:
#         print("---\nAPPLYING FILTER\n")
#     filt = raw.copy().load_data().filter(**filter_design, verbose=verbose)

#     if plot:
#         filter_params = mne.filter.create_filter(
#             raw.get_data(), raw.info["sfreq"], **filter_design
#         )

#         freq_ideal = [
#             0,
#             filter_design["l_freq"],
#             filter_design["l_freq"],
#             filter_design["h_freq"],
#             filter_design["h_freq"],
#             raw.info["sfreq"] / 2,
#         ]
#         gain_ideal = [0, 0, 1, 1, 0, 0]

#         fig, axs = plt.subplots(nrows=3, figsize=(8, 8), layout="tight", dpi=100)
#         mne.viz.misc.plot_filter(
#             filter_params,
#             raw.info["sfreq"],
#             freq=freq_ideal,
#             gain=gain_ideal,
#             fscale="log",
#             flim=(0.01, 80),
#             dlim=(0, 6),
#             axes=axs,
#             show=False,
#         )
#         if savefig == True:
#             plt.savefig(fname="Data/filter_design.png", dpi=300)
#         plt.show()

#     if line_remove != None:
#         if verbose == True:
#             print("---\nAPPLYING NOTCH FILTER\n")
#         filt = filt.notch_filter([line_remove])

#     if eog_channels != None or eog_channels != False:
#         if verbose == True:
#             print("---\nAPPLYING SSP FOR EOG-REMOVAL\n")
#         eog_projs, _ = mne.preprocessing.compute_proj_eog(
#             filt,
#             n_grad=0,
#             n_mag=0,
#             n_eeg=1,
#             reject=None,
#             no_proj=True,
#             ch_name=eog_channels,
#             verbose=verbose,
#         )
#         filt.add_proj(eog_projs, remove_existing=True)
#         filt.apply_proj()
#         filt.drop_channels(eog_channels)

#     return filt


# def artefact_rejection(filt, subjectname, epo_duration=5, verbose=True):
#     """
#     Convert Raw file to Epochs and conduct artefact rejection/augmentation on the signal.

#     Parameters
#     ----------
#     filt: Raw-type (MNE-Python) EEG file
#     subjectname: A string for subject's name
#     epo_duration (optional): An integer for the duration for epochs

#     Returns
#     -------
#     epochs: Epochs-type (MNE-Python) EEG file
#     """
#     if verbose == True:
#         print("---\nDIVIDING INTO EPOCHS\n")
#     epochs = mne.make_fixed_length_epochs(
#         filt, duration=epo_duration, preload=True, verbose=verbose
#     )

#     if verbose == True:
#         print("---\nEPOCHS BEFORE AR\n")
#     epochs.average().plot()
#     epochs.plot_image(title="GFP without AR ({})".format(subjectname))

#     if verbose == True:
#         print("---\nAPPLYING GLOBAL AR\n")
#     reject_criteria = get_rejection_threshold(epochs)
#     print("Dropping epochs with rejection threshold:", reject_criteria)
#     epochs.drop_bad(reject=reject_criteria, verbose=verbose)

#     if verbose == True:
#         print("---\nAPPLYING LOCAL AR\n")
#     ar = AutoReject(thresh_method="random_search", random_state=1)
#     ar.fit(epochs)
#     epochs_ar, reject_log = ar.transform(epochs, return_log=True)
#     reject_log.plot("horizontal")

#     if verbose == True:
#         print("---\nEPOCHS AFTER AR\n")
#     epochs_ar.average().plot()
#     fig, ax = plt.subplots(3, 1)
#     epochs_ar.plot_image(title="GFP with AR ({})".format(subjectname), fig=fig)
#     fig.savefig(f"temp/gfp_postar_{subjectname}.png")
#     plt.show()

#     return epochs_ar


def fir_filter(signal, Fs, cutoff, type="bandpass", width=0.2, ripple=65):
    numtaps, beta = kaiserord(ripple, width / (0.5 * Fs))
    if numtaps > len(signal) / 3.5:
        numtaps = int(len(signal) / 3.5)
    numtaps |= 1  # Set the lowest bit to 1, making the numtaps odd
    filter_coefs_peak = firwin(
        numtaps=numtaps, cutoff=cutoff, window=("kaiser", beta), pass_zero=type, fs=Fs
    )
    return filtfilt(filter_coefs_peak, 1, signal)


def wavelet_filter(signal, thresh=0.63, wavelet="db4"):
    threshold = thresh * np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per")
    coeff[1:] = (pywt.threshold(i, threshold, "soft") for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, "per")


def savitzky_golay(df, window_length=5, polyorder=2):
    """
    Applies a savitzky-golay filter to the data

    Parameters
    ----------
    df : pd.DataFrame
        data
    window_length : int
        length of the filter window
    polyorder : int
        order of the polynomial used to fit the samples

    Returns:
        filt_df: pd.DataFrame
    """
    filt_df = df.copy()
    for col in df[EEG_CH].columns:
        filt_df.loc[:, col] = savgol_filter(df.loc[:, col], window_length, polyorder)
    return filt_df


def butterworth(data, lowcut, highcut, order, fs):
    """
    Applies a butterworth bandpass filter to the data
    Parameters
    ----------
    data : array like
    lowcut : int
    highcut : int
        critical frequencies
    order : int
    fs : int
        sample frequency

    Returns:
        filt_df: pd.DataFrame
    """
    filt_df = data.copy()
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    for col in data[NIRS_DEVICES].columns:
        print(col)
        filt_df.loc[:, col] = filtfilt(b, a, data.loc[:, col])
    return filtfilt(b, a, data)  # filt_df


def notch(df, freq=60.0, fs=EEG_FS):
    """
    Applies a notch filter to the data to filter out line noise.

    Parameters
    ----------
    df : pd.DataFrame
        data
    freq : float
        Frequency to filter out
    fs : int
        sample frequency of the data

    Returns:
        filt_df: pd.DataFrame

    """
    ch = df.columns.drop(["datetime", "timestamp"])
    filt_df = df.copy()

    b, a = iirnotch(freq, 30.0, fs=fs)
    filt_df.loc[:, ch] = filtfilt(b, a, df[ch].values, axis=0)  # notch
    return filt_df


def kaiser(
    df,
    low=1.0,
    high=60.0,
    width_low=2.0,
    width_high=6.0,
    ripple_low=72.0,
    ripple_high=20.0,
    fs=EEG_FS,
):
    """
    Applies a band pass filter to the data
    Args:
        df (pd.DataFrame): EEG dataframe
        low (float, optional): Low cutoff frequency. Defaults to 1.0.
        high (float, optional): High cutoff frequency. Defaults to 50.
        width_low (float, optional): Width of transition from pass at low cutoff (rel. to Nyq).
        width_high (float, optional): Width of transition from pass at high cutoff (rel. to Nyq).
        ripple_low (float, optional): Desired attenuation in the stop band at low cutoff (dB).
        ripple_high (float, optional): Desired attenuation in the stop band at high cutoff (dB).
    Returns:
        Preprocessor: Preprocessor
    """
    nyq = fs / 2.0

    # Compute order and Kaiser parameter for the FIR filter.
    N_high, beta_high = kaiserord(ripple_high, width_high / nyq)
    N_low, beta_low = kaiserord(ripple_low, width_low / nyq)

    # Compute coefficients for FIR filters
    taps_high = firwin(N_high, high / nyq, window=("kaiser", beta_high))
    taps_low = firwin(N_low, low / nyq, window=("kaiser", beta_low), pass_zero=False)

    filt_df = df.copy()
    filt_df.loc[:, EEG_CH] = filtfilt(
        taps_low, 1.0, df[EEG_CH].values, axis=0
    )  # highpass
    filt_df.loc[:, EEG_CH] = filtfilt(
        taps_high, 1.0, filt_df[EEG_CH].values, axis=0
    )  # lowpass
    return filt_df


def bandpass_np(data: np.ndarray, low, high, fs=EEG_FS):
    """
    Input:
        data (np.ndarray) : Data in time x channel format
    Output:
        data (np.ndarray) : Filtered data in time x channel format (0.5 - 40 Hz)
    """
    return filter_data(data.T, fs, low, high, verbose=False).T


def notch_np(data, fs, freqs=[60.0]):
    """
    Input:
        data (np.ndarray) : Data in time x channel format
    Output:
        data (np.ndarray) : Filtered data in time x channel format (Notch filter - 50 Hz)
    """
    return mne.filter.notch_filter(data.T, fs, freqs, verbose=False).T
