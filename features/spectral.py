"""
Author:			Alex Karim El Adl
Project:		Multimodal-BCI-Glasses
File:			/features/spectral.py

Description: decompose EEG signal into frequency bands
"""

from typing import Tuple

import numpy as np
from scipy.signal import periodogram, welch

from config import EEG_BANDS, EEG_FS


def to_dB(res):
    return 10 * np.log(res)


def psd_estimator_fft(frame, fs=EEG_FS, dB=False):
    psd = np.absolute(np.fft.rfft(frame, axis=0) / fs) ** 2
    freq = np.fft.rfftfreq(len(frame), 1.0 / fs)
    if dB:
        psd = to_dB(psd)
    return freq, psd


def psd_estimator_welch(frame, fs=EEG_FS, dB=False):
    freq, psd = welch(frame.T, fs)
    psd = psd.T
    if dB:
        psd = to_dB(psd)
    return freq, psd


def psd_estimator_periodogram(frame, fs=EEG_FS, dB=False):
    freq, psd = periodogram(frame.T, fs)
    psd = psd.T
    if dB:
        psd = to_dB(psd)
    return freq, psd


def band_power(frame, bands=EEG_BANDS, fs=EEG_FS):
    psd = np.absolute(np.fft.rfft(frame, axis=0) / fs)
    freq = np.fft.rfftfreq(len(frame), 1.0 / fs)

    def avg_band_power(band_definition: Tuple[float, float]):
        band_lower, band_upper = band_definition
        return np.mean(psd[(freq >= band_lower) & (freq < band_upper)], axis=0) ** 2

    bp = [avg_band_power(freqs) for _, freqs in bands]
    return np.array(bp)


def band_power_from_psd(psd, freq, bands=EEG_BANDS):
    def avg_band_power(band_definition: Tuple[float, float]):
        band_lower, band_upper = band_definition
        return np.mean(psd[(freq >= band_lower) & (freq < band_upper)], axis=0)

    bp = [avg_band_power(freqs) for _, freqs in bands]
    return np.array(bp)



# def extract_psd(df_list, features, win_size, win_stride):
#     """
#     extracts EEG Power Spectral Density Features
#     """
#     feat_data = []
#     for df in tqdm(df_list):
#         indices = df.drop('-1', level=2, axis=0).index.drop_duplicates()
#         feat_df = pd.DataFrame(index=indices).sort_index()

#         for timing in TIMINGS:
#             for stimulus in STIMULI:
#                 for rnd in range(ROUNDS):
#                     for task in TASKS:
#                         period_df = df.loc[(timing, stimulus, str(rnd), task)]
#                         windowed = window_data(period_df, win_size, win_stride)
#                         psd = welch(windowed.swapaxes(1, 2), fs=250, nperseg=250)[
#                             1
#                         ]  # [:,:,:60] #up to 60Hz because that is how we filtered

#                         feat = []
#                         cols = []
#                         for win, win_data in enumerate(psd):
#                             for ch, ch_data in zip(NIRS_CH, win_data):
#                                 bands_data = [ch_data[1:4], ch_data[4:8], ch_data[8:14], ch_data[14:30], ch_data[30:50]]
#                                 for band, band_data in zip(EEG_BANDS, bands_data):
#                                     if 'mean' in features:
#                                         feat.append(np.mean(band_data))
#                                         cols.append(f'mean_{band}_{ch}_win{win}')
#                                     if 'max' in features:
#                                         feat.append(np.max(band_data))
#                                         cols.append(f'max_{band}_{ch}_win{win}')
#                                     if 'min' in features:
#                                         feat.append(np.min(band_data))
#                                         cols.append(f'min_{band}_{ch}_win{win}')
#                                     if 'std' in features:
#                                         feat.append(np.std(band_data))
#                                         cols.append(f'std_{band}_{ch}_win{win}')
#                                     # if 'slope' in features:
#                                     #     feat.append(linregress(win_df['timestamp'].values, win_df[f'{val}_{dev}'].values)[0]) # slope
#                                     #     cols.append(f'slope_{val}_{win}_{dev}')
#                                     if 'skew' in features:
#                                         feat.append(skew(band_data))
#                                         cols.append(f'skew_{band}_{ch}_win{win}')
#                                     if 'kurt' in features:
#                                         feat.append(kurtosis(band_data))
#                                         cols.append(f'kurt_{band}_{ch}_win{win}')

#                                 # cols.extend([f'{band}_{ch}_win{win}', f'theta_{ch}_win{win}', f'alpha_{ch}_win{win}', f'beta_{ch}_win{win}', f'gamma_{ch}_win{win}'])
#                         feat_df.loc[(timing, stimulus, str(rnd), task), cols] = feat
#         feat_data.append(feat_df)
#     return feat_data
