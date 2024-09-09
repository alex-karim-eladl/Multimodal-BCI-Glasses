"""
Author:			Alex Karim El Adl
Project:		Multimodal-BCI-Glasses
File:			/features/extract_mne.py

Description: feature extraction using mne-features
"""

from typing import List

import mne
import pandas as pd
from mne_features import extract_features

from config import *

NIRS_FEAT = [ 'mean', 'std', 'skewness', 'kurtosis', 'ptp_amp'] 

EEG_FEAT = [ 'pow_freq_bands' ]


EEG_PARAM = {
    'pow_freq_bands__freq_bands': EEG_BANDS,
    'pow_freq_bands__normalize': True,
    'pow_freq_bands__psd_method': 'welch',
}

# ! complete feature list
# [
#     'app_entropy',
#     'decorr_time',
#     'energy_freq_bands',
#     'higuchi_fd',
#     'hjorth_complexity',
#     'hjorth_complexity_spect',
#     'hjorth_mobility',
#     'hjorth_mobility_spect',
#     'hurst_exp',
#     'katz_fd',
#     'kurtosis',
#     'line_length',
#     'mean',
#     'pow_freq_bands',
#     'ptp_amp',
#     'quantile',
#     'rms',
#     'samp_entropy',
#     'skewness',
#     'spect_edge_freq',
#     'spect_entropy',
#     'spect_slope',
#     'std',
#     'svd_entropy',
#     'svd_fisher_info',
#     'teager_kaiser_energy',
#     'variance',
#     'wavelet_coef_energy',
#     'zero_crossings',
#     'max_cross_corr',
#     'nonlin_interdep',
#     'phase_lock_val',
#     'spect_corr',
#     'time_corr',
# ]

def extract_feat(epochs: mne.Epochs, feat: List[str], param: dict):
    feat_df = extract_features(epochs.get_data(), epochs.info['sfreq'], feat, param, return_as_df=True)
    feat_df['label'] = epochs.events[:, -1]
    return feat_df
    # feat_df.to_csv(FEA / 'eeg_clean_feat.csv', index=False)

def load_feat(filename: str, subjects: str | List[str], grouping: str | List[str]):
    if isinstance(subjects, str):
        subjects = [subjects]

    if grouping == 'all':
        parts = EXP_PARTS
    elif grouping == 'regular':
        parts = [f'regular_{stimulus}' for stimulus in STIMULI]
    elif grouping == 'random':
        parts = [f'random_{stimulus}' for stimulus in STIMULI]
    elif grouping == 'visual':
        parts = [f'{timing}_visual' for timing in TIMINGS]
    elif grouping == 'auditory':
        parts = [f'{timing}_auditory' for timing in TIMINGS]
    elif grouping == 'imagined':
        parts = [f'{timing}_imagined' for timing in TIMINGS]
    elif grouping in EXP_PARTS:
        parts = [grouping]
    else:
        raise ValueError(f'Invalid grouping: {grouping}')

    dfs = []
    for subject in subjects:
        for part in parts:
            path = FEAT_DIR / subject / part / filename

            df = pd.read_csv(path, header=[0, 1])
            dfs.append(df)

    if len(dfs) == 1:
        df = dfs[0]
    else:
        df = pd.concat(dfs, ignore_index=True)

    return df


def get_feat_xy(fname: str, subjects: str | List[str], grouping: str | List[str]):
    df = load_features(fname, subjects, grouping)

    y = df.pop('label').to_numpy().ravel()
    X = df.to_numpy()
    return X, y
