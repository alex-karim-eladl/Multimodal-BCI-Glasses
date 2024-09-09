import warnings

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, linregress, skew
from tqdm import tqdm

from config import *


def extract_fnirs(df_list, features, win_size, win_stride, devices=NIRS_DEVICES):
    """
    extracts fnirs features with sliding window
    """
    warnings.filterwarnings('ignore')
    feat_data = []
    for df in tqdm(df_list):
        indices = df.drop('-1', level=2, axis=0).index.drop_duplicates()
        feat_df = pd.DataFrame(index=indices).sort_index()

        for timing in TIMINGS:
            for stimulus in STIMULI:
                for rnd in range(ROUNDS):
                    for task in TASKS:
                        rows = (timing, stimulus, str(rnd), task)
                        period_df = df.loc[rows].copy()
                        period_df['timestamp'] -= period_df['timestamp'].to_numpy()[0]

                        for win, win_start in enumerate(range(0, 29 - win_size, win_stride)):
                            win_end = win_start + win_size
                            win_df = period_df[(period_df.timestamp >= win_start) & (period_df.timestamp < win_end)]

                            feat = []
                            cols = []
                            for dev in devices:
                                for val in HBS:
                                    if 'mean' in features:
                                        feat.append(win_df[f'{val}_{dev}'].mean())
                                        cols.append(f'mean_{val}_{win}_{dev}')
                                    if 'max' in features:
                                        feat.append(win_df[f'{val}_{dev}'].max())
                                        cols.append(f'max_{val}_{win}_{dev}')
                                    if 'min' in features:
                                        feat.append(win_df[f'{val}_{dev}'].min())
                                        cols.append(f'min_{val}_{win}_{dev}')
                                    if 'std' in features:
                                        feat.append(win_df[f'{val}_{dev}'].std())
                                        cols.append(f'std_{val}_{win}_{dev}')
                                    if 'slope' in features:
                                        feat.append(
                                            linregress(win_df['timestamp'].values, win_df[f'{val}_{dev}'].values)[0]
                                        )  # slope
                                        cols.append(f'slope_{val}_{win}_{dev}')
                                    if 'skew' in features:
                                        feat.append(skew(win_df[f'{val}_{dev}']))
                                        cols.append(f'skew_{val}_{win}_{dev}')
                                    if 'kurt' in features:
                                        feat.append(kurtosis(win_df[f'{val}_{dev}']))
                                        cols.append(f'kurt_{val}_{win}_{dev}')
                            feat_df.loc[rows, cols] = feat
        feat_data.append(feat_df)
    return feat_data

