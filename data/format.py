"""
Author:			Alex Karim El Adl
Project:		Multimodal-BCI-Glasses
File:			/data/format.py

Description: data format conversion functions
"""

import json
import logging
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

# import xarray as xr

log = logging.getLogger(__name__)


def mat_to_csv(src: Path, dst: Path, key: str = None, cols: List[str] = None) -> None:
    """
    Convert .mat file to .csv file
    Args:
        src (Path): .mat file path
        dst (Path): .csv file path
        key (str, optional): specific key to extract from .mat file
        cols (List[str], optional): dataframe column labels
    """
    mat = loadmat(src)
    if key is not None:
        mat = mat[key]
    df = pd.DataFrame(mat, columns=cols) if cols is not None else pd.DataFrame(mat)
    df.to_csv(dst, index=False)
        


def pkl_to_json(src: Path, dst: Path = None) -> None:
    """
    Convert subject timing info from .pkl to .json file

    Args:
        src (Path): .pkl file path
        dst (Path, optional): output file path. Defaults to same as src with .json extension.
    """    
    with src.open('rb') as f:
        data = pickle.load(f)

    info = {}
    for timing in ['regular', 'random']:
        for stimulus in ['visual', 'auditory', 'mental']:
            if f'{stimulus}_{timing}_start' not in data:
                log.info(f'{stimulus}_{timing}_start not found')
                continue

            if timing == 'regular':
                part_len = 560.0
            else:
                part_len = 840.0

            if stimulus == 'visual':
                stim_len = 4.0
            elif stimulus == 'auditory':
                stim_len = 7.0
            else:
                stim_len = 28.0

            if stimulus == 'mental':
                part = f'{timing}_imagined'
            else:
                part = f'{timing}_{stimulus}'

            if timing == 'random':
                math = data[f'{stimulus}_{timing}_math_starts']
                rest = data[f'{stimulus}_{timing}_rest_starts']
            else:
                math = [0, 56, 112, 168, 224, 280, 336, 392, 448, 504]
                rest = [28, 84, 140, 196, 252, 308, 364, 420, 476, 532]

            info[part] = {
                'start': data[f'{stimulus}_{timing}_start'],  # CHECK UTC / LOCAL TIME !
                'end': data[f'{stimulus}_{timing}_end'],
                'math': math,
                'rest': rest,
                'stim_len': stim_len,
                'task_len': 28.0,
                'part_len': part_len,
            }

    # if dst is None:
    dst = src.with_suffix('.json')

    with dst.open('w') as f:
        json.dump(info, f, indent=4)


# * ________________________________ TO ARRAY ________________________________ *#
def mne_to_npy(mne_raw, save_file=None):
    npy = mne_raw.get_data().swapaxes(-1, -2)
    if save_file is not None:
        np.save(save_file, npy)
        log.info(f'Saved {save_file}')
    return npy


def mne_dict_to_npy(mne_dict, save_dir=None, suffix=''):
    npy_dict = {}
    for dtype, data in mne_dict.items():
        npy_dict[dtype] = mne_to_npy(data, save_dir / f'{dtype}{suffix}.npy')
    return npy_dict


# def epochs_to_xarray(epochs):
#     # self.dataset = xr.Dataset({'eeg': self.eeg_xr, 'nirs': self.nirs_xr, 'imu': self.imu_xr})
#     """
#     create xarray from mne epochs

#     Args:
#         epochs (Dict[str, mne.Epochs]): dict of mne.Epochs objects for each part
#     Returns:
#         xr.DataArray: _description_
#     """
#     # create xarray from mne epochs
#     


# * _________________________________ TO MNE _________________________________ *#
