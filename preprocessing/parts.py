import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

from config import *
from data.load_raw import load_attentivu, load_blueberries

log = logging.getLogger(__name__)


def window_data(data: np.ndarray | pd.DataFrame, win_size: int, win_stride: int, ch, fs):
    size = win_size * fs
    stride = win_stride * fs

    if isinstance(data, pd.DataFrame):
        data = data[ch].values
    
    n_windows = (data.shape[0] - size) // stride + 2
    new_shape_math = (n_windows, size) + data.shape[1:]
    new_strides_math = (stride * data.strides[0],) + data.strides
    windowed_data = np.lib.stride_tricks.as_strided(data, shape=new_shape_math, strides=new_strides_math)
    return windowed_data


def resample(df: pd.DataFrame, origin: float | datetime, fs: float, method: str = 'index') -> pd.DataFrame:
    """
    Resamples the dataframe to a new frequency

    Args:
        df (pd.DataFrame): dataframe to resample
        origin (float | datetime): start time
        fs (float): sampling frequency
        method (str): interpolation method

    Returns:
        pd.DataFrame: resampled dataframe
    """
    if isinstance(origin, float):
        origin = datetime.fromtimestamp(origin, tz=timezone.utc)

    # ensure datetime index
    df.set_index('datetime', inplace=True)
    df = pd.concat([df, df.asfreq('1ms')]).sort_index()
    return (
        df[~df.index.duplicated()]
        .interpolate(method)
        .dropna()
        .resample(f'{1000.0/fs}ms', origin=origin)
        .mean()
        .reset_index()
    )


def get_part(
    df: pd.DataFrame,
    start_ts: float,
    end_ts: float,
    fs: float = None,
    pad_len: int = 5,
) -> pd.DataFrame:
    """
    Get a part of the dataframe between start and end timestamps
    """
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    if fs is not None:
        pad = timedelta(seconds=pad_len + 5)  # leave buffer to avoid edge effects
        df = df.loc[(df.datetime >= start_dt - pad) & (df.datetime < end_dt + pad)]
        df = resample(df, start_dt, fs, 'time')

    pad = timedelta(seconds=pad_len)
    df = df.loc[(df.datetime >= start_dt - pad) & (df.datetime < end_dt + pad)]

    # make timestamp relative to start time
    df.timestamp = df.timestamp - start_ts
    return df


def add_stim(df: pd.DataFrame, timing: Dict) -> pd.DataFrame:
    # df = pd.read_csv(proc_dir / 'nirs.csv', parse_dates=['datetime'])
    df['stim'] = 0
    for event, id in EVENTS.items():
        for offset in timing[event]:
            df.loc[(df['timestamp'] >= offset).idxmin(), 'stim'] = id
    return df


def segment_eeg(subject: str, pad_len: int, overwrite: bool = False):
    raw_dir = RAW_DIR / subject
    proc_dir = PROC_DIR / subject

    with raw_dir.joinpath('timing.json').open() as file:
        timing = json.load(file)

    ncsv = len(list(proc_dir.glob('**/eeg.csv')))
    if not overwrite and ncsv >= len(timing):
        log.info(f'{subject} data already prepared, {ncsv} files found, set force to rerun')
        return

    # load raw data and timing info
    eeg_raw = load_attentivu(raw_dir / 'attentivu/exg.csv')[['datetime', 'timestamp', *EEG_CH]]

    for part, info in timing.items():
        part_dir = proc_dir / part
        part_dir.mkdir(parents=True, exist_ok=True)

        with part_dir.joinpath('timing.json').open('w') as file:
            json.dump(info, file)

        eeg_df = get_part(eeg_raw, info['start'], info['end'], EEG_FS, pad_len)
        eeg_df.to_csv(part_dir / 'eeg.csv', index=False)

        log.info(f'Extracted and saved {part} data for {subject}')
        print(f'{part}\t{eeg_df.shape}')


def segment_imu(subject: str, pad_len: int, overwrite: bool = False):
    raw_dir = RAW_DIR / subject
    proc_dir = PROC_DIR / subject

    with raw_dir.joinpath('timing.json').open() as file:
        timing = json.load(file)

    ncsv = len(list(proc_dir.glob('**/imu.csv')))
    if not overwrite and ncsv >= len(timing):
        log.info(f'{subject} data already prepared, {ncsv} files found, set force to rerun')
        return

    # load raw data and timing info
    imu_raw = load_attentivu(raw_dir / 'attentivu/imu.csv')[['datetime', 'timestamp', *IMU_CH]]

    for part, info in timing.items():
        part_dir = proc_dir / part
        part_dir.mkdir(parents=True, exist_ok=True)

        with part_dir.joinpath('timing.json').open('w') as file:
            json.dump(info, file)

        imu_df = get_part(imu_raw, info['start'], info['end'], IMU_FS, pad_len)
        imu_df.to_csv(part_dir / 'imu.csv', index=False)

        log.info(f'Extracted and saved {part} data for {subject}')
        print(f'{part}\t{imu_df.shape}')


def segment_nirs(subject: str, pad_len: int, overwrite: bool = False):
    raw_dir = RAW_DIR / subject
    proc_dir = PROC_DIR / subject

    with raw_dir.joinpath('timing.json').open() as file:
        timing = json.load(file)

    ncsv = len(list(proc_dir.glob('**/nirs.csv')))
    if not overwrite and ncsv >= len(timing):
        log.info(f'{subject} data already prepared, {ncsv} files found, set force to rerun')
        return

    # load raw data and timing info
    nirs_raw = load_blueberries(raw_dir / 'blueberry')[['datetime', 'timestamp', *NIRS_CH]]

    for part, info in timing.items():
        part_dir = proc_dir / part
        part_dir.mkdir(parents=True, exist_ok=True)

        with part_dir.joinpath('timing.json').open('w') as file:
            json.dump(info, file)

        nirs_df = get_part(nirs_raw, info['start'], info['end'], NIRS_FS, pad_len)
        nirs_df.to_csv(part_dir / 'nirs.csv', index=False)

        log.info(f'Extracted and saved {part} data for {subject}')
        print(f'{part}\t{nirs_df.shape}')

