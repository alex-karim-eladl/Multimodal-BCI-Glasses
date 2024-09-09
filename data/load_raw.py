"""
Project:		Multimodal-BCI-Glasses
File:			/data/load_raw.py
Author:			Alex Karim El Adl

Description: methods for loading raw data from AttentivU and Blueberry devices
"""

import logging
from zipfile import ZipFile

# from pprint import pprint
# import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import gdown
import pandas as pd

from config import *

log = logging.getLogger(__name__)

URL = 'https://drive.google.com/file/d/1NLdr5lMesZa4UrXiyrlj8zN49hWFF0Z3/view?usp=sharing'

def download_raw():
    log.info('downloading raw data...')
    # dst = DATA_DIR / 'raw'
    dst = './data/raw.zip'
    gdown.download(URL, dst, quiet=False, fuzzy=True)

    log.info(f'unzipping {dst}')
    with ZipFile(dst, 'r') as zip:
        zip.extractall(DATA_DIR)


def load_attentivu(file: Path) -> pd.DataFrame:
    """
    load dataframe containing EEG/EOG/IMU recordings from AttentivU .csv files

    Args:
        file (Path): path to .csv file

    Returns:
        pd.DataFrame: dataframe containing loaded data
    """
    if not RAW_DIR.exists():
        download_raw()

    dt_fmt = '%Y-%m-%d %H:%M:%S.%f'
    df = (
        pd.read_csv(file)
        .assign(datetime=lambda x: pd.to_datetime(x.UTC_TIME, format=dt_fmt, utc=True))
        .assign(timestamp=lambda x: x.datetime.apply(datetime.timestamp))
        .drop(columns=['SAMPLE_NUMBER', 'UTC_TIME'])
        .drop_duplicates('datetime')
        .sort_values('datetime')
        .dropna()
    )

    log.debug(f'loaded {file.name} {df.shape}')
    return df


def load_blueberry(file: Path) -> pd.DataFrame:
    # if path.is_file() and path.suffix == '.csv':
    # load dataframe from csv
    df = (
        pd.read_csv(file, low_memory=False)
        .query('timestamp != "timestamp"')
        .apply(pd.to_numeric)
        .assign(datetime=lambda x: pd.to_datetime(x.timestamp - 4 * 60 * 60, unit='s', utc=True))
        .drop(columns=['timestamp', 'ambient', 'ms_device'])
        .dropna()
        .drop_duplicates('datetime')
        .sort_values('datetime')
    )
    log.debug(f'loaded {file.name} {df.shape}')
    return df


def load_blueberries(sub_dir: Path, devices: List[str] = NIRS_DEVICES) -> pd.DataFrame:
    """
    load blueberry data from .csv file(s), return merged dataframe if multiple files or directory is given

    Args:
        path (Path | List[Path]): path to blueberry .csv file or
            directory containing blueberry files

    Returns:
        pd.DataFrame: dataframe containing data from .csv file(s)
    """
    if not RAW_DIR.exists():
        download_raw()
        
    # csv_paths = list(sub_dir.glob('*.csv'))
    # log.debug(f'found csv paths: {csv_paths}')

    paths = [sub_dir / f'{dev}.csv' for dev in devices]
    dfs = [load_blueberry(file).set_index('datetime').add_suffix(f'_{file.stem}') for file in paths]

    merged_df = (
        pd.concat(dfs, axis=1)  # , keys=[f.stem for f in csv_paths])
        .reset_index()
        .assign(timestamp=lambda x: x.datetime.apply(datetime.timestamp))
    )

    # cols = merged_df.columns
    # cols = cols[0] + cols[1:-1] + cols[:-1]
    # print(cols)
    # merged_df = merged_df[cols]

    log.debug(f'merged nirs {merged_df.shape}')
    return merged_df
