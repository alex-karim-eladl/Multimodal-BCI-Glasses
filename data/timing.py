from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import mne
import pandas as pd
from string import Template
from typing import Dict, List

# from utils.io import load_attentivu, load_blueberry

log = logging.getLogger(__name__)


class DeltaTemplate(Template):
    delimiter = '%'


def strf_td(td: pd.Timedelta, format: str) -> str:
    """
    returns timedelta as formatted string

    Args:
        td (pd.Timedelta): timedelta object
        format (str): standard strf format

    Returns:
        str: _description_
    """
    d = {'D': td.days}
    d['H'], rem = divmod(td.seconds, 3600)
    d['M'], d['S'] = divmod(rem, 60)
    t = DeltaTemplate(format)
    return t.substitute(**d)


def ts_to_dt(timestamp: float, utc_diff: int = 0) -> datetime:
    """
    Converts timestamp to datetime object in UTC timezone

    Args:
        timestamp (float):
        utc_diff (int): hours offset from UTC

    Returns:
        datetime: _description_
    """
    utc_time = timestamp + utc_diff * 60 * 60
    return datetime.fromtimestamp(utc_time, tz=timezone.utc)


def recording_times(src_dir: Path, save_path: Path = None) -> pd.DataFrame:
    """
    Creates dataframe containing the start and end
    times of each csv file in the directory

    Args:
        src_dir (Path): directory containing csv files
        save_path (Path, optional): path to save dataframe

    Returns:
        pd.DataFrame: dataframe containing time info
    """
    data_times = pd.DataFrame(columns=['date', 'start', 'end', 'duration'])
    bby_files = ['f7', 'f8', 'fp']
    for file in src_dir.glob('**/*.csv'):
        rel_path = file.relative_to(src_dir)
        if file.parent == 'attentivu' or 'exg' in file.stem or 'imu' in file.stem:
            log.debug(f'processing {rel_path}')
            df = load_attentivu(file)
            data_times.loc[rel_path] = df_times(df)
        elif file.parent == 'blueberry' or file.stem in bby_files or file.stem.endswith('L') or file.stem.endswith('R') or file.stem.endswith('F'):
            log.debug(f'processing {rel_path}')
            df = load_blueberry(file)
            data_times.loc[rel_path] = df_times(df)
        else:
            log.info(f'skipping {rel_path}')

    if save_path is not None:
        data_times.to_csv(save_path)

    return data_times


def event_times(timing_file: Path) -> pd.DataFrame:
    """
    Returns dataframe containing formatted start and end times
    of each experiment part

    Args:
        timing_file (Path): path to timing.json file
    Returns:
        pd.DataFrame: _description_
    """
    with timing_file.open() as file:
        timing = json.load(file)

    part_times = pd.DataFrame(index=timing, columns=['start', 'end'])
    utc = timezone.utc
    for part, info in timing.items():
        # start = ts_to_dt(info['start']).time().strftime('%H:%M:%S.%f')[:-3]
        # end = ts_to_dt(info['end']).time().strftime('%H:%M:%S.%f')[:-3]
        fmt = '%H:%M:%S.%f'
        start = datetime.fromtimestamp(info['start'], tz=utc).time().strftime(fmt)[:-3]
        end = datetime.fromtimestamp(info['end'], tz=utc).time().strftime(fmt)[:-3]
        part_times.loc[part] = [start, end]
    return part_times.sort_values('start')


def df_times(df: pd.DataFrame) -> List[str]:
    min = df.datetime.min()
    max = df.datetime.max()

    return [
        min.date().strftime('%m/%d'),
        min.time().strftime('%H:%M:%S.%f')[:-3],
        max.time().strftime('%H:%M:%S.%f')[:-3],
        strf_td(max - min, '%H:%M'),
    ]




# #* _________________________________ SEGMENT ________________________________ *#


# def create_epochs(raw, params=None, get_data=False):
#     """
#     create mne.Epochs object from mne.io.Raw or parts Dict containing mne.io.Raw values
#     Args:
#         raw (mne.io.Raw | Dict[Tuple(str,str),mne.io.Raw]): _description_
#         params (Dict): parameters passed to mne.Epochs (mainly tmin, tmax, baseline, detrend, reject, flat)
#         get_data (bool, optional): returns ndarray if true, mne.Epochs if False
#     Returns:
#         mne.Epochs | Dict[Tuple[str,str],mne.Epochs]: _description_
#     """
#     if params is None: # default parameters
#         params = dict(tmin=0, tmax=TASK_LENGTH, baseline=None, detrend=None, reject=None, flat=None)

#     if isinstance(raw, dict):
#         return {
#             dtype: create_epochs(data, params, get_data) for dtype, data in raw.items()
#         }
#     else:
#         events, event_id = mne.events_from_annotations(raw)
#         epochs = mne.Epochs(raw, events, event_id, preload=True, **params)
#         if get_data:
#             epochs = {
#                 task: epochs[task].get_data(copy=True).swapaxes(1,2) for task in TASKS
#             }
#         return epochs


#     exp_start = get_start_time(info)
#     offsets, descrip = [], []
#     for part in parts:
#         part_offset = (get_start_time(info, part) - exp_start).total_seconds()
#         for math_offset, rest_offset in zip(*get_offsets(info, part)):
#             offsets.extend([math_offset+part_offset, rest_offset+part_offset])
#             descrip.extend(['/'.join(part)+'/math', '/'.join(part)+'/rest'])
#     return mne.Annotations(offsets, float(TASK_LENGTH), descrip, exp_start)