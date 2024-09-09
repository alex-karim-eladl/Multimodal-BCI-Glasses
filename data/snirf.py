"""
Author:			Alex Karim El Adl
Project:		Multimodal-BCI-Glasses
File:			/dataset/snirf.py

Description: save data to Shared Near Infrared Spectroscopy Format (SNIRF) for package compatibility
                https://github.com/fNIRS/snirf
"""

from datetime import datetime, timezone
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from snirf import Snirf

from config import *
from data.load_raw import load_attentivu
from preprocessing.parts import get_part

log = logging.getLogger(__name__)


def save_snirf(
    subj_id: str,
    path: Path | str,
    nirs_df: pd.DataFrame,
    start_time: float | datetime,
    aux_df: pd.DataFrame = None,
    events: Dict[str, float] = None,
    wavelen: List[str] = NIRS_WAVELEN,
    devices: List[str] = NIRS_DEVICES,
):
    # infer devices from dataframe:
    # devices = nirs_df.columns.levels[0].unique().drop(['datetime', 'timestamp'])

    if isinstance(start_time, float):
        start_time = datetime.fromtimestamp(start_time, tz=timezone.utc)

    # infer start time from df
    # start_time = nirs_df.loc[nirs_df.timestamp >= 0].datetime.iloc[0]

    # create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    snirf = Snirf(str(path), 'w')
    snirf.formatVersion = '1.0'

    snirf.nirs.appendGroup()
    snirf.nirs[0].metaDataTags.SubjectID = subj_id
    snirf.nirs[0].metaDataTags.MeasurementDate = start_time.strftime('%Y-%m-%d')  # YYYY-MM-DD
    snirf.nirs[0].metaDataTags.MeasurementTime = start_time.strftime('%H:%M:%S.%f-05:00')  # hh:mm:ss.sTZD
    snirf.nirs[0].metaDataTags.LengthUnit = 'mm'
    snirf.nirs[0].metaDataTags.TimeUnit = 's'
    snirf.nirs[0].metaDataTags.FrequencyUnit = 'Hz'

    snirf.nirs[0].probe.sourcePos3D = np.array([NIRS_SOURCE_POS[dev] for dev in devices])
    snirf.nirs[0].probe.detectorPos3D = np.array([NIRS_DETECTOR_POS[dev][i] for dev in devices for i in range(2)])
    snirf.nirs[0].probe.wavelengths = np.array(list(map(int, wavelen)))

    snirf.nirs[0].data.appendGroup()
    snirf.nirs[0].data[0].time = nirs_df.timestamp.to_numpy()  # - datetime.timestamp( start_time )

    dataTimeSeries = []
    for d, dev in enumerate(devices):
        for o, opt in enumerate(['10', '27']):
            for w, wl in enumerate(wavelen):
                snirf.nirs[0].data[0].measurementList.appendGroup()
                snirf.nirs[0].data[0].measurementList[-1].sourceIndex = d + 1
                snirf.nirs[0].data[0].measurementList[-1].detectorIndex = 2 * d + o + 1
                snirf.nirs[0].data[0].measurementList[-1].wavelengthIndex = w + 1
                snirf.nirs[0].data[0].measurementList[-1].dataType = 1
                snirf.nirs[0].data[0].measurementList[-1].dataTypeIndex = 1  # ???

                dataTimeSeries.append(nirs_df.loc[:, f'{wl}nm{opt}mm_{dev}'].to_numpy())
    snirf.nirs[0].data[0].dataTimeSeries = np.array(dataTimeSeries).swapaxes(0, 1)

    if aux_df is not None:
        for col in IMU_CH:
            snirf.nirs[0].aux.appendGroup()
            snirf.nirs[0].aux[-1].name = IMU_CH_MAP[col]
            snirf.nirs[0].aux[-1].dataTimeSeries = aux_df.loc[:, col].to_numpy().reshape(-1, 1)
            snirf.nirs[0].aux[-1].time = aux_df.timestamp.to_numpy()
            # log.debug(f'imu col le: {snirf.nirs[0].aux[-1].dataTimeSeries.shape}')
        log.info(f'added aux {aux_df.shape}')

    if events is not None:
        np.array(
            [
                events['math'] + events['rest'],  # offsets: task block start times relative to part start (s)
                20 * [events['task_len']],  # durations: all 28s
                10 * [EVENTS['math']] + 10 * [EVENTS['rest']],  # values: 1 for math, 0 for rest
            ]
        ).T
        # for subj_id, stim_data in events.items():
        snirf.nirs[0].stim.appendGroup()
        snirf.nirs[0].stim[-1].name = subj_id
        snirf.nirs[0].stim[-1].dataTimeSeries = events
        log.info(f'added stim {events}')

    result = snirf.validate()
    if not result.is_valid():
        result.display(severity=2)

    snirf.save()
    snirf.close()

    # move log file to logs directory
    # shutil.move( dir / 'pysnirf2.log', DATA_DIR / 'logs/pysnirf2.log' )
    log.info(f'Saved {path}')


def create_snirfs(subject: str, wls: List[str], overwrite: bool = False):
    raw_dir = RAW_DIR / subject
    proc_dir = PROC_DIR / subject

    with raw_dir.joinpath('timing.json').open() as file:
        timing = json.load(file)

    nsnirf = len(list(proc_dir.glob('**/*.snirf')))
    if not overwrite and nsnirf >= len(timing):
        log.info(f'{subject} snirf data already prepared, {nsnirf} files found')
        return

    # imu_df = load_attentivu(raw_dir / 'attentivu/imu.csv')
    for part, info in timing.items():
        part_dir = proc_dir / part
        nirs_df = pd.read_csv(part_dir / 'nirs.csv', parse_dates=['datetime'])

        # # save 3 wl dat with imu as auxilary
        # snirf_path = part_dir / f'snirf/{"_".join(NIRS_WAVELEN)}.snirf'
        # imu_nirs = get_part(imu_df, info['start'], info['end'], NIRS_FS)
        # save_snirf(subject, snirf_path, nirs_df, info['start'], imu_nirs, info)

        # for wls in list(combinations(NIRS_WAVELEN, 2)):
        #     snirf_path = part_dir / f'snirf/{"_".join(wls)}.snirf'
        #     save_snirf(subject, snirf_path, nirs_df, info['start'], wavelen=wls) #events=info
        snirf_path = part_dir / f'snirf/{"_".join(wls)}.snirf'
        save_snirf(subject, snirf_path, nirs_df, info['start'], wavelen=wls)


# def create_stim(part_dir: Path, modality: str):
#     data = pd.read_csv(part_dir / 'imu.csv')[IMU_CH]
#     stim = np.zeros((1, n_epochs * n_samples))

# def eeg_mne(part_dir: Path):

#     with part_dir.joinpath('timing.json').open() as file:
#         timing = json.load(file)

#     annot = create_annotations(timing)

#     # if modality == 'eeg':
#     data = pd.read_csv(part_dir / 'eeg.csv')#[EEG_CH]
#     data['stim'] = 0


#     # samples = data.shape[0]

#     stim = np.zeros((1, samples))

#     info = mne.create_info(EEG_CH, EEG_FS, 'eeg')

#         mne_raw = mne.io.RawArray(data.to_numpy().T, info)
#         # mne_raw = mne.io.RawArray(data.to_numpy().T * 1e-6, info)
#     elif modality == 'imu':
#         info = mne.create_info(IMU_CH, IMU_FS, 'misc')
#         data = pd.read_csv(part_dir / 'imu.csv')[IMU_CH]
#         mne_raw = mne.io.RawArray(data.to_numpy().T, info)
#     else:
#         mne_raw = mne.io.read_raw_snirf(part_dir / f'{modality}.snirf', preload=True)

#     mne_raw.set_meas_date(annot.orig_time).set_annotations(annot)


#     # if isinstance(data, pd.DataFrame):
#     #     data = data[ch_names].to_numpy()

#     mne_raw = .set_meas_date(annotations.orig_time).set_annotations(annot)
#     return mne_raw


# def snirf_to_mne(path: Path, annotations: mne.Annotations) -> mne.io.Raw:
#     """
#     Load snirf file and convert to MNE Raw object with annotations

#     Args:
#         path (Path): path to snirf file
#         annotations (mne.Annotations): event annotations

#     Returns:
#         mne.io.Raw: mne.Raw object
#     """
#     mne_raw = (

#     )
#     return mne_raw


# def load_xarray(subject: str):
#     dir = DATA_DIR / 'raw' / subject

#     dims = ('part', 'trial', 'channel', 'time')
#     coords = {
#         'part': ['_'.join(part) for part in EXP_PARTS],
#         'trial': ROUNDS*['math'] + ROUNDS*['rest'],
#         'channel': epochs[EXP_PARTS[0]].ch_names,
#         'time': epochs[EXP_PARTS[0]].times
#     }

#     data = np.array([
#         np.concatenate([epochs[part][task].get_data() for task in TASKS]) for part in EXP_PARTS
#     ])

#     return xr.DataArray(data, coords, dims)
