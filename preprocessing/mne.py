import json
import logging
from pathlib import Path
from typing import Dict, List

import mne
import pandas as pd
from mne.preprocessing import nirs
from mne_nirs.channels import get_long_channels
from mne_nirs.signal_enhancement import (
    enhance_negative_correlation,
    short_channel_regression,
)

from config import *
from preprocessing.mbll import mbll_mne

log = logging.getLogger(__name__)


def create_annotations(part_info: Dict, pad_len: int) -> mne.Annotations:
    offset, desc = [], []
    for task in ['math', 'rest']:
        offset += [off + pad_len for off in part_info[task]]
        desc += len(part_info[task]) * [task]
    return mne.Annotations(offset, TASK_LEN, desc, part_info['start'])

    # # create exp annotations
    # time, stim = part.split('_')
    # offset, desc = [], []
    # for task in ['math', 'rest']:
    #     offset += info[task]
    #     desc += len(info[task]) * [f'{time}/{stim}/{task}']
    # # offsets = part_info['math'] + part_info['rest']
    # # descrip = len(part_info['math']) * [f'{time}/{stim}/math'] + len(part_info['rest']) * ['rest']
    # return mne.Annotations(offset, float(TASK_LEN), desc, info['start'])


def create_epochs(
    raw: mne.io.Raw,
    tmin=0,
    tmax=TASK_LEN,
    baseline=(None, 0),
    detrend=None,
    reject=None,
    flat=None,
    path: Path = None,
):
    """
    Create epochs from raw data and save to file

    Args:
        raw (mne.io.Raw): _description_
        path (Path, optional): _description_. Defaults to None.
        tmin (int, optional): _description_. Defaults to 0.
        tmax (_type_, optional): _description_. Defaults to TASK_LEN.
        baseline (_type_, optional): _description_. Defaults to None.
        detrend (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline, detrend, reject=reject, flat=flat, preload=True)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        epochs.save(path, overwrite=True)

    return epochs


def create_eeg_raws(subject: str, pad_len: int, scale: float = 1, overwrite: bool = False):  # , stim: bool = False):
    raw_dir = RAW_DIR / subject

    with raw_dir.joinpath('timing.json').open() as file:
        timing = json.load(file)

    # raws = []
    for part, info in timing.items():
        part_dir = PROC_DIR / subject / part

        mne_dir = part_dir / 'mne'
        mne_dir.mkdir(exist_ok=True)
        file = mne_dir / 'eeg_raw.fif'

        if not overwrite and file.exists():
            log.info(f'{subject} {part} eeg_raw.fif already exists')
            continue

        eeg_df = pd.read_csv(part_dir / 'eeg.csv', parse_dates=['datetime'])
        annot = create_annotations(info, pad_len)

        eeg_ch = EEG_CH
        ch_names = EEG_POS
        ch_types = len(EEG_CH) * ['eeg']

        # if stim: #     eeg_ch += ['stim'] #     ch_names += ['stim'] #     ch_types += ['stim']

        eeg_info = mne.create_info(ch_names, EEG_FS, ch_types)
        eeg_raw = mne.io.RawArray(eeg_df[eeg_ch].to_numpy().T * scale, eeg_info)  # scaling here?
        eeg_raw.set_meas_date(annot.orig_time)
        eeg_raw.set_annotations(annot)
        eeg_raw.set_montage('standard_1020')
        # eeg_raw.set_eeg_reference('average')

        eeg_raw.save(file, overwrite=True)
        log.info(f'Saved eeg_raw.fif for {subject} {part}')


def create_imu_raws(subject: str, pad_len: int = 5, overwrite: bool = False):  # , stim: bool = False):
    raw_dir = RAW_DIR / subject
    with raw_dir.joinpath('timing.json').open() as file:
        timing = json.load(file)

    for part, info in timing.items():
        part_dir = PROC_DIR / subject / part

        mne_dir = part_dir / 'mne'
        mne_dir.mkdir(exist_ok=True)
        file = mne_dir / 'imu_raw.fif'

        if not overwrite and file.exists():
            log.info(f'{subject} {part} imu_raw.fif already exists')
            continue

        imu_df = pd.read_csv(part_dir / 'imu.csv', parse_dates=['datetime'])
        annot = create_annotations(info, pad_len)

        imu_ch = IMU_CH
        ch_types = len(IMU_CH) * ['misc']

        # if stim:
        #     imu_ch += ['stim']
        #     ch_types += ['stim']

        imu_info = mne.create_info(imu_ch, IMU_FS, ch_types)
        imu_raw = mne.io.RawArray(imu_df[imu_ch].to_numpy().T, imu_info)
        imu_raw.set_meas_date(annot.orig_time).set_annotations(annot)
        imu_raw.save(file, overwrite=True)
        log.info(f'Saved imu_raw.fif for {subject} {part}')


def create_nirs_raws(subject: str, pad_len: int, wls: List[str], overwrite: bool = False):
    raw_dir = RAW_DIR / subject
    with raw_dir.joinpath('timing.json').open() as file:
        timing = json.load(file)

    for part, info in timing.items():
        part_dir = PROC_DIR / subject / part

        mne_dir = part_dir / 'mne'
        mne_dir.mkdir(exist_ok=True)
        file = mne_dir / 'nirs_raw.fif'

        if not overwrite and file.exists():
            log.info(f'{subject} {part} nirs_raw.fif already exists')
            continue

        nirs_raw = mne.io.read_raw_snirf(part_dir / f'snirf/{"_".join(wls)}.snirf', preload=True)
        annot = create_annotations(info, pad_len)

        nirs_raw.set_meas_date(annot.orig_time)
        nirs_raw.set_annotations(annot)
        # nirs_raw = nirs_raw.resample(2 * NIRS_FS)
        nirs_raw.save(file, overwrite=True)
        log.info(f'Saved nirs_raw.fif for {subject} {part}')


def concat_raws(file: str, subjects: str | List[str], parts: str | List[str]) -> mne.io.Raw:
    if isinstance(subjects, str):
        subjects = [subjects]
    if isinstance(parts, str):
        parts = [parts]

    raws = [mne.io.read_raw_fif(PROC_DIR / subj / part / file, preload=True) for subj in subjects for part in parts]

    return mne.concatenate_raws(raws)


def concat_epochs(file: str, subjects: str | List[str], parts: str | List[str]) -> mne.Epochs:
    if isinstance(subjects, str):
        subjects = [subjects]
    if isinstance(parts, str):
        parts = [parts]

    epochs = [mne.read_epochs(PROC_DIR / subj / part / file) for subj in subjects for part in parts]

    return mne.concatenate_epochs(epochs)


def preprocess_nirs(
    subject: str,
    lfreq: float = 0.01,
    hfreq: float = 0.7,
    regress: bool = True,
    tddr: bool = True,
    enhance: bool = True,
    fs: int = None,
):
    for part in EXP_PARTS:
        part_dir = PROC_DIR / subject / part
        plots_dir = DATA_DIR / 'plots' / subject / part
        plots_dir.mkdir(parents=True, exist_ok=True)

        # * ----------------------------------- NIRS ---------------------------------- *#
        nirs_raw = mne.io.read_raw_fif(part_dir / 'mne/nirs_raw.fif', preload=True)
        od = nirs.optical_density(nirs_raw)

        # sci = nirs.scalp_coupling_index(raw_od, l_freq=0.7, h_freq=1.5)
        # od.info['bads'] = list(compress(raw_od.ch_names, sci < sci_thresh))

        if regress:
            od = short_channel_regression(od)
        if tddr:
            od = nirs.temporal_derivative_distribution_repair(od)

        hb = nirs.beer_lambert_law(od)

        if enhance:
            hb = enhance_negative_correlation(hb)

        hb.rename_channels(HB_CH_MAP)

        if regress:
            hb = get_long_channels(hb)

        hb.filter(lfreq, hfreq)

        if fs is not None:
            hb.resample(fs)

        nirs_epochs = create_epochs(hb, path=part_dir / 'mne/nirs_epo.fif')

        hbo_epochs = nirs_epochs.copy().pick_types(fnirs='hbo')
        hbo_epochs.save(part_dir / 'mne/hbo_epo.fif', overwrite=True)

        hbr_epochs = nirs_epochs.copy().pick_types(fnirs='hbr')
        hbr_epochs.save(part_dir / 'mne/hbr_epo.fif', overwrite=True)


def preprocess_eeg(
    subject: str, notch: float = 60, lfreq: float = 0.1, hfreq: float = 40, reference: str = 'average', fs: int = None
):
    for part in EXP_PARTS:
        proc_dir = PROC_DIR / subject / part

        eeg_raw = mne.io.read_raw_fif(proc_dir / 'mne/eeg_raw.fif', preload=True)

        if fs is not None:
            eeg_raw.resample(fs)

        if notch is not None:
            eeg_raw.notch_filter(notch)

        eeg_raw.filter(lfreq, hfreq)
        eeg_raw.set_eeg_reference(reference)
        # eeg_raw.plot(scalings=scalings, duration=560)

        events, event_id = mne.events_from_annotations(eeg_raw)
        eeg_epochs = create_epochs(
            eeg_raw,
            events,
            event_id,
            tmin=-0.5,
            tmax=4.5,
            baseline=None,
            detrend=1,
            reject=None,
            flat=None,
            path=proc_dir / 'mne/eeg_epo.fif',
        )
        eeg_epochs.save(proc_dir / 'mne/eeg_epo.fif', overwrite=True)
