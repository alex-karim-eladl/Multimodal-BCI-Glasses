"""
Author:			Alex Karim El Adl
Project:		Multimodal-BCI-Glasses
File:			/data/moabb_datasets.py

Description: datasets for MoABB benchmarking https://github.com/NeuroTechX/moabb
"""


import logging
from pathlib import Path

import mne
from moabb.datasets.base import BaseDataset

from config import *

log = logging.getLogger(__name__)

# results = benchmark(
#     pipelines="./pipelines_braindecode",
#     evaluations=["CrossSession"],
#     paradigms=["LeftRightImagery"],
#     include_datasets=datasets,
#     results="./results/",
#     overwrite=False,
#     plot=False,
#     output="./benchmark/",
#     n_jobs=-1,
# )
# @hydra.main(config_path="../configs", config_name="config")


class EEGDataset(BaseDataset):
    def __init__(self):
        super().__init__(
            subjects=list(range(1, len(SUBJECTS) + 1)),
            sessions_per_subject=6,
            events=EVENTS,
            code='EEGDataset',
            interval=[0, TASK_LEN],
            paradigm='arithmetic',
        )

    def _get_single_subject_data(self, id):
        """Return data for a single subject."""
        # file_path_list = self.data_path(subject)
        sub_dir = Path(self.data_path(id)[0])
        sessions = {}
        for p, part_dir in enumerate(sub_dir.iterdir()):
            raw = mne.io.read_raw_fif(part_dir / 'eeg_raw.fif', preload=True)
            sessions[str(p)] = {'0': raw}

        return sessions

    def data_path(self, id=None, force_update=False, update_path=None, verbose=None):
        #     """Download the data from one subject."""
        if id not in self.subject_list:
            raise (ValueError('Invalid subject number'))

        # !TODO: download data
        # self.download()

        # TODO DL and prep data if not already done
        name = SUBJECTS[id - 1]
        path = PROC_DIR / name
        return [path.resolve()]

    #     url = "{:s}subject_0{:d}.mat".format(ExampleDataset_URL, subject)
    #     path = dl.data_dl(url, "ExampleDataset")
    #     return [path]  # it has to return a list

    def download(self):
        """Download the data."""
        raise NotImplementedError("Please download the data manually.")
        # url = "" 
        # filename = "sample_data"
        # filepath = data_dir + filename + ".zip"
        # # Download the data
        # gdown.download(url=url, output=filepath, quiet=False, fuzzy=True) print("Download complete!")
        # # unzip the data
        # with zipfile.ZipFile(filepath, 'r') as zip:
        # zip.extractall(data_dir) print("Unzip complete!")

        # for subject in self.subject_list:
        #     url = "{:s}subject_0{:d}.mat".format(ExampleDataset_URL, subject)
        #     dl.data_dl(url, "ExampleDataset")
        # return

class NIRSData(BaseDataset):
    def __init__(self):
        super().__init__(
            subjects=list(range(1, len(SUBJECTS) + 1)),
            sessions_per_subject=6,
            events=EVENTS,
            code='EEGDataset',
            interval=[0, TASK_LEN],
            paradigm='arithmetic',
        )

    def _get_single_subject_data(self, id):
        """Return data for a single subject."""
        # file_path_list = self.data_path(subject)
        sub_dir = Path(self.data_path(id)[0])
        sessions = {}
        for p, part_dir in enumerate(sub_dir.iterdir()):
            raw = mne.io.read_raw_fif(part_dir / 'nirs_raw.fif', preload=True)
            sessions[str(p)] = {'0': raw}

        return sessions
    
    def data_path(self, id=None, force_update=False, update_path=None, verbose=None):

        if id not in self.subject_list:
            raise (ValueError('Invalid subject number'))

        name = SUBJECTS[id - 1]
        path = PROC_DIR / name
        return [path.resolve()]
