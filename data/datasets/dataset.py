"""
Author:			Alex Karim El Adl
Project:		Multimodal-BCI-Glasses
File:			/data/dataset.py

Description: EEG Dataset class
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# plotly
import plotly.express as px
import plotly.graph_objects as go
from utils import get_memory_usage
from plotly.subplots import make_subplots
from scipy.signal import filtfilt, periodogram, welch
from tqdm import tqdm
from typeguard import typechecked

from utils import segment_mask

DELTA = (0.5, 4)  # (0.5, 4)
THETA = (4, 8)  # (4, 7) is common
ALPHA = (8, 14)  # (8, 12)
BETA = (14, 30)
GAMMA = (30, 45)  # (30, 100)

DEFAULT_BANDS = np.array([DELTA, THETA, ALPHA, BETA, GAMMA])
DEFAULT_BAND_IDX = {
    "delta": 0,
    "theta": 1,
    "alpha": 2,
    "beta": 3,
    "gamma": 4,
}

valid_filters_keys = [
    'paths',
    'subject',
    'session',
    'recording',
    'dataset',
    'contains_all_labels',
    'contains_any_labels',
    'all_labels_present',
    'any_labels_present',
    'NotNaN',
]


def valid_filters(filters):
    return all([key in valid_filters_keys for key in filters.keys()])


@typechecked
class EEGDataset:
    """
    The directory structure of a EEG Dataset is assumed to be either one of the following:

    variant 1 (single recording per subject):
    data_dir/
        subject1/
            <data files>
        subject2/
            <data files>

    variant 2 (multiple recordings per subject):
    data_dir/
        subject1/
            recording1/
                <data files>
            recording2/
                <data files>
        subject2/
            recording1/
                <data files>
            recording2/
                <data files>

    variant 3 (multiple recordings per subject, mutiple recordings per session):
    data_dir/
        subject1/
            session1/
                recording1/
                    <data files>
                recording2/
                    <data files>
            session2/
                recording1/
                    <data files>
        subject2/
            session1/
                recording1/
                    <data files>
            session2/
                recording1/
                    <data files>

    The <data files> might differ from dataset to dataset.
    But they all contain some signal (EEG) data and optionally some metadata.
    The metadata can be in the same file as the signal or in multiple files.
    """

    def __init__(
        self,
        data_dir: Path = None,
        bands: List[Tuple[float, float]] = DEFAULT_BANDS,
        band_idx: Dict[str, int] = DEFAULT_BAND_IDX,
        sampling_rate: int = 256,
        data_files: Dict[str, Dict[str, Any]] = dict(data=dict(pattern='exg*.csv')),
        filters: Optional[Dict[str, Any]] = None,
        channel_names: List[str] = ['CH1', 'CH2', 'CH3'],
    ):
        """
        Parameters
        ----------
        data_dir : Path
            Path to the dataset directory.
            If None, an empty dataset is created and all other arguments are ignored.
        bands : List[Tuple[float, float]] , optional
            List of EEG band definitions ([(lower_1, upper_1), (lower_2, upper_2), ...]), by default DEFAULT_BANDS
        band_idx : Dict[str, int], optional
            Dictionary mapping EEG bands names to indices in the band list, by default DEFAULT_BAND_IDX
        sampling_rate : int, optional
            Sampling rate of the EEG data, by default 256
        data_files : Dict[str, Dict[str, Any]]
            Dictionary mapping data file types to their file name patterns and additional parameters.
            Each data file type is a dictionary with the following keys:
            - pattern [str]: file name pattern, this must be provided
            - csv_read_kwargs [Dict[str, Any]]: keyword arguments for pandas read_csv function, optional
            - csv_channel_cols [List[str]]: list of column names that contain the channel data, must only be provided if the data is in a csv file
        """
        if data_dir is None:
            self._data_dir = None
            self._file_paths = []
            self.subjects = []
            self.sessions = []
            self.recordings = []
            self.data = []
            self.metadata = []
        else:
            assert data_dir.exists(), f'Data directory {data_dir} does not exist'
            assert 'data' in data_files, "data_files_pattern must contain a 'data' key"
            for key, value in data_files.items():
                assert 'pattern' in value, f"data_files_pattern[{key}] must contain a 'pattern' key"

            assert filters is None or valid_filters(filters), (
                'filters must only contain the following keys: ' + ', '.join(valid_filters_keys)
            )
            self._data_dir = data_dir
            self._bands = bands
            self._band_idx = band_idx
            self._data_files = data_files
            self._sampling_rate = sampling_rate
            self._channel_names = channel_names
            self._filters = filters

            self._file_paths: List[Dict[str, Path]] = self._get_file_paths()
            self.subjects: List[str] = self._getsubjects()
            self.sessions: List[str] = self._getsessions()
            self.recordings: List[str] = self._getrecordings()

            self.data: List[np.ndarray] = []
            self.metadata: List[Dict[str, Any]] = []

            self.data, self.metadata = self._read_data()
            self._update_metadata()
            self._preprocess_metadata()

        # make sure that all the lists have the same length
        assert (
            len(self.subjects) == len(self.sessions) == len(self.recordings) == len(self.data) == len(self.metadata)
        ), 'The number of subjects, sessions, recordings, data files and metadata files must be the same'

    @staticmethod
    def from_preprocessed_data(
        data: Optional[List[np.ndarray]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        old_dataset: Optional['EEGDataset'] = None,
        reduced_memory: bool = False,
    ):
        """
        Creates a EEGDataset object from preprocessed data and metadata.
        This is useful if you want to create a dataset from a subset of another dataset.

        Parameters
        ----------
        data : List[np.ndarray]
            List of preprocessed data, if None old_dataset.data is used
        metadata : List[Dict[str, Any]]
            List of metadata, if None old_dataset.metadata is used
        old_dataset : EEGDataset, optional
            The old dataset, by default None
            This is used to copy the bands and band_idx and other attributes from the old dataset.
        """
        # assert not all is None
        assert not (
            data is None and metadata is None and old_dataset is None
        ), 'At least one of data and metadata or old_dataset must be provided'
        if old_dataset is not None and data is None:
            data = old_dataset.data
            metadata = old_dataset.metadata

        dataset = EEGDataset()
        if reduced_memory:
            dataset.data = data
            dataset.metadata = metadata
        else:
            # make sure that all data is properly copied and not just referenced
            # this requires a lot of memory, but it also makes sure that the data is not modified by accident
            dataset.data = data[:]
            dataset.metadata = copy.deepcopy(metadata)

        if old_dataset is not None:
            dataset._bands = old_dataset._bands
            dataset._band_idx = old_dataset._band_idx
            dataset._sampling_rate = old_dataset._sampling_rate
            dataset._channel_names = old_dataset._channel_names
            if not reduced_memory:
                dataset._data_files = (
                    copy.deepcopy(old_dataset._data_files) if hasattr(old_dataset, '_data_files') else None
                )
                dataset._file_paths: List[Dict[str, Path]] = old_dataset._file_paths[:]
                dataset.subjects: List[str] = old_dataset.subjects[:]
                dataset.sessions: List[str] = old_dataset.sessions[:]
                dataset.recordings: List[str] = old_dataset.recordings[:]
            else:
                dataset._data_files = old_dataset._data_files if hasattr(old_dataset, '_data_files') else None
                dataset._file_paths: List[Dict[str, Path]] = old_dataset._file_paths
                dataset.subjects: List[str] = old_dataset.subjects
                dataset.sessions: List[str] = old_dataset.sessions
                dataset.recordings: List[str] = old_dataset.recordings

        return dataset

    def __getitems__(self, index: Union[List[int], np.ndarray]) -> EEGDataset:
        dataset = EEGDataset()
        dataset.data = [self.data[i] for i in index]
        dataset.metadata = [self.metadata[i] for i in index]
        dataset.subjects: List[str] = [str(self.subjects[i]) for i in index]
        dataset.sessions: List[str] = [str(self.sessions[i]) for i in index]
        dataset.recordings: List[str] = [str(self.recordings[i]) for i in index]

        dataset._bands = self._bands
        dataset._band_idx = self._band_idx
        dataset._sampling_rate = self._sampling_rate
        dataset._channel_names = self._channel_names
        if hasattr(self, '_data_files'):
            dataset._data_files = self._data_files

        return dataset

    def _get_file_paths(self) -> List[Dict[str, Path]]:
        """
        Finds all data files for each recording.
        """
        file_paths: List[Dict[str, Path]] = []

        assert 'data' in self._data_files.keys(), "data_files_pattern must contain a 'data' key"
        files_pattern = {f_type: f.get('pattern', None) for f_type, f in self._data_files.items()}

        main_data_files = self._data_dir.glob(f"**/{files_pattern['data']}")
        for main_data_file in main_data_files:
            main_data_dir = main_data_file.parent
            # find other files within the same directory (recursively if necessary)
            data_files = {}
            for f_type, file_pattern in files_pattern.items():
                files = list(main_data_dir.glob(f'**/{file_pattern}'))
                if len(files) == 0:
                    # if there are no files of this type, the data might be structured differently or it doesn't exist
                    # try to find it below the subject's root directory
                    subject_dir_name: str = main_data_file.relative_to(self._data_dir).parts[0]
                    subject_dir = self._data_dir / subject_dir_name
                    files = list(subject_dir.glob(f'**/{file_pattern}'))
                    if len(files) == 0:
                        warnings.warn(
                            f"No files found for {f_type} in matching {main_data_dir / '**' / file_pattern} or {subject_dir / '**' / file_pattern}"
                        )
                        continue
                assert len(files) == 1, f'Multiple files found for {f_type}'
                data_files[f_type] = files[0]

            file_paths.append(data_files)

        return file_paths

    def _get_directory_names(self, level: int) -> List[str]:
        """
        Returns a list of directory names on the specified level (relative to the data directory).
        """

        directory_names = []
        for file_path in self._file_paths:
            # get the first directory name relative to the data directory
            rel_path = file_path['data'].relative_to(self._data_dir)
            if (len(rel_path.parts) - 1) <= level:
                directory_name = 'None'
            else:
                directory_name = rel_path.parts[level]
            directory_names.append(directory_name)

        return directory_names

    def _getsubjects(self) -> List[str]:
        """
        Returns a list of subjects.
        The subject ids are extracted from the directory names
        and assumed to be on the first level of the directory structure.
        (relative to the data directory)
        """

        return self._get_directory_names(0)

    def _getsessions(self) -> List[str]:
        """
        Returns a list of sessions.
        The session ids are extracted from the directory names
        and assumed to be on the second level of the directory structure.

        If there is no second level directory, the session id is set to None.
        """

        return self._get_directory_names(1)

    def _getrecordings(self) -> List[str]:
        """
        Returns a list of recordings.
        The recording ids are extracted from the directory names
        and assumed to be on the third level of the directory structure.

        If there is no third level directory, the recording id is set to None.
        """

        return self._get_directory_names(2)

    def _csv_read_data(
        self,
        csv_file_path: Path,
        csv_read_kwargs: Dict[str, Any] = {},
        csv_channel_cols: Optional[List[str]] = None,
    ) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Reads a csv file, extracts the signal and additional metadata.
        The signal columns (channel data) are defined by self._csv_channel_cols.
        All other columns are considered metadata.
        """
        assert csv_file_path.exists(), f'Data file {csv_file_path} does not exist'
        csv_channel_cols = csv_channel_cols if csv_channel_cols is not None else []
        csv_read_kwargs = csv_read_kwargs if csv_read_kwargs is not None else {}

        data_df = pd.read_csv(csv_file_path, **csv_read_kwargs)

        # check that all channels are present, otherwise add a column with NaNs
        for channel in csv_channel_cols:
            if channel not in data_df.columns:
                warnings.warn(f'Could not find channel {channel} in {csv_file_path}, adding a column with NaNs')
                data_df[channel] = np.nan

        if len(csv_channel_cols) > 0:
            channel_data = data_df[csv_channel_cols].to_numpy()
        else:
            channel_data = None
            csv_channel_cols = []

        # interpret all remaining columns as metadata
        metadata = data_df.drop(columns=csv_channel_cols).to_dict(orient='list')

        # convert metadata to numpy arrays
        for key, value in metadata.items():
            metadata[key] = np.array(value)

        # channel data has to be float or int
        # if it is object type (strings) we have to convert it to float
        if channel_data is not None and channel_data.dtype == np.object:
            print(csv_file_path)
            channel_data = channel_data.astype(float)

        return channel_data, metadata

    def _pkl_read_data(
        self,
        pkl_file_path: Path,
        pkl_read_kwargs: Dict[str, Any] = {},
        pkl_channel_cols: Optional[List[str]] = None,
    ) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Reads a pkl file, extracts the signal and additional metadata.
        The signal columns (channel data) are defined by self._pkl_channel_cols.
        All other columns are considered metadata.
        """
        assert pkl_file_path.exists(), f'Data file {pkl_file_path} does not exist'
        pkl_channel_cols = pkl_channel_cols if pkl_channel_cols is not None else []
        pkl_read_kwargs = pkl_read_kwargs if pkl_read_kwargs is not None else {}

        data_df = pd.read_pickle(pkl_file_path, **pkl_read_kwargs)

        # check that all channels are present, otherwise add a column with NaNs
        for channel in pkl_channel_cols:
            if channel not in data_df.columns:
                warnings.warn(f'Could not find channel {channel} in {pkl_file_path}, adding a column with NaNs')
                data_df[channel] = np.nan

        if len(pkl_channel_cols) > 0:
            channel_data = data_df[pkl_channel_cols].to_numpy()
        else:
            channel_data = None
            pkl_channel_cols = []

        # interpret all remaining columns as metadata
        metadata = data_df.drop(columns=pkl_channel_cols).to_dict(orient='list')

        # convert metadata to numpy arrays
        for key, value in metadata.items():
            metadata[key] = np.array(value)

        return channel_data, metadata

    def _comply_with_path_filter(self, index: int, filters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Checks whether the data complies with the filter based on the file path (subject, session, recording).
        """
        if filters is None:
            return True

        assert valid_filters(filters), 'filters must only contain the following keys: ' + ', '.join(valid_filters_keys)

        path_filter_keys = ['subject', 'session', 'recording']
        path_filters = copy.deepcopy(filters['paths']) if 'paths' in filters else []
        assert isinstance(path_filters, list), "'paths' filters must be a list"
        assert all(
            [filter_key in path_filter_keys for path_filter in path_filters for filter_key in path_filter.keys()]
        ), "'paths' filters must only contain the following keys: " + ', '.join(path_filter_keys)

        # convert filters to lists if necessary
        filters = copy.deepcopy({k: v for k, v in filters.items() if k in path_filter_keys})
        for filter_name in filters.keys():
            filter_criteria = filters[filter_name]
            if not isinstance(filter_criteria, list):
                filters[filter_name] = [filter_criteria]

        # make sure all filters have the same length
        filter_lengths = [len(filters[k]) for k in filters.keys()]
        assert len(set(filter_lengths)) <= 1, 'All filters must have the same length'

        # convert dict of lists into a list of dictionaries
        filters = [dict(zip(filters.keys(), values)) for values in zip(*filters.values())]
        filters += path_filters

        if len(filters) == 0:
            # if all filters are empty, return True
            return True

        # these filters should return true if any of them is true
        for filter_dict in filters:
            # all of the indivudual checks should return true
            check_results = True
            for filter_name, filter_criteria in filter_dict.items():
                # convert all filter_criteria to strings
                filter_criteria = str(filter_criteria)
                if filter_name == 'subject':
                    check_results = check_results and (self.subjects[index] in filter_criteria)
                elif filter_name == 'session':
                    check_results = check_results and (self.sessions[index] in filter_criteria)
                elif filter_name == 'recording':
                    check_results = check_results and (self.recordings[index] in filter_criteria)

            if check_results:
                return True
        # if none of the filters returned true, return false
        return False

    def _comply_with_data_filter(
        self, data: np.ndarray, metadata: Dict[str, np.ndarray], filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Checks whether the data complies with the filter.
        """
        if filters is None:
            return True

        assert valid_filters(filters), 'filters must only contain the following keys: ' + ', '.join(valid_filters_keys)

        # check if the data complies with the filter
        check_results = True

        for filter_name, filter_criteria in filters.items():
            if not isinstance(filter_criteria, list):
                filter_criteria = [filter_criteria]

            # convert all filter_criteria to strings
            filter_criteria = [str(c) for c in filter_criteria]

            if filter_name == 'dataset':
                check_results = 'dataset' in metadata.keys() and np.all(
                    np.isin(np.unique(metadata['dataset']), filter_criteria)
                )
            elif filter_name == 'contains_all_labels':
                check_results = 'label' in metadata.keys() and np.all(
                    np.isin(np.unique(metadata['label']), filter_criteria)
                )
            elif filter_name == 'contains_any_labels':
                check_results = 'label' in metadata.keys() and np.any(
                    np.isin(np.unique(metadata['label']), filter_criteria)
                )
            elif filter_name == 'all_labels_present':
                check_results = 'label' in metadata.keys() and np.all(
                    np.isin(filter_criteria, np.unique(metadata['label']))
                )
            elif filter_name == 'any_labels_present':
                check_results = 'label' in metadata.keys() and np.any(
                    np.isin(filter_criteria, np.unique(metadata['label']))
                )
            elif filter_name == 'NotNaN':
                check_results = not np.isnan(data).any()

            # terminate early if any of the checks fails
            if not check_results:
                # print(f"Filter {filter_name} failed")
                break

        return bool(check_results)

    # 1) sigal contains all/any of certain labels
    # 2) all/any of certain labels are contained in the signal
    # variable name for  1) is contains_all_labels
    # variable name for  2) is labels_in_signal

    def _comply_with_filter(
        self, index: int, data: np.ndarray, metadata: Dict[str, np.ndarray], filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        if filters is None:
            return True

        check_results = [
            self._comply_with_path_filter(index=index, filters=filters),
            self._comply_with_data_filter(data=data, metadata=metadata, filters=filters),
        ]

        return bool(np.all(check_results))

    def _read_data(self) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Reads the 'data' data file for each recording
        and extracts the signal and metadata contained in that file.
        """

        data = []
        metadata = []
        skip_files = []

        # we only want to read data files in if we really have to
        # which means, if we can already filter files before reading them in this would massively speed up the process
        for i, _ in enumerate(self._file_paths):
            if not self._comply_with_path_filter(i, self._filters):
                skip_files.append(i)
                continue

        # remove skipped files
        for i in np.sort(skip_files)[::-1]:
            del self._file_paths[i]
            del self.subjects[i]
            del self.sessions[i]
            del self.recordings[i]

        # reset skipped files
        skip_files = []

        for i, file_path in tqdm(enumerate(self._file_paths), total=len(self._file_paths)):
            data_file = file_path['data']

            # get file extension
            ext = data_file.suffix
            if ext == '.csv':
                csv_read_kwargs = self._data_files['data'].get('csv_read_kwargs', None)
                csv_channel_cols = self._data_files['data'].get('csv_channel_cols', None)
                channel_data, non_channel_data = self._csv_read_data(
                    data_file, csv_read_kwargs=csv_read_kwargs, csv_channel_cols=csv_channel_cols
                )

                # because of memory constraints, we skip files that don't comply with the filter criteria
                if len(channel_data) == 0 or not self._comply_with_filter(
                    index=i, data=channel_data, metadata=non_channel_data, filters=self._filters
                ):
                    skip_files.append(i)
                    del channel_data
                    del non_channel_data
                    continue

                data.append(channel_data)
                metadata.append(non_channel_data)

            else:
                raise NotImplementedError(f'File extension {ext} is not supported')

        # remove skipped files
        print(f'Skipped {len(skip_files)} files out of {len(self._file_paths)}')
        for i in np.sort(skip_files)[::-1]:
            del self._file_paths[i]
            del self.subjects[i]
            del self.sessions[i]
            del self.recordings[i]

        assert len(data) == len(self._file_paths), 'The number of data files should match the number of file paths'
        assert len(data) > 0, 'No data files were found or all files were skipped'
        return data, metadata

    def split_rulebased(self, filters: Optional[Dict[str, Any]] = None) -> Tuple[EEGDataset, EEGDataset]:
        """
        To be applied after the data has been loaded

        Splits the data based on the filter rules, and returns two EEGDataset objects.
        The first object contains the data that complies with the filter rules, and the second object
        contains the data that does not comply with the filter rules.
        """

        # filter based on file paths first
        comply_files = []

        # filter based on the data itself
        for i, (data, metadata) in tqdm(enumerate(zip(self.data, self.metadata))):
            if self._comply_with_filter(index=i, data=data, metadata=metadata, filters=filters):
                comply_files.append(i)

        # get the indices of the files that do not comply with the filter rules
        non_comply_files = np.setdiff1d(np.arange(len(self.data)), np.array(comply_files)).tolist()

        assert len(comply_files) + len(non_comply_files) == len(
            self.data
        ), 'The number of files should match the number of file paths'

        # print some info
        print(
            f'Splitting the data into {len(comply_files)} files that comply with the filter rules and {len(non_comply_files)} files that do not comply with the filter rules'
        )
        if len(comply_files) == 0:
            warnings.warn('No files comply with the filter rules. Please check the filter rules and try again.')
        return self.__getitems__(comply_files), self.__getitems__(non_comply_files)

    def relabel(
        self,
        target_variable: Union[str, List[str]],
        val_filters: Optional[Dict[str, Any]] = None,
        test_filters: Optional[Dict[str, Any]] = None,
        overwrite_original: bool = True,
    ) -> EEGDataset:
        """
        Relabels the dataset based on the target variable and the filter rules.
        Target variable can be one of the following:
            - 'subject', 'session', 'recording'
            - 'recording_rel_pos_xx' (relative position of each sample in the recording), bucketed into XX bins
            - split (this will use the val_filters and test_filters to assign train, val or test as labels)
            - split_wo_test (this will use the val_filters to assign train or val as labels and filter out all test samples)

        If the target variable is 'split', then the val_filters and test_filters should be specified.
        The val_filters and test_filters should be dictionaries that specify the filter rules for the validation and test sets respectively.

        Parameters
        ----------
        target_variable : str
            The target variable to relabel the dataset with
        val_filters : Optional[Dict[str, Any]], optional
            The filter rules for the validation set, by default None
        test_filters : Optional[Dict[str, Any]], optional
            The filter rules for the test set, by default None
        overwrite_original : bool, optional
            Whether to overwrite the original labels or not, by default True
            If False then the original labels will stay in the metadata as "label" and the new labels will be stored in f"label_{target_variable}"
        """

        # compute some helpful variables across the dataset
        # get the length of each recording
        recordings_length = [len(metadata['label']) for metadata in self.metadata]
        recordings_length = np.array(recordings_length)
        # get all unique subjects
        unique_subjects = np.unique(self.subjects)
        # get all unique subject & session combinations
        unique_subject_session = np.unique(
            [subject + '_' + session for subject, session in zip(self.subjects, self.sessions)]
        )
        # get all unique subject & session & recording combinations
        unique_subject_session_recording = np.unique(
            [
                subject + '_' + session + '_' + recording
                for subject, session, recording in zip(self.subjects, self.sessions, self.recordings)
            ]
        )

        # we want to have the unique subject & session & recording combinations also for train data only
        # to make sure the  list only contains recordings from the training recordings
        # so we need to discard all recordings from val_filters and test_filters
        # the filter dicts contain a key called path which contains a list of
        # dicts with subject, session and recording keys that specify the recordings to be filtered out
        train_unique_subject_session = unique_subject_session.copy()
        train_unique_subject_session_recording = unique_subject_session_recording.copy()
        if val_filters is not None:
            for path in val_filters['paths']:
                train_unique_subject_session = np.setdiff1d(
                    train_unique_subject_session, [path['subject'] + '_' + path['session']]
                )
                train_unique_subject_session_recording = np.setdiff1d(
                    train_unique_subject_session_recording,
                    [path['subject'] + '_' + path['session'] + '_' + path['recording']],
                )
        if test_filters is not None:
            for path in test_filters['paths']:
                train_unique_subject_session = np.setdiff1d(
                    train_unique_subject_session, [path['subject'] + '_' + path['session']]
                )
                train_unique_subject_session_recording = np.setdiff1d(
                    train_unique_subject_session_recording,
                    [path['subject'] + '_' + path['session'] + '_' + path['recording']],
                )
        # sort them again
        train_unique_subject_session = np.sort(train_unique_subject_session)
        train_unique_subject_session_recording = np.sort(train_unique_subject_session_recording)
        # reverse the order to make it ascending
        train_unique_subject_session = train_unique_subject_session[::-1]
        train_unique_subject_session_recording = train_unique_subject_session_recording[::-1]

        rec_bin_random_multi_rand_idx = None
        if isinstance(target_variable, str):
            target_variable = [target_variable]
        if len(target_variable) > 1:
            assert not overwrite_original, 'If you want to relabel the dataset with multiple target variables, then you need to set overwrite_original to False'
        for target_var in target_variable:
            for i, metadata in enumerate(self.metadata):
                # first save the original labels
                original_labels = metadata['label']

                # get the target variable
                if target_var == 'subject':
                    target = self.subjects[i]
                elif target_var == 'session':
                    # combine the subject and session to get a unique identifier for each session
                    target = self.subjects[i] + '_' + self.sessions[i]
                elif target_var == 'recording':
                    # combine the subject, session and recording to get a unique identifier for each recording
                    target = self.subjects[i] + '_' + self.sessions[i] + '_' + self.recordings[i]
                elif target_var == 'recording_binary_sorted':
                    # similar to recording, but splitting the set of all recordings into two groups
                    # based on sorting the recording name (which is encoding the date of recording) in ascending order
                    # and then splitting the recordings into two groups
                    # the first group will be assigned label 0 and the second group will be assigned label 1
                    group1 = train_unique_subject_session_recording[len(train_unique_subject_session_recording) // 2 :]
                    group2 = train_unique_subject_session_recording[: len(train_unique_subject_session_recording) // 2]
                    recording_name = self.subjects[i] + '_' + self.sessions[i] + '_' + self.recordings[i]
                    if recording_name in group1:
                        target = 'recording_group_0'
                    elif recording_name in group2:
                        target = 'recording_group_1'
                    else:
                        # this means it's a recording that is in val or test
                        # so we assign it to group 1
                        target = 'recording_group_1'
                elif target_var == 'recording_binary_random':
                    # same as recording_binary_sorted, but instead of splitting by sorting the recording name
                    # we split the recordings randomly

                    rand_idx = np.arange(len(train_unique_subject_session_recording))
                    np.random.shuffle(rand_idx)
                    group1 = train_unique_subject_session_recording[rand_idx[: len(rand_idx) // 2]]
                    group2 = train_unique_subject_session_recording[rand_idx[len(rand_idx) // 2 :]]
                    recording_name = self.subjects[i] + '_' + self.sessions[i] + '_' + self.recordings[i]
                    if recording_name in group1:
                        target = 'recording_group_0'
                    elif recording_name in group2:
                        target = 'recording_group_1'
                    else:
                        # this means it's a recording that is in val or test
                        # so we assign it to group 1
                        target = 'recording_group_1'
                elif target_var.startswith('recording_binary_random_'):
                    # optionally we can also specificy the number of binary splits to be performed
                    # e.g. for recording_binary_random_4 we will still split in two groups (binary)
                    # but we will relabel four times so we will have 4 times 2 groups (binary)
                    # this would result in four classifiers that are trained
                    num_classifiers = int(target_var.split('_')[-1])
                    if rec_bin_random_multi_rand_idx is None:
                        rec_bin_random_multi_rand_idx = []
                        for j in range(num_classifiers):
                            rand_idx = np.arange(len(train_unique_subject_session_recording))
                            # shuffle the indices along the first dimension
                            np.random.shuffle(rand_idx)
                            rec_bin_random_multi_rand_idx.append(rand_idx)
                    target = []
                    for j in range(num_classifiers):
                        group1 = train_unique_subject_session_recording[
                            rec_bin_random_multi_rand_idx[j][: len(rec_bin_random_multi_rand_idx[j]) // 2]
                        ]
                        group2 = train_unique_subject_session_recording[
                            rec_bin_random_multi_rand_idx[j][len(rec_bin_random_multi_rand_idx[j]) // 2 :]
                        ]
                        recording_name = self.subjects[i] + '_' + self.sessions[i] + '_' + self.recordings[i]
                        if recording_name in group1:
                            target.append('recording_group_0')
                        elif recording_name in group2:
                            target.append('recording_group_1')
                        else:
                            # this means it's a recording that is in val or test
                            # so we assign it to group 1
                            target.append('recording_group_1')
                elif target_var == 'session_binary_sorted':
                    group1 = train_unique_subject_session[len(train_unique_subject_session) // 2 :]
                    group2 = train_unique_subject_session[: len(train_unique_subject_session) // 2]
                    session_name = self.subjects[i] + '_' + self.sessions[i]
                    if session_name in group1:
                        target = 'session_group_0'
                    elif session_name in group2:
                        target = 'session_group_1'
                    else:
                        # this means it's a recording that is in val or test
                        target = 'session_group_1'
                elif target_var == 'session_binary_random':
                    rand_idx = np.arange(len(train_unique_subject_session))
                    np.random.shuffle(rand_idx)
                    group1 = train_unique_subject_session[rand_idx[: len(rand_idx) // 2]]
                    group2 = train_unique_subject_session[rand_idx[len(rand_idx) // 2 :]]
                    session_name = self.subjects[i] + '_' + self.sessions[i]
                    if session_name in group1:
                        target = 'session_group_0'
                    elif session_name in group2:
                        target = 'session_group_1'
                    else:
                        # this means it's a recording that is in val or test
                        target = 'session_group_1'
                elif target_var.startswith('recording_rel_pos_'):
                    # the the maximum length a recording in the dataset can have
                    max_len = recordings_length.max()
                    # get the number of bins
                    num_bins = int(target_var.split('_')[-1])
                    # get the relative position of each sample in the recording
                    rel_pos = np.arange(len(original_labels)) / max_len
                    # bucketize the relative position into num_bins bins
                    target = np.digitize(rel_pos, np.linspace(0, 1, num_bins + 1))
                    # since np.digitize returns the index of the upper bound, we need to subtract 1
                    # so we get targets that are in the range [0, num_bins - 1]
                    target -= 1

                    if target[-1] != num_bins - 1:
                        warnings.warn(
                            f'The recording is significantly shorter than the maximum length of a recording in the dataset and'
                            f'therefore does not have enough samples to fill all the bins. Only contains bins {np.unique(target)}'
                        )

                    # convert the target to string
                    target = target.astype(str)
                elif target_var.startswith('contrastive_label_math_'):
                    # idea: we want to generate targets that are contrastiv to the original labels
                    # for example if a recording has labels [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    # we want to group all samples with label 0 together and all samples with label 1 together
                    # and afterwards within each group we want to assign labels randomly to subsequences of samples
                    # so that the labels are contrastive to the original labels

                    original_labels_math_binary = np.array(original_labels)
                    original_labels_math_binary[original_labels != 'math'] = 'non_math'
                    segments = segment_mask(original_labels_math_binary)
                    if len(target_var.split('_')) > 3:
                        size_sequences = int(target_var.split('_')[-1])
                    else:
                        size_sequences = 10
                    # create a new array of labels
                    target = np.array(['-1'] * len(original_labels))
                    # for each segment
                    for segment_start_idx, segment_end_idx in segments:
                        # get the label of the segment
                        segment_label = original_labels[segment_start_idx]
                        # get the length of the segment
                        segment_len = segment_end_idx - segment_start_idx

                        # calculate how many subsequences we can create from the segment
                        # subsequences should be of size sample_rate * size_sequences
                        subsequence_sample_size = self._sampling_rate * size_sequences
                        num_subsequences = math.ceil(segment_len / subsequence_sample_size)

                        # assign alternating labels to the subsequences
                        # so that the labels are contrastive to the original labels
                        subsequences_labels = np.array(['0'] * num_subsequences)
                        subsequences_labels[::2] = '1'

                        # assign the labels to the target array
                        for j, subsequence_label in enumerate(subsequences_labels):
                            start_idx = segment_start_idx + j * subsequence_sample_size
                            end_idx = segment_start_idx + (j + 1) * subsequence_sample_size
                            end_idx = min(end_idx, segment_end_idx + 1)
                            target[start_idx:end_idx] = subsequence_label

                    # assert no undefined labels are left
                    assert np.all(target != '-1')

                elif target_var == 'split':
                    if val_filters is None or test_filters is None:
                        raise ValueError(
                            "val_filters and test_filters should be specified if the target variable is 'split'"
                        )
                    if self._comply_with_path_filter(index=i, filters=val_filters):
                        target = 'val'
                    elif self._comply_with_path_filter(index=i, filters=test_filters):
                        target = 'test'
                    else:
                        target = 'train'
                elif target_var == 'split_wo_test':
                    raise NotImplementedError('This is not implemented yet')
                elif target_var == 'random_binary':
                    target = np.random.choice(['0', '1'], size=original_labels.shape)
                else:
                    raise ValueError(f'Target variable {target_var} is not supported')

                if isinstance(target, str):
                    new_labels = np.array([target] * len(original_labels))
                elif isinstance(target, list) and isinstance(target[0], str):
                    new_labels = [np.array([t] * len(original_labels)) for t in target]
                else:
                    new_labels = target

                if isinstance(new_labels, list):
                    for j, label in enumerate(new_labels):
                        assert (
                            label.shape == original_labels.shape
                        ), 'The shape of the new labels should match the shape of the original labels'
                else:
                    assert (
                        new_labels.shape == original_labels.shape
                    ), 'The shape of the new labels should match the shape of the original labels'

                if overwrite_original:
                    self.metadata[i]['label'] = new_labels
                    self.metadata[i]['original_label'] = original_labels
                else:
                    self.metadata[i]['label'] = original_labels
                    if isinstance(new_labels, np.ndarray):
                        self.metadata[i][f'label_{target_var}'] = new_labels
                    elif isinstance(new_labels, list):
                        for j, label in enumerate(new_labels):
                            self.metadata[i][f'label_{target_var}_{j}'] = label
                    else:
                        raise ValueError('new_labels should be either a numpy array or a list of numpy arrays')

        return self

    def get_unqiue_meta(self, key) -> np.ndarray:
        """
        Returns the unique values of a metadata key in the dataset
        """
        meta = []
        for metadata in self.metadata:
            meta.extend(metadata[key])
        return np.unique(np.array(meta))

    def get_discriminator_labels(self) -> List[str]:
        """
        Return the meta data keys that are used as discriminator labels
        """
        # discriminator keys in metadata start with label_
        discriminator_keys = [key for key in self.metadata[0].keys() if key.startswith('label_')]
        # remove the label_
        discriminator_keys = [key.split('label_')[-1] for key in discriminator_keys]
        return discriminator_keys

    @property
    def unique_labels(self) -> np.ndarray:
        """
        Returns the unique labels in the dataset
        """
        return self.get_unqiue_meta('label')

    def split_random(self, ratio: float) -> Tuple[EEGDataset, EEGDataset]:
        """
        To be applied after the data has been loaded

        Splits the dataset randomly into two datasets, with the ratio of the first dataset to the second dataset
        being determined by the ratio parameter.
        """

        # get the number of files in each dataset
        n_files = len(self.data)
        n_files_1 = int(n_files * ratio)

        # get the indices of the files in each dataset
        indices_1 = np.random.choice(np.arange(n_files), n_files_1, replace=False)
        indices_2 = np.setdiff1d(np.arange(n_files), indices_1).tolist()

        return self.__getitems__(indices_1), self.__getitems__(indices_2)

    def _update_metadata(self):
        """
        Iterates over all additional file types defined by data_files_pattern that may have been found for any given recording.
        For each file type, the data is read and the corresponding metadata is extracted and added to the metadata of the recording.
        """
        for i, file_path in enumerate(self._file_paths):
            for file_type, file in file_path.items():
                if file_type == 'data':
                    continue

                # get file extension
                ext = file.suffix
                if ext == '.csv':
                    csv_read_kwargs = self._data_files[file_type].get('csv_read_kwargs', None)
                    csv_channel_cols = self._data_files[file_type].get('csv_channel_cols', None)
                    _, metadata = self._csv_read_data(
                        file, csv_read_kwargs=csv_read_kwargs, csv_channel_cols=csv_channel_cols
                    )
                    # warning if newly added metadata keys already exist
                    for key in metadata.keys():
                        if key in self.metadata[i].keys():
                            warnings.warn(f'Metadata key {key} already exists, overwriting with new value')
                    self.metadata[i].update(metadata)
                elif ext == '.json':
                    metadata = json.load(file.open())
                    # warning if newly added metadata keys already exist
                    for key in metadata.keys():
                        if key in self.metadata[i].keys():
                            warnings.warn(f'Metadata key {key} already exists, overwriting with new value')
                    self.metadata[i].update(metadata)
                elif ext == '.pkl':
                    metadata = pickle.load(file.open('rb'))
                    # warning if newly added metadata keys already exist
                    for key in metadata.keys():
                        if key in self.metadata[i].keys():
                            warnings.warn(f'Metadata key {key} already exists, overwriting with new value')
                    self.metadata[i].update(metadata)

                # ADD PKL READ

                else:
                    raise NotImplementedError(f'File extension {ext} is not supported')

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns the data and metadata for the recording with the specified index.
        """
        return {
            'subject': self.subjects[index],
            'session': self.sessions[index],
            'recording': self.recordings[index],
            'data': self.data[index],
            'metadata': self.metadata[index],
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def num_channel(self):
        return len(self._channel_names)

    def _preprocess_metadata(self):
        """
        Preprocesses the metadata of a recording:
        - converts UTC_TIME to datetime objects
        - adds relative time (time since start of recording) versions of event and recording time
        - calculates recording start and end time as well as duration and adds it to the metadata
        - transforms the event image_names to event types (labels) and adds them to the metadata
        - estiamtes the sampling rate and adds it to the metadata

        Note: This is specific to the CALIB dataset and should be overwritten for other datasets.
        """
        for i, metadata in enumerate(self.metadata):
            # convert UTC_TIME to datetime objects
            if 'UTC_TIME' in metadata.keys():
                if metadata['UTC_TIME'].dtype.kind in ['U', 'S']:
                    metadata['UTC_TIME'] = pd.to_datetime(metadata['UTC_TIME']).to_numpy()
                elif metadata['UTC_TIME'].dtype == float:
                    metadata['UTC_TIME'] = pd.to_datetime(metadata['UTC_TIME'], unit='s').to_numpy()
                else:
                    raise NotImplementedError(f"Type {type(metadata['UTC_TIME'][0])} of UTC_TIME is not supported")

                # add recording start time to metadata
                metadata['recording_start_time'] = metadata['UTC_TIME'][0]
                # add recording end time to metadata
                metadata['recording_end_time'] = metadata['UTC_TIME'][-1]

                # add relative time (time since start of recording) in seconds to metadata
                metadata['relative_time'] = (metadata['UTC_TIME'] - metadata['recording_start_time']).astype(
                    float
                ) * 1e-9
                assert np.isclose(metadata['relative_time'], 0).any(), 'The relative time should start at 0'

                # add recording duration to metadata
                metadata['recording_duration'] = (
                    metadata['recording_end_time'] - metadata['recording_start_time']
                ).item() * 1e-9
                # estimate sampling rate
                metadata['sampling_rate'] = len(self.data[i]) / metadata['recording_duration']

                if 'events' in self._file_paths[i].keys():
                    # convert event_time to datetime objects
                    if metadata['event_time'].dtype.kind in ['U', 'S']:
                        metadata['event_time'] = pd.to_datetime(metadata['event_time']).to_numpy()
                    elif metadata['event_time'].dtype == float:
                        metadata['event_time'] = pd.to_datetime(metadata['event_time'], unit='s').to_numpy()
                    else:
                        raise NotImplementedError(
                            f"Type {type(metadata['event_time'][0])} of event_time is not supported"
                        )

                    # sometimes the timestamps from the events and data doesn't line up...
                    # to identify where this happens, calculate the difference between the first and last timestamp for both
                    # and also check whether
                    # a) the first event timestamp is after the first data timestamp
                    # b) the last event timestamp is before the last data timestamp

                    # # calculate difference between first and last timestamp for both
                    # diff_data = (metadata["UTC_TIME"][-1] - metadata["UTC_TIME"][0]).item() * 1e-9
                    # diff_events = (metadata["event_time"][-1] - metadata["event_time"][0]).item() * 1e-9

                    # # also calculate the duration of the data based on the assumption that the data is sampled at 256Hz
                    # diff_data_assumed = len(self.data[i]) / 256

                    # ratio_data_diff = diff_data_assumed / diff_data

                    # # check whether first event timestamp is after first data timestamp
                    # first_event_after_first_data = metadata["event_time"][0] > metadata["UTC_TIME"][0]
                    # # check whether last event timestamp is before last data timestamp
                    # last_event_before_last_data = metadata["event_time"][-1] < metadata["UTC_TIME"][-1]

                    # print(f"Debug information for recording {self.subjects[i]} {self.sessions[i]}:")
                    # print(f"First event after first data: {first_event_after_first_data}")
                    # print(f"Last event before last data: {last_event_before_last_data}")
                    # print(f"Ratio of data duration to assumed data duration based on timestamps: {ratio_data_diff}")
                    # print(f"Ratio of event duration to event duration based on timestamps: {diff_events / diff_data}")
                    # print(
                    #     f"Ratio of event duration to event duration based on timestamps (assumed data duration): {diff_events / diff_data_assumed}")
                    # print(f"Data duration:", diff_data)
                    # print(f"Data duration (assumed):", diff_data_assumed)
                    # print(f"Event duration:", diff_events)
                    # print()

                    # add event relative time (time since start of recording) in seconds to metadata
                    metadata['event_relative_time'] = (
                        metadata['event_time'] - metadata['recording_start_time']
                    ).astype(float) * 1e-9

                    # convert image_names to event_types
                    metadata['event_type'] = image_name_to_event_type(metadata['image_name'])
                    metadata['event_relative_end_time'] = np.append(
                        metadata['event_relative_time'][1:], metadata['recording_duration']
                    )
                    metadata['event_duration'] = np.diff(metadata['event_relative_end_time'])
                else:
                    # warning if no event_time is found
                    warnings.warn(f'No event found for recording {self.subjects[i]}')
            else:
                # add relative time (time since start of recording) in seconds to metadata
                metadata['relative_time'] = np.arange(len(self.data[i])) / self._sampling_rate

    def write_to_disk(self, path: Union[str, Path], overwrite: bool = False):
        """
        Writes the recordings to disk.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the directory where the recordings should be saved.
        overwrite : bool, optional
            If True, existing recordings will be overwritten, by default False
        """
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)

        for i, recording in tqdm(enumerate(self)):
            # create directory for recording
            recording_path = path / f"{recording['subject']}"

            if recording['session'] is not None:
                recording_path = recording_path / f"{recording['session']}"

            if recording['recording'] is not None:
                recording_path = recording_path / f"{recording['recording']}"

            if recording_path.exists():
                if overwrite:
                    shutil.rmtree(recording_path)
                else:
                    raise FileExistsError(f'Recording {recording_path} already exists')
            recording_path.mkdir(parents=True)

            # write data
            data_path = recording_path / 'data.csv'
            data_df = pd.DataFrame(recording['data'], columns=self._channel_names)

            # iterate through the meta data and add those to the data frame that have the same length as the data
            # add the rest to the metadata dictionary
            metadata = {}
            for key, value in recording['metadata'].items():
                if hasattr(value, '__len__') and len(value) == len(data_df):
                    data_df[key] = value
                else:
                    metadata[key] = value

            data_df.to_csv(data_path, index=False)
            # write metadata
            metadata_path = recording_path / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

    def save_to_disk_numpy(self, path: Union[str, Path]):
        """
        Writes the recordings to disk, but in a different format and using numpy.
        Instead of having individual files for each recording,
        the data, metadata, subject, session, and recording are saved in a individual numpy files.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the directory where the recordings should be saved.
        """

        start_time = datetime.now()
        path = Path(path)

        # overwrite by default and delete existing files if they exist
        if path.exists():
            shutil.rmtree(path)

        if not path.exists():
            path.mkdir(parents=True)

        attributes = ['data', 'metadata']
        class_attributes = [
            'subjects',
            'sessions',
            'recordings',
            '_bands',
            '_band_idx',
            '_sampling_rate',
            '_channel_names',
        ]

        # save attributes into individual numpy files (.npz)
        for attribute in attributes:
            if hasattr(self, attribute):
                attribute_data = getattr(self, attribute)
                attribute_path = path / f'{attribute}'

                # handle different data types (e.g. numpy array, List[ndarray], List[Dict[str, numpy array]])
                if isinstance(attribute_data, list):
                    if isinstance(attribute_data[0], np.ndarray):
                        np.savez(attribute_path, *attribute_data)
                    elif isinstance(attribute_data[0], dict):
                        # get all keys
                        keys = set()
                        for d in attribute_data:
                            keys.update(d.keys())
                        # convert to a datastructure of Dict[str, List[numpy array]]
                        data = {key: [] for key in keys}
                        for d in attribute_data:
                            for key in keys:
                                data[key].append(d.get(key, None))
                        # save each key as a separate numpy array into a subdirectory
                        subdir = path / f'{attribute}'
                        subdir.mkdir(parents=True)
                        for key in keys:
                            np.savez(subdir / f'{key}', *data[key])
                        # np.savez(attribute_path, **attribute_data)
                    else:
                        raise NotImplementedError(f'Type {type(attribute_data[0])} not supported')
                else:
                    raise NotImplementedError(f'Type {type(attribute_data)} not supported')
            else:
                warnings.warn(f"Attribute '{attribute}' not found when saving to disk")

        # create a dictionary with the class attributes and save it as a json file
        class_attributes_dict = {}
        for attribute in class_attributes:
            assert hasattr(self, attribute), f'Attribute {attribute} not found'
            class_attributes_dict[attribute] = getattr(self, attribute)

        class_attributes_path = path / 'class_attributes'
        np.savez(class_attributes_path, **class_attributes_dict)

        print(f'Writing to disk took {datetime.now() - start_time}')

    @staticmethod
    def load_from_disk_numpy(path: Union[str, Path], include: Optional[List[str]] = None) -> EEGDataset:
        """
        Loads the recordings from disk that were saved using the save_to_disk_numpy function.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the directory where the recordings are saved.
        include : Optional[List[str]], optional
            List of metadata attributes that should be loaded, by default None (all attributes are loaded)
        Returns
        -------
        EEGDataset
            The recordings that were saved on disk.
        """
        start_time = datetime.now()
        start_memory = get_memory_usage()

        path = Path(path)
        data_path = path / 'data.npz'
        class_attributes_path = path / 'class_attributes.npz'
        metadata_path = path / 'metadata'

        assert path.exists(), f'Path {path} does not exist'
        # assert this is a valid numpy style EEGDataset directory
        for p in [data_path, class_attributes_path]:
            assert (
                p.exists()
            ), f'Path {p} does not exist, make sure .save_to_disk_numpy() was used to generate this dataset directory'

        # create an empty EEGDataset
        eeg_dataset = EEGDataset()

        # load data
        eeg_dataset.data = list(np.load(data_path, allow_pickle=True).values())

        # load class attributes
        class_attributes = np.load(class_attributes_path, allow_pickle=True)
        for key, value in class_attributes.items():
            setattr(eeg_dataset, key, value)

        if hasattr(eeg_dataset, '_sampling_rate') and isinstance(eeg_dataset._sampling_rate, np.ndarray):
            eeg_dataset._sampling_rate = eeg_dataset._sampling_rate.item()

        # convert to List[str]
        for attribute in ['subjects', 'sessions', 'recordings']:
            if hasattr(eeg_dataset, attribute):
                setattr(eeg_dataset, attribute, np.array([str(s) for s in getattr(eeg_dataset, attribute)]))

        # load metadata
        metadata_files = metadata_path.glob('*.npz')
        metadata = {}
        for file in metadata_files:
            key = file.stem
            if include is None or key in include:
                metadata[key] = list(np.load(file, allow_pickle=True).values())

        # convert metadata which is Dict[str, List[numpy array]] to List[Dict[str, numpy array]]
        metadata_list = []
        for i in range(len(eeg_dataset.data)):
            metadata_dict = {}
            metadata_dict.update({key: value[i] for key, value in metadata.items()})
            metadata_list.append(metadata_dict)

        eeg_dataset.metadata = metadata_list

        print(f'Loaded {len(eeg_dataset)} recordings from disk in {datetime.now() - start_time}')
        print(f'Memory usage increased by {get_memory_usage() - start_memory} GB')
        assert len(eeg_dataset.data) == len(
            eeg_dataset.metadata
        ), f'Number of recordings ({len(eeg_dataset.data)}) and metadata ({len(eeg_dataset.metadata)}) do not match'

        return eeg_dataset


@typechecked
def image_name_to_event_type(image_names: np.ndarray) -> np.ndarray:
    """
    Converts the image name to the event type.

    Event types are
        1) eye movement events: blink, close, left, right
        2) engagement events: engagement, disengagement
        3) break event: break

    Decoding of event types from image names:
        1) eye movement events are encoded in the image name, e.g. contain the string "blink" or "left"
        2) for engagement evemts the images are named in the form of image0, image1, image2, ...
            - images 0-6; 8-10; 12-14 and image 19 are always a disengagement label
            - the remaining images have an engagement label

    Parameters
    ----------
    image_names : np.ndarray
        Array of image names.

    Returns
    -------
    np.ndarray
        Array of event types.
    """

    simple_events = [
        'blink',
        'close',
        'left',
        'right',
        'break',
        'apple',
        'baby',
        'book',
        'bottle',
        'chocolate',
        'cup',
        'dog',
        'duck',
        'pig',
        'potato',
        'table',
        'ten',
        'ball',
        'bye',
        'girl',
        'hi',
        'kitten',
        'paper',
        'people',
        'math',
        'rest',
    ]
    simple_events_part2 = ['imit1', 'imit2', 'imit3', 'name1', 'name2', 'name3', 'read1', 'read2', 'read3']

    event_types = ['unknown'] * len(image_names)
    for i, image_name in enumerate(image_names):
        for event_type in simple_events:
            if event_type in image_name:
                event_types[i] = event_type
                continue
        for event_type in simple_events_part2:
            if event_type in image_name:
                event_types[i] = f'{event_types[i]}_{event_type}'
                continue

        if image_name.startswith('image') and '.mp4' not in image_name:
            image_number = int(image_name.split('image')[1].split('.')[0])
            if image_number in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 19]:
                event_types[i] = 'disengagement'
            else:
                event_types[i] = 'engagement'

        if '.mp4' in image_name:
            image_number = int(image_name.replace('.mp4', '').replace('video', '').split(' ')[-1])

            if image_number > 4:
                event_types[i] = 'video engagement'
            else:
                event_types[i] = 'video disengagement'

        if event_types[i] == 'unknown':
            # warning if event type could not be decoded
            warnings.warn(f"Event type of '{image_name}' could not be decoded")
    return np.array(event_types)
