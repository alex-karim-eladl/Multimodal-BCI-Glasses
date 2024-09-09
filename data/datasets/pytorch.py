from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from data.datasets.dataset import EEGDataset
from preprocessing.preprocessor import Preprocessor

from nptyping import NDArray, Shape
from tqdm import tqdm

import copy
import shutil
import warnings
from collections import defaultdict

import numpy as np
import torch
from utils import get_memory_usage
from scipy.signal import filtfilt, periodogram, welch
from sklearn.utils.class_weight import compute_class_weight
from typeguard import typechecked


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


DEFAULT_RATIOS = [("alpha", "delta"), ("alpha", "theta"), ("delta", "beta"), ("theta", "gamma"), ("delta", "alpha")]

TRANSFORMATIONS: Dict[str, Callable] = {}
DEFAULT_TRANSFORMS: List[str] = []
EPS = 1e-16

# add a register_transform decorator to register a transformation, with the option to add it to the default list


@typechecked
def register_transform(*args, **kwargs):
    def decorator(fn):
        # for some reason register_transform can not take multiple keyword arguments
        # if we don't pass them via **kwargs, so we have to do this
        default = kwargs.get("default", False)
        name = kwargs.get("name", None)
        if name is None:
            name = fn.__name__.lower()
        TRANSFORMATIONS[name] = fn
        if default:
            DEFAULT_TRANSFORMS.append(name)
        return fn

    return decorator


@typechecked
class EEGPytorchDataset(torch.utils.data.Dataset):
    def __init__(self,
                 window_size: float,
                 sampling_rate: int,
                 label_dict: Dict[str, int],
                 discriminators_label_dict: Optional[Dict[str, Dict[str, int]]] = None,
                 stride: Optional[float] = None,
                 sequence_partitioning_params: Optional[Dict[str, Any]] = {},
                 relabeling_strategy: Optional[str] = None,
                 preprocessing_params: Optional[Dict[str, Any]] = {},
                 return_meta: bool = False,
                 eeg_dataset: Optional[EEGDataset] = None,
                 label_smoothing: Optional[int] = None):

        self.window_size = int(window_size * sampling_rate)
        self.stride = int(round(stride * self.window_size)) if stride is not None else sampling_rate
        self.sampling_rate = sampling_rate
        self.label_dict = label_dict
        self.discriminators_label_dict = discriminators_label_dict
        self.eeg_dataset = eeg_dataset
        self.sequence_partitioning_params = sequence_partitioning_params
        self.preprocessing_params = preprocessing_params
        self.return_meta = return_meta
        self.relabeling_strategy = relabeling_strategy
        self.label_smoothing = label_smoothing

        self.set_data()
        # only if dataset is provided, otherweise keep data and metadata as None to be set later
        if eeg_dataset is not None:
            start_memory = get_memory_usage()
            self.prepare_data(eeg_dataset)
            end_memory = get_memory_usage()
            print(
                f"EEGPytorchDataset Memory usage increase: {end_memory - start_memory} GB")

    def set_data(self,
                 data: Optional[np.ndarray] = None,
                 labels: Optional[np.ndarray] = None,
                 recording_ids: Optional[np.ndarray] = None,
                 recording_order: Optional[np.ndarray] = None,
                 discriminator_labels: Optional[Dict[str, np.ndarray]] = None,
                 data_index: Optional[np.ndarray] = None,
                 label_index: Optional[np.ndarray] = None,
                 recording_names: Optional[np.ndarray] = None,):
        """
        Set the data and metadata for the dataset

        Parameters
        ----------
        data_index: Optional[np.ndarray]
            Array of shape (n_samples, sequence_length) containing the indices of the data for each sequence, referecing the data that belongs to a particular sample
        label_index: Optional[np.ndarray]
            Array of shape (n_samples,) containing the indices of the labels for each sequence, referecing the label that belongs to a particular sample
        data: np.ndarray
            Array of shape (n_sequences, window_size, n_channels) containing windowed EEG data
        labels: np.ndarray
            Array of shape (n_labeled_sequences,) containing the labels for each sample of each windowed EEG data
        recording_ids: np.ndarray
            Array of shape (n_sequences,) containing the recording id each window originated from
        recording_order: np.ndarray
            Array of shape (n_sequences,) containing the recording order for each sample

        """
        if data is not None:
            # let's check all the shapes are correct and that the data is consistent

            # TODO shape validation
            n_sequences = data.shape[0]
            n_samples = data_index.shape[0]
            assert label_index.shape == (
                n_samples,), f"label_index must be of shape (n_samples,)=({n_samples},), but is {label_index.shape}"
            assert labels.shape[
                0] <= n_sequences, f"labels must be of shape (n_labeled_sequences,) where n_labeled_sequences <= n_sequences, but {labels.shape[0]} > {n_sequences}"
            assert recording_ids.shape == (
                n_sequences,), f"recording_ids must be of shape (n_sequences,)=({n_sequences},), but is {recording_ids.shape}"
            assert recording_order.shape == (
                n_sequences,), f"recording_order must be of shape (n_sequences,)=({n_sequences},), but is {recording_order.shape}"
            assert len(np.unique(recording_names)) == len(recording_names), "recording_names must be unique"

            # assert all recording_ids map to a recording_name
            assert np.all(np.isin(recording_ids, np.arange(len(recording_names)))), "recording_ids must map to a recording_name"
        else:
            # either all data is None or none of it
            assert labels is None and recording_ids is None and recording_order is None, "Either all arguments must be None or none of it"

        self.data: np.ndarray = data
        self.labels: np.ndarray = labels
        self.recording_ids: np.ndarray = recording_ids
        self.recording_order: np.ndarray = recording_order
        self.data_index: np.ndarray = data_index
        self.label_index: np.ndarray = label_index
        self.recording_names: np.ndarray = recording_names
        self.discriminator_labels: Dict[str, np.ndarray] = discriminator_labels

    @staticmethod
    def split(dataset: EEGPytorchDataset, index: np.ndarray) -> EEGPytorchDataset:
        new_dataset = EEGPytorchDataset(
            window_size=dataset.window_size,
            stride=dataset.stride,
            sampling_rate=dataset.sampling_rate,
            label_dict=dataset.label_dict,
            return_meta=dataset.return_meta,
            discriminators_label_dict=dataset.discriminators_label_dict,
        )

        # get the indices
        data_index = dataset.data_index[index]
        label_index = dataset.label_index[index]

        unique_data_index = np.unique(data_index)
        unique_label_index = np.unique(label_index)

        # remap the indices via unique_data_index
        data_index = np.searchsorted(unique_data_index, data_index)
        label_index = np.searchsorted(unique_label_index, label_index)

        new_dataset.set_data(
            data=dataset.data[unique_data_index],
            labels=dataset.labels[unique_label_index],
            discriminator_labels={
                k: v[unique_label_index] for k, v in dataset.discriminator_labels.items()
            } if dataset.discriminators_label_dict is not None else None,
            recording_ids=dataset.recording_ids[unique_data_index],
            recording_order=dataset.recording_order[unique_data_index],
            data_index=data_index,
            label_index=label_index,
            recording_names=dataset.recording_names,
        )
        return new_dataset

    @staticmethod
    def splits(dataset: EEGPytorchDataset, indices: List[np.ndarray]) -> List[EEGPytorchDataset]:
        return [EEGPytorchDataset.split(dataset, idx) for idx in indices]

    @property
    def n_frames(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[-1]

    @property
    def n_classes(self) -> int:
        return max(self.label_dict.values()) + 1

    @property
    def n_classes_discriminator(self) -> Optional[Dict[str, int]]:
        return {
            k: max(v.values()) + 1
            for k, v in self.discriminators_label_dict.items()
        } if self.discriminators_label_dict is not None else None

    @property
    def shape(self) -> tuple:
        return (*self.data_index.shape, *self.data.shape[1:])

    def __len__(self) -> int:
        return self.data_index.shape[0]

    def __getitem__(self, index: Union[int, np.ndarray]) -> Tuple[np.ndarray,
                                                                  np.ndarray,
                                                                  Optional[np.ndarray],
                                                                  Optional[np.ndarray],
                                                                  Optional[np.ndarray],
                                                                  Optional[Dict[str, np.ndarray]]]:
        """
        Get a windowed EEG sample and its corresponding metadata

        Parameters
        ----------
        index: int or np.ndarray 
            Index of the sample to get, if array is provided, the shape of the array must be (n_items,) 
            and all returned arrays will have shape (n_items, ...) 
        Returns
        -------
        data: np.ndarray
            Array of shape (window_size, n_channels) containing windowed EEG data
        labels: np.ndarray
            Array of shape (window_size,) or containing the labels for each sample of the windowed EEG data
        recording_ids: np.ndarray
            Array of shape (1,) containing the recording id the window originated from
        recording_order: np.ndarray
            Array of shape (window_size,) containing the recording order for each sample
        """

        return (self.data[self.data_index[index]],
                np.array(self.labels[self.label_index[index]]),
                np.array(self.recording_ids[self.data_index[index]]) if self.return_meta else None,
                np.array(self.recording_order[self.data_index[index]]) if self.return_meta else None,
                np.array(self.recording_names[np.array(self.recording_ids[self.data_index[index]])]
                         ) if self.return_meta else None,
                {k: np.array(v[self.label_index[index]])
                    for k, v in self.discriminator_labels.items()} if self.discriminators_label_dict is not None else None,
                )

    def __getitems__(self, index: Union[np.ndarray, List[int]]) -> Tuple[np.ndarray,
                                                                         np.ndarray,
                                                                         Optional[np.ndarray],
                                                                         Optional[np.ndarray],
                                                                         Optional[np.ndarray],
                                                                         Optional[Dict[str, np.ndarray]]]:
        return self.__getitem__(np.array(index))

    def reduce_labels(self, labels_windows: np.ndarray, num_classes: int) -> np.ndarray:

        labels_windows_onehot = np.squeeze(np.eye(num_classes)[labels_windows])
        if self.relabeling_strategy is None or self.relabeling_strategy == "majority":
            # data is labeled for each sample, but we want to predict the label for the whole window
            # so we need to find the majority label for each window
            # to support multi-class classification we use the argmax of the one-hot encoded labels
            labels_windows = labels_windows_onehot.sum(-2).argmax(-1)
        elif self.relabeling_strategy == "last_second":
            # labels_windows = np.median(labels_windows[..., -self.sampling_rate:], axis=-1)
            labels_windows = labels_windows_onehot[..., -self.sampling_rate:, :].sum(1).argmax(-1)
        else:
            raise ValueError(f"Unknown relabeling strategy {self.relabeling_strategy}")
        return labels_windows

    def sliding_window(self,
                       data: List[np.ndarray],
                       labels: List[np.ndarray],
                       discriminator_labels: Optional[Dict[str, List[np.ndarray]]] = None) -> Tuple[np.ndarray,
                                                                                                    np.ndarray,
                                                                                                    Optional[np.ndarray],
                                                                                                    Optional[np.ndarray],
                                                                                                    Optional[Dict[str, np.ndarray]]]:
        """
        Apply sliding window transformations to both the data and metadata

        Parameters
        ----------
        data: List[np.ndarray]
            List of numpy arrays of shape (n_samples, n_channels)
        labels: List[np.ndarray]
            List of numpy arrays of shape (n_samples,) containing the labels for each sample of each recording

        """
        # make sure all lists have the same length
        assert len(data) == len(labels), "data and labels must have the same length"
        frames_data = []
        frames_labels = []
        frames_discriminator_labels = defaultdict(list)
        frames_recording_ids = []
        frames_recording_order = []
        for i in tqdm(range(len(data))):
            # Apply the sliding window view both to the data and the labels
            data_windows = np.lib.stride_tricks.sliding_window_view(data[i], self.window_size, axis=0)[::self.stride]
            # data_windows is now of shape (n_windows, n_channels, window_size) but we want (n_windows, window_size, n_channels)
            data_windows = np.moveaxis(data_windows, -2, -1)
            frames_data.append(data_windows)

            labels_windows = np.lib.stride_tricks.sliding_window_view(labels[i], self.window_size, axis=0)[::self.stride]
            labels_windows = self.reduce_labels(labels_windows, self.n_classes).astype(float)
            assert labels_windows.shape[0] == data_windows.shape[0], "Labels and data must have the same number of windows"

            #label smoothing
            if self.label_smoothing:
                #find transitions from 0 to 1 and 1 to 0
                transitions = np.diff(labels_windows)
                zero_one = np.where(transitions == 1)
                one_zero = np.where(transitions == -1)

                #linear label in-/decrease
                for transition in zero_one[0]:
                    end = min(len(labels_windows), int(transition.item() + self.label_smoothing))
                    labels_windows[transition.item():end] = np.linspace(0, 1, self.label_smoothing)

                for transition in one_zero[0]:
                    start = max(0, int(transition.item() - self.label_smoothing))+1
                    labels_windows[start:transition.item()+1] = np.linspace(1, 0, self.label_smoothing)

            frames_labels.append(labels_windows)

            if discriminator_labels is not None:
                for discriminator_name, discriminator_label in discriminator_labels.items():
                    discriminator_label_windows = np.lib.stride_tricks.sliding_window_view(discriminator_label[i],
                                                                                           self.window_size,
                                                                                           axis=0)[::self.stride]
                    discriminator_label_windows = self.reduce_labels(
                        discriminator_label_windows, self.n_classes_discriminator[discriminator_name])
                    assert discriminator_label_windows.shape[0] == data_windows.shape[0], "Labels and data must have the same number of windows"
                    frames_discriminator_labels[discriminator_name].append(discriminator_label_windows)

            # for more detailed model evaluation we also want to keep track of
            # a) windows of the same recording
            # b) the order of the windows in the recording

            # first the recording id
            frames_recording_ids.append(np.repeat(i, labels_windows.shape[0]))

            # then the order of the windows in the recording
            frames_recording_order.append(np.arange(labels_windows.shape[0]))

        return (np.concatenate(frames_data),
                np.concatenate(frames_labels),
                np.concatenate(frames_recording_ids) if len(frames_recording_ids) > 0 else None,
                np.concatenate(frames_recording_order) if len(frames_recording_order) > 0 else None,
                {
                    k: np.concatenate(v) for k, v in frames_discriminator_labels.items()
        } if len(frames_discriminator_labels) > 0 else None,
        )

    def _reduce_metadata(self, metadata: Dict[str, np.ndarray], key="label") -> np.ndarray:
        """
        Reduces the metadata to a single array by extracting the value for the given key, 
        by default reduce to labels

        Parameters
        ----------
        metadata: Dict[str, np.ndarray]
            Dictionary containing metadata for each sample in data
        key: str
            Key to extract from metadata, by default "label"

        Returns
        -------
        labels: np.ndarray
            Array of shape (n_samples,) containing the labels for each sample
        """

        labels = metadata[key] if key in metadata else None
        return labels

    def prepare_data(self, eeg_dataset: EEGDataset) -> None:

        data: List[np.ndarray] = eeg_dataset.data
        metadata: List[Dict[str, Any]] = eeg_dataset.metadata

        # for evaluation purposes we want to have a unique human readable identifier for each recording
        # we use the combination of subect, session and recording names for this
        recording_names: np.ndarray = np.array([
            f"{eeg_dataset.subjects[i]}_{eeg_dataset.sessions[i]}_{eeg_dataset.recordings[i]}"
            for i in range(len(data))
        ])

        # make sure all np.arrays of X are two dimensional
        assert all([len(x.shape) == 2 for x in data]), "All signals must be two dimensional"
        # make sure all signals have the same number of channels
        assert all([x.shape[1] == data[0].shape[1] for x in data]), "All signals must have the same number of channels"

        labels = []
        for metadata_i in metadata:
            labels_i = self._reduce_metadata(metadata_i)
            labels.append(labels_i)

        # get the number of samples for each label name
        label_name_counts = {
            label_name: np.mean([np.mean(labels_i == label_name) for labels_i in labels])
            for label_name in self.label_dict.keys()
        }
        # label_names are mapped to class indices, so we also want their counts
        label_indices_counts = {
            label_indices: sum(
                [label_name_counts[label_name]
                 for label_name, label_ind in self.label_dict.items()
                 if label_ind == label_indices])
            for label_indices in self.label_dict.values()
        }

        print("Label name counts:", json.dumps(label_name_counts, indent=4))
        print("Label indices counts:", json.dumps(label_indices_counts, indent=4))

        #label everything except math to 0
        label_mapping_fn = np.vectorize(lambda x: self.label_dict[x])

        for i in tqdm(range(len(labels))):
            labels[i] = label_mapping_fn(labels[i])
        
        discriminator_labels = None
        if self.discriminators_label_dict is not None:
            # apply the same code as for the labels, but for the discriminator labels
            discriminator_labels = defaultdict(list)
            for discriminator_name, discriminator_label_dict in self.discriminators_label_dict.items():
                # first collect all labels from the metadata
                for metadata_i in metadata:
                    discriminator_labels[discriminator_name].append(
                        self._reduce_metadata(metadata_i, f"label_{discriminator_name}")
                    )
                # get the number of samples for each label name
                discriminator_label_name_counts = {
                    label_name: np.mean([np.mean(labels_i == label_name)
                                        for labels_i in discriminator_labels[discriminator_name]])
                    for label_name in discriminator_label_dict.keys()
                }
                # label_names are mapped to class indices, so we also want their counts
                discriminator_label_indices_counts = {
                    label_indices: sum(
                        [discriminator_label_name_counts[label_name]
                            for label_name, label_ind in discriminator_label_dict.items()
                            if label_ind == label_indices])
                    for label_indices in discriminator_label_dict.values()
                }

                print(f"Discriminator {discriminator_name} label name counts:",
                      json.dumps(discriminator_label_name_counts, indent=4))
                print(f"Discriminator {discriminator_name} label indices counts:",
                      json.dumps(discriminator_label_indices_counts, indent=4))

                discriminator_label_mapping_fn = np.vectorize(lambda x: discriminator_label_dict[x])

                for i in tqdm(range(len(discriminator_labels[discriminator_name]))):
                    discriminator_labels[discriminator_name][i] = discriminator_label_mapping_fn(
                        discriminator_labels[discriminator_name][i])

        # Apply sliding window and sequence partitioning
        windowed_recordings = self.sliding_window(data, labels, discriminator_labels)

        data = self.preprocess_real_time(windowed_recordings[0], self.preprocessing_params)
        windowed_recordings = (data, *windowed_recordings[1:])

        (data_index,
         label_index) = self.apply_sequence_partitioning(*windowed_recordings[:4], **self.sequence_partitioning_params)

        self.set_data(
            *windowed_recordings,
            data_index=data_index,
            label_index=label_index,
            recording_names=recording_names,
        )

    def apply_sequence_partitioning(self,
                                    data: np.ndarray,
                                    labels: np.ndarray,
                                    recording_ids: Optional[np.ndarray],
                                    recording_order: Optional[np.ndarray],
                                    strategy: Optional[str] = None,
                                    calibration_sequence_start: Optional[int] = None,
                                    calibration_sequence_length: Optional[int] = None,
                                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies sequence partitioning to the windowed data.
        For training sequential models we need to partition the data of each recording into sequences
        which a model can process sequentially (e.g. LSTM, ConditionalEEGNet)
        There can be multiple partitioning strategies depending on the model.
        a) full recording: each sequence contains the whole recording (this includes padding the sequences to the same length), used for vanilla RNN
        b) fixed length: each sequence contains a fixed number of windows, used for fixed context length RNN
        c) calibration sequence: each sequence first contains a calibration sequence of a fixed length, followed by the sample to predict on, used for ConditionalEEGNet
        d) no partitioning: no partitioning is applied, used for vanilla EEGNet

        Parameters
        ----------
        strategy: str
            The strategy to use for sequence partitioning, possible values are "full_recording", "fixed_length", "calibration_sequence", None


        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
            Tuple containing the indices of the data, labels and metadata. 
            To be as memory efficient as possible, instead of returning copies of the data
            we create indices referring to the original data

        """

        if strategy == "full_recording":
            raise NotImplementedError("full_recording sequence partitioning not implemented yet")
        elif strategy == "fixed_length":
            raise NotImplementedError("fixed_length sequence partitioning not implemented yet")
        elif strategy == "calibration_sequence":
            # calibration sequence partitioning
            # sequence length is interpreted as number of windows in the calibration sequence
            assert calibration_sequence_length is not None, "calibration_sequence_length must be specified for calibration_sequence partitioning"
            assert calibration_sequence_start is not None, "calibration_sequence_start must be specified for calibration_sequence partitioning"
            # neither can be < 0
            assert calibration_sequence_length > 0, f"calibration_sequence_length must be > 0, got {calibration_sequence_length}"
            assert calibration_sequence_start >= 0, f"calibration_sequence_start must be >= 0, got {calibration_sequence_start}"
            unique_recording_ids = np.unique(recording_ids)
            # for each recording get the calibration sequence
            data_index = []
            label_index = []
            for recording_id in unique_recording_ids:
                # get indices of recording
                recording_indices = np.where(recording_ids == recording_id)[0]
                recording_order_i = recording_order[recording_indices]

                # make sure the recording_indices are in the correct order as specified by recording_order
                recording_indices = recording_indices[recording_order_i]

                # get calibration sequence, which is defined to be the first n windows of the recording
                calibration_sequence_end = calibration_sequence_start + calibration_sequence_length
                calib_data_index = recording_indices[calibration_sequence_start:calibration_sequence_end]

                # for this recording create the index of all the samples to predict on
                data_index.append(np.concatenate(
                    [np.broadcast_to(calib_data_index, (len(recording_indices), len(calib_data_index))),
                        recording_indices[:, None]],
                    axis=-1
                )
                )
                label_index.append(recording_indices)

            return (
                np.concatenate(data_index),
                np.concatenate(label_index)
            )
        elif strategy is None:
            return (np.arange(len(data))[..., None],
                    np.arange(len(labels)))
        else:
            raise ValueError("Unknown sequence partitioning strategy")

    def preprocess_real_time(self,
                             data: np.ndarray,
                             preprocessing_params: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Applies preprocessing to the already windowed data using the Preprocessor class

        Parameters
        ----------
        preprocessing_params: Dict[str, Any]
            Dictionary containing the preprocessing function name (from the Preprocessor class) and its parameters, 
            see Preprocessor.apply for more details
        """
        if len(preprocessing_params) > 0:
            eegdataset = EEGDataset.from_preprocessed_data([data], [{}] * len(data), old_dataset=self.eeg_dataset)

            data = Preprocessor(eegdataset, reduced_memory=True).apply(preprocessing_params).to_dataset().data[0]

        return data

    def get_class_weights(self) -> np.ndarray:
        """
        Returns the class weights for the current dataset
        """

        # get all labels:
        all_index = np.arange(len(self))
        all_labels = np.array(self.labels[self.label_index[all_index]])
        unique_labels = np.unique(all_labels)
        # using sklearn to compute class weights
        class_weight_vector = compute_class_weight(
            class_weight="balanced",
            classes=unique_labels,
            y=all_labels
        )
        # correcting for the case that some labels are not present in the dataset
        corrected_class_weight_vector = []
        for class_i in np.arange(0, self.n_classes):
            if class_i in unique_labels:
                corrected_class_weight_vector.append(class_weight_vector[np.where(unique_labels == class_i)[0][0]])
            else:
                corrected_class_weight_vector.append(0)

        return np.array(corrected_class_weight_vector)

    @staticmethod
    def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Only converts the data and labels to tensors, but not the metadata which is not required for training
        """

        data, labels = batch[:2]
        metadata = batch[2:5]
        if len(batch) == 6 and batch[5] is not None:
            # we have discriminator labels
            discriminator_labels: Dict[str, np.ndarray] = batch[5]

            # stack the discriminator labels and then concat them to the labels
            discriminator_labels = np.concatenate([np.expand_dims(discriminator_labels[key], axis=-1)
                                                  for key in discriminator_labels], axis=-1)
            labels = np.concatenate([np.expand_dims(labels, axis=-1), discriminator_labels], axis=-1)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long), *metadata


@typechecked
@register_transform(default=True)
def original(data: np.ndarray, metadata: Dict[np.ndarray], **kwargs) -> Tuple[np.ndarray, Dict[np.ndarray]]:
    assert len(data.shape) == 3, "Input must be of shape (n_sequences, window_size, n_channels)"
    return np.expand_dims(data, axis=-1), metadata


@typechecked
@register_transform(default=True)
def power_spectral_density(x: np.ndarray,
                           sampling_rate: int,
                           to_db=True,
                           return_freqs=False,
                           window_size: float = 1.0,
                           window_stride: Optional[float] = None,
                           **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    assert len(x.shape) == 3, "Input must be of shape (n_sequences, window_size, n_channels)"

    x = x.swapaxes(-1, -2)
    window_size = int(window_size * sampling_rate)
    window_stride = int(window_stride * sampling_rate) if window_stride is not None else sampling_rate
    x = np.lib.stride_tricks.sliding_window_view(x, window_size, axis=-1)[..., ::window_stride, :]
    # the shape of x is now (n_sequences, n_channels, n_windows, window_size),
    # but we want to swap the n_channels and n_windows dimensions
    x = x.swapaxes(1, 2)

    f, psd = welch(x, fs=sampling_rate)
    # n_samples is reduced to 1
    if to_db:
        psd = 10 * np.log10(psd+EPS)

    assert np.isnan(psd).sum() == 0, "NaN values in power spectral density"
    assert np.isfinite(psd).all(), "Infinite values in power spectral density"
    if return_freqs:
        return f, psd
    return psd


@typechecked
@register_transform(default=True)
def bandpower(x: np.ndarray,
              sampling_rate: int,
              **kwargs) -> np.ndarray:
    assert len(x.shape) == 3, "Input must be of shape (n_windows, window_size, n_channels)"
    f, psd = power_spectral_density(x, return_freqs=True, sampling_rate=sampling_rate, **kwargs)
    (n_windows, n_samples, n_channels, _) = psd.shape
    band_power = np.zeros((n_windows, n_samples, n_channels, len(DEFAULT_BANDS)))
    for j, band in enumerate(DEFAULT_BANDS):
        band_lower, band_upper = band
        freq_mask = (f >= band_lower) & (f < band_upper)
        assert freq_mask.sum() > 0, \
            f"No frequencies of band {band}({band_lower}Hz to {band_upper}Hz), try increasing the window size for the underlying power spectral density"
        band_power[..., j] = np.mean(psd[..., freq_mask], axis=-1)
    assert np.isnan(band_power).sum() == 0, "NaN values in bandpower"
    assert np.isfinite(band_power).all(), "Infinite values in bandpower"
    return band_power


@typechecked
@register_transform(default=True)
def bandpower_ratios(x: np.ndarray,
                     band_names: List[Tuple[str, str]] = DEFAULT_RATIOS,
                     band_idx: Dict[str, int] = DEFAULT_BAND_IDX,
                     **kwargs) -> np.ndarray:
    assert len(x.shape) == 3, "Input must be of shape (n_windows, window_size, n_channels)"
    bandpowers = bandpower(x, **kwargs)
    bandpower_ratios = np.zeros((*bandpowers.shape[:-1], len(band_names)))
    for j, (band_1, band_2) in enumerate(band_names):
        band_1_idx = band_idx[band_1]
        band_2_idx = band_idx[band_2]
        bandpower_ratios[..., j] = bandpowers[..., band_1_idx] / (bandpowers[..., band_2_idx]+EPS)

    assert np.isnan(bandpower_ratios).sum() == 0, "NaN values in bandpower_ratios"
    assert np.isfinite(bandpower_ratios).all(), "Infinite values in bandpower_ratios"

    return bandpower_ratios


@typechecked
@register_transform(default=True)
def downsample(x: np.ndarray,
               factor: int,
               **kwargs) -> np.ndarray:
    assert len(x.shape) == 3, "Input must be of shape (n_windows, window_size, n_channels)"
    return x[:, ::factor, :, None]


def split_index(index: np.ndarray, ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly split an index into two parts according to a ratio"""
    assert 0 < ratio < 1, "Ratio must be between 0 and 1"
    n = len(index)
    n_train = int(n * ratio)
    idx_1 = np.random.choice(index, n_train, replace=False)
    idx_2 = np.setdiff1d(index, idx_1)
    return idx_1, idx_2


if __name__ == "__main__":
    # Directories
    # DATA_DIR = Path(".data/real_time_data")
    # assert DATA_DIR.exists(), "Data directory does not exist"
    # channels = ["TP9", "AF7", "AF8", "TP10"]
    # data_files = {
    #     "data": {
    #         "pattern": "data.csv",
    #         "csv_read_kwargs": dict(sep=","),
    #         "csv_channel_cols": channels
    #     },
    # }
    from mind_ml.data import LABEL_MAP_PARADIGMS
    from mind_ml.utils import get_memory_usage
    start_memory = get_memory_usage()
    # subject = "EE831E72-96D4-4BCD-94F7-3690F8CC183B"
    # subject = "patient-10222021-01272022"
    # subject = "6A7242D6-A67A-481F-B635-055BCF42DAD1"
    # dataset = EEGDataset(
    #     data_dir=DATA_DIR,
    #     data_files=data_files,
    #     filters={
    #         # "subject": [subject],
    #         # "dataset": "training-data-ed850ad",
    #         "NotNaN": [],
    #         "all_labels_present": ["math"],
    #     },
    #     channel_names=channels,
    # )
    # print(f"Memory usage: {get_memory_usage() - start_memory} GB")
    # dataset.save_to_disk_numpy(Path(".data/training_data_ed850ad_numpy"))
    # del dataset

    dataset = EEGDataset.load_from_disk_numpy(Path(".data/datasets_realtime/cached_datasets/dataset_010"))

    # dataset = Preprocessor(dataset).normalize().to_dataset()

    py_dataset = EEGPytorchDataset(eeg_dataset=dataset,
                                   window_size=1.0,
                                   stride=0.25,
                                   sampling_rate=256,
                                   label_dict=LABEL_MAP_PARADIGMS["math_vs_all"],
                                   preprocessing_params={"normalize": {}})

    train_test_ratio = 0.8
    batch_size = 32

    # shapes of the data
    print(py_dataset.shape)

    train_idx, test_idx = split_index(np.arange(len(py_dataset)), train_test_ratio)

    # fixed idx for debugging
    # train_idx = np.arange(int(len(py_dataset) * train_test_ratio))
    # test_idx = np.arange(int(len(py_dataset) * (train_test_ratio)), len(py_dataset))

    train_dataset, test_dataset = EEGPytorchDataset.splits(py_dataset, [train_idx, test_idx])
    # print the sizes of the datasets
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, collate_fn=EEGPytorchDataset.collate_fn)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
    #                                          shuffle=False, collate_fn=EEGPytorchDataset.collate_fn)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
    #                                           shuffle=False, collate_fn=EEGPytorchDataset.collate_fn)

    for i, (batch) in tqdm(enumerate(train_loader)):
        # print(data)
        data, labels = batch[:2]

        continue

    # for i, (data) in tqdm(enumerate(val_loader)):
    #     # print(data)
    #     continue

    # for i, (data) in tqdm(enumerate(test_loader)):
    #     # print(data)
    #     continue
