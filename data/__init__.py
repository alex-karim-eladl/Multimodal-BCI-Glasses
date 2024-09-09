from typing import List, Any, Dict, Tuple, Union, Optional, Callable
from pathlib import Path
import json

from filelock import SoftFileLock

import torch
from torch.utils.data import DataLoader

import numpy as np

from typeguard import typechecked

from data.datasets.dataset import EEGDataset
from data.datasets.manager import DatasetManager
from data.datasets.pytorch import EEGPytorchDataset, split_index


LABEL_MAP_PARADIGMS = {
    "math_vs_all": {
        "math": 1,
        "calibration": 0,
        "rest": 0,
        "nan": 0,
        "done": 0,
        "break": 0,
        "unknown": 0
    },
    "train_vs_all": {
        "train": 0,
        "val": 1,
        "test": 1,
    },
    "train_vs_val": {
        "train": 0,
        "val": 1,
    },
    "train_vs_val_vs_test": {
        "train": 0,
        "val": 1,
        "test": 2,
    },
    "random_binary": {
        "0": 0,
        "1": 1,
    },
}


@typechecked
def validate_data_config(base_data_dir: Union[str, Path],
                         base_dataset_name: Union[str, Path],
                         batch_size: int,
                         shuffle_train: bool,
                         label_mapping_paradigm: str,
                         preprocessing: Optional[Dict[str, Any]] = None,
                         preprocessing_real_time: Optional[Dict[str, Any]] = None,
                         sequence_partitioning_params: Optional[Dict[str, Any]] = None,
                         dataset_filters: Optional[Dict[str, Any]] = None,
                         train_val_ratio: Optional[float] = None,
                         train_val_filters: Optional[Dict[str, Any]] = None,
                         **kwargs) -> None:
    """
    Validate the data configuration parameters as so far as they are not already typeched down the line
    """
    assert label_mapping_paradigm in LABEL_MAP_PARADIGMS.keys() or label_mapping_paradigm == "auto",\
        "Label mapping paradigm not supported, please choose from: {}".format(list(LABEL_MAP_PARADIGMS.keys()))
    pass


@typechecked
def create_data_loaders(data_config: Dict[str, Any],
                        num_workers: int,
                        return_meta: bool = False,
                        return_sampling_rate: bool = False,
                        skip_test_dataset: bool = True,
                        ) -> Union[
    Tuple[DataLoader, DataLoader, Optional[DataLoader]],
    Tuple[DataLoader, DataLoader, Optional[DataLoader], int],
]:

    data_params = dict(data_config)
    # assert existens of all required config values
    validate_data_config(**data_params)

    # pop out all non-dataset related data config values
    # data loading parameters
    base_data_dir = data_params.pop("base_data_dir")
    base_dataset_name = data_params.pop("base_dataset_name")
    dataset_filters = data_params.pop("dataset_filters", None)
    preprocessing = data_params.pop("preprocessing", None)
    preprocessing_real_time = data_params.pop("preprocessing_real_time", {})
    sequence_partitioning_params = data_params.pop("sequence_partitioning_params", {})

    # datasplit params
    train_test_ratio = data_params.pop("train_test_ratio", None)
    train_val_ratio = data_params.pop("train_val_ratio", None)
    # train_val_filters explicitly specifies train recordings
    train_filters = data_params.pop("train_filters", None)
    # for backwards compatibility check old name "train_val_filters"
    if train_filters is None:
        train_filters = data_params.pop("train_val_filters", None)
    # val_filters explicitly specifies val recordings
    val_filters = data_params.pop("val_filters", None)
    # test_filters explicitly specifies test recordings
    test_filters = data_params.pop("test_filters", None)
    shuffle_train = data_params.pop("shuffle_train")

    target_variable = data_params.pop("target_variable", None)
    # for DANN models we need to specify the target variable for the discriminator
    discriminator_target_variable = data_params.pop("discriminator_target_variable", None)
    discriminator_target_variable

    # torch Dataset params
    window_size = data_params.pop("window_size")
    window_stride = data_params.pop("window_stride")
    label_mapping_paradigm = data_params.pop("label_mapping_paradigm")
    relabeling_strategy = data_params.pop("relabeling_strategy", None)

    # torch DataLoader Params
    batch_size = data_params.pop("batch_size")

    assert len(data_params) == 0, "Unrecognized data config parameters: {}".format(data_params.keys())

    # load data from disk
    dataset_manager = DatasetManager(base_data_dir=base_data_dir)
    dataset = dataset_manager.load_dataset(
        dataset_info=dict(
            filters=dataset_filters,
            preprocessing=preprocessing,
            base_dataset_path=base_dataset_name,
        ),
        include=["label"]
    )

    if target_variable is not None:
        dataset = dataset.relabel(target_variable=target_variable,
                                  test_filters=test_filters,
                                  val_filters=val_filters)
        if label_mapping_paradigm == "auto":
            # generate label mapping paradigm based on target variable
            # map each unique value to a unique integer, making sure the unique values are sorted
            unique_values = np.sort(dataset.unique_labels)
            LABEL_MAP_PARADIGMS["auto"] = {str(val): i for i, val in enumerate(unique_values)}

    if discriminator_target_variable is not None:
        discriminator_target_variable = [discriminator_target_variable] if isinstance(
            discriminator_target_variable, str) else discriminator_target_variable
        dataset = dataset.relabel(target_variable=discriminator_target_variable,
                                  test_filters=test_filters,
                                  val_filters=val_filters,
                                  overwrite_original=False)
        # for discriminator we will automatically add a new label mapping paradigm
        for target_var in dataset.get_discriminator_labels():
            unique_values = np.sort(dataset.get_unqiue_meta(f"label_{target_var}"))
            LABEL_MAP_PARADIGMS[f"discriminator_{target_var}"] = {
                str(val): i for i, val in enumerate(unique_values)}

    sampling_rate = dataset._sampling_rate
    if isinstance(sampling_rate, np.ndarray):
        sampling_rate = sampling_rate.item()

    # split into train and validation
    pytorch_dataset_params = dict(
        window_size=window_size,
        stride=window_stride,
        sampling_rate=sampling_rate,
        label_dict=LABEL_MAP_PARADIGMS[label_mapping_paradigm],
        discriminators_label_dict={
            target_var: LABEL_MAP_PARADIGMS[f"discriminator_{target_var}"]
            for target_var in dataset.get_discriminator_labels()
        } if discriminator_target_variable is not None else None,
        preprocessing_params=preprocessing_real_time,
        sequence_partitioning_params=sequence_partitioning_params,
        relabeling_strategy=relabeling_strategy,
    )

    py_test_dataset = None
    test_dataloader = None

    use_intra_recording_split = (train_filters is None and val_filters is None) or (
        target_variable is not None and train_val_ratio is not None)
    if use_intra_recording_split:
        # convert to pytorch dataset first (windowing the data)
        # and then split into train and validation based on windowed data
        py_dataset = EEGPytorchDataset(eeg_dataset=dataset, **pytorch_dataset_params, return_meta=return_meta)

        all_data_idx = np.arange(len(py_dataset))
        if train_test_ratio is not None:
            all_data_idx, test_idx = split_index(all_data_idx, train_test_ratio)
        train_idx, val_idx = split_index(all_data_idx, train_val_ratio)

        if train_test_ratio is not None and not skip_test_dataset:
            py_train_dataset, py_val_dataset, py_test_dataset = EEGPytorchDataset.splits(
                py_dataset, [train_idx, val_idx, test_idx])

        py_train_dataset, py_val_dataset = EEGPytorchDataset.splits(py_dataset, [train_idx, val_idx])
    else:
        # first split test data
        test_dataset = None
        if test_filters is not None:
            # first return is data confirming with filters, second one is all the remaining data
            test_dataset, dataset = dataset.split_rulebased(filters=test_filters)
        if val_filters is not None:
            val_dataset, train_dataset = dataset.split_rulebased(filters=val_filters)
            if train_filters is not None:
                train_dataset, _ = train_dataset.split_rulebased(filters=train_filters)
        elif train_filters is not None:
            train_dataset, val_dataset = dataset.split_rulebased(filters=train_filters)
        else:
            raise ValueError("Either val_filters or train_filters or both must be specified")

        py_train_dataset = EEGPytorchDataset(eeg_dataset=train_dataset, **pytorch_dataset_params, return_meta=return_meta)
        py_val_dataset = EEGPytorchDataset(eeg_dataset=val_dataset, **pytorch_dataset_params, return_meta=return_meta)
        if test_dataset is not None and not skip_test_dataset:
            py_test_dataset = EEGPytorchDataset(eeg_dataset=test_dataset, **pytorch_dataset_params, return_meta=return_meta)

    print(f"Train dataset size: {len(py_train_dataset)}")
    print(f"Validation dataset size: {len(py_val_dataset)}")

    # create data loaders
    train_dataloader = torch.utils.data.DataLoader(py_train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                                   collate_fn=EEGPytorchDataset.collate_fn, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(py_val_dataset, batch_size=batch_size, shuffle=False,
                                                 collate_fn=EEGPytorchDataset.collate_fn, num_workers=num_workers)
    if py_test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(py_test_dataset, batch_size=batch_size, shuffle=False,
                                                      collate_fn=EEGPytorchDataset.collate_fn, num_workers=num_workers)
    if return_sampling_rate:
        return train_dataloader, val_dataloader, test_dataloader, sampling_rate
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":

    d1, d2 = create_data_loaders(
        data_config=dict(
            base_data_dir=".data/datasets_realtime/",
            base_dataset_name="original_data_numpy",
            dataset_filters={
                "subject": ["patient-10222021-01272022"],
                "all_labels_present": ["math"]
            },
            train_val_ratio=0.8,
            window_size=1,
            window_stride=0.25,
            shuffle_train=True,
            label_mapping_paradigm="math_vs_all",
            batch_size=512,
            preprocessing=dict(
                normalize={}
            ),
        ),
        num_workers=0
    )
