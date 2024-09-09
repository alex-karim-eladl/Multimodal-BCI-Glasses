import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import yaml
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchsummary import summary
from typeguard import typechecked

from data import create_data_loaders
from models import create_model, create_trainer


@typechecked
def train_iteration(device: str,
                    device_id: int,
                    seed: int,
                    num_workers: int):

    run = wandb.init()
    wandb.config["seed"] = seed
    config = wandb.config

    pl.seed_everything(seed)

    train_dataloader, val_dataloader, _ = create_data_loaders(data_config=dict(config.data),
                                                              num_workers=num_workers)

    # get data shape
    batch_size, sequence_length, sample_length, num_channels = next(iter(train_dataloader))[0].shape
    if config.model["name"] == "EEGNet":
        model_summary_input_size = (1, num_channels, sample_length)
    elif config.model["name"] == "ConditionalEEGNet":
        model_summary_input_size = (sequence_length, 1, num_channels, sample_length)
    elif config.model["name"] == "DANN":
        model_summary_input_size = (1, num_channels, sample_length)
    elif config.model["name"] == "CNN":
        model_summary_input_size = (num_channels, sample_length)
    else:
        model_summary_input_size = None
    # create model
    model = create_model(model_config=dict(config.model),
                         optimizer_config=dict(config.optimizer),
                         batch_size=batch_size,
                         sample_length=sample_length,
                         num_channels=num_channels,
                         num_classes=train_dataloader.dataset.n_classes,
                         num_classes_discriminator=train_dataloader.dataset.n_classes_discriminator,
                         class_weights=train_dataloader.dataset.get_class_weights(),
                         model_summary_input_size=model_summary_input_size,
                         base_model_params=getattr(config, "base_model", None))

    # create trainer
    trainer = create_trainer(trainer_config=dict(config.trainer),
                             device=device,
                             device_id=device_id)

    wandb_logger = WandbLogger(save_dir="./wandb-logs/")
    wandb_logger.watch(model, log="all", log_freq=10)

    trainer.fit(model, train_dataloader, val_dataloader)

    # check whether there is a ModelCheckpoint in the trainer.callbacks
    # if so, load the best model
    for callback in trainer.callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            best_model = callback.best_model_path
            print(f"Loading best model from {best_model}")
            model_checkpoint = torch.load(best_model)
            model.load_state_dict(model_checkpoint["state_dict"])
            trainer.test(model, val_dataloader)

    # save model
    model_file_path = Path(wandb.run.dir) / "model.pth"
    print(f"Saving model to {model_file_path}")

    config["sample_length"] = sample_length
    config["num_channels"] = num_channels

    model_artifact = wandb.Artifact(
        name=f"{config.model['name']}-{run.name}",
        type="model",
        metadata=dict(config=dict(config))
    )

    torch.save(model.state_dict(), model_file_path)
    model_artifact.add_file(model_file_path)
    run.log_artifact(model_artifact)

    print(f"Finished training for run [{run.id}] - '{run.name}'")


def default_train_iteration():

    # check if torch cuda is available
    if torch.cuda.is_available():
        device = "gpu"
    else:
        device = "cpu"

    try:
        train_iteration(device=device,
                        device_id=0,
                        seed=0,
                        num_workers=0)
    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc())
        exit(1)


@typechecked
def merge_dicts(base_dict: dict, extension_dict: dict, bread_crumb: List[str] = []) -> dict:
    """
    Merges two dictionaries, give precedence to the extension_dict.
    Provide warning if a key is overwritten.
    If a value is a dictionary, the function is called recursively.
    """
    base_dict = base_dict.copy()
    for key, value in extension_dict.items():
        current_bread_crumb = bread_crumb + [key]

        # wandb configs only allow either a value or values key within a dictionary
        # therefore when merging we want to overwrite value with values and vice versa
        if key == "value" and "values" in base_dict:
            base_dict["value"] = base_dict.pop("values")
        elif key == "values" and "value" in base_dict:
            base_dict["values"] = base_dict.pop("value")

        if key in base_dict:
            if isinstance(base_dict[key], dict):
                assert isinstance(value, dict),\
                    f"Cannot merge {'.'.join(current_bread_crumb)}, because it is a dictionary in the base_dict and a {type(value)} in the extension_dict."
                base_dict[key] = merge_dicts(base_dict[key], value, bread_crumb=current_bread_crumb)
                continue
            else:
                print(
                    f"Warning: Overwriting key {'.'.join(current_bread_crumb)} with value {value}, previous value was {base_dict[key]}")
        base_dict[key] = value
    return base_dict


@typechecked
def read_yaml(path: Union[str, Path]) -> dict:
    """
    Reads a yaml file.
    """
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)


@typechecked
def read_sweep_yaml(sweep_yaml: Union[str, Path]) -> dict:
    """
    Reads a yaml file for wandb sweeps.
    Allows yaml files to be extended via "extends: node at the root level.

    The value of extends can be a string or a list of strings, 
    which are interpreted as paths to other yaml files.
    These files are read one after another and merged into a single dictionary.
    Already existing keys will be overwritten.
    """
    yaml_dict = read_yaml(sweep_yaml)
    if "extends" in yaml_dict:
        base_yamls = yaml_dict.pop("extends")
        if isinstance(base_yamls, str):
            base_yamls = [base_yamls]
        for base_yaml in base_yamls:
            yaml_dict = merge_dicts(read_sweep_yaml(base_yaml), yaml_dict)
    return yaml_dict


def initialize_sweep(sweep_yaml: Union[str, Path], project: str = "debug", entity: Optional[str] = None,) -> str:

    # load yaml as dict
    sweep_config = read_sweep_yaml(sweep_yaml)

    # initialize wandb sweep
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    return sweep_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a wandb agent for a given sweep"
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default="tobiasschmidt/debug/fonlqsmy",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        help="device to use for training either 'cpu' or 'gpu'"
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="gpu device id to use for training, ignored if device is 'cpu'"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed to use for training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of workers of torch dataloaders"
    )
    parser.add_argument(
        "--run_counts",
        type=int,
        default=1,
        help="number of runs to execute, 0 means infinite"
    )

    args = parser.parse_args()
    print(args.sweep_id)
    sweep_id_parts = args.sweep_id.split("/")
    assert len(sweep_id_parts) in [2, 3], f"Invalid sweep id {args.sweep_id}"
    entity = sweep_id_parts[-3] if len(sweep_id_parts) == 3 else None
    project = sweep_id_parts[-2]
    sweep_id = sweep_id_parts[-1]

    wandb.agent(args.sweep_id,
                project=project,
                entity=entity,
                function=lambda: train_iteration(device=args.device,
                                                 device_id=args.device_id,
                                                 seed=args.seed,
                                                 num_workers=args.num_workers),
                count=args.run_counts)
