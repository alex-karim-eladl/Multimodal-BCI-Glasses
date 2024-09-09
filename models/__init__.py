from typing import List, Any, Dict, Tuple, Union, Optional
from pathlib import Path

import torch
from typeguard import typechecked
from torchsummary import summary
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
import wandb
from mind_ml.models.EEGNet import EEGNetLightning, ConditionalEEGNetLightning
from mind_ml.models.DANN import DANNLightning
import pytorch_lightning as pl

from mind_ml.models.CNN import CNNLightning

@typechecked
def create_model(model_config: Dict[str, Any],
                 optimizer_config: Dict[str, Any],
                 sample_length: int,
                 num_channels: int,
                 base_model_params: Optional[Dict[str, Any]] = None,
                 model_summary_input_size: Optional[tuple] = None,
                 num_classes: int = 2,
                 num_classes_discriminator: Optional[Dict[str, int]] = None,
                 **kwargs) -> pl.LightningModule:

    model_params = dict(model_config)
    if "EEGNet" in model_config["name"]:
        model_params.update(dict(
            num_classes=num_classes,
            channels=num_channels,
            sample_length=sample_length
        ))
    elif model_config["name"] == "DANN":
        model_params["feature_extractor_kwargs"].update(dict(
            channels=num_channels,
            sample_length=sample_length
        ))
        model_params["task_classifier_kwargs"].update(dict(
            num_classes=num_classes,
        ))
        model_params["discriminator_classifier_kwargs"].update(dict(
            num_classes=list(num_classes_discriminator.values())
        ))
    elif model_config["name"] == "CNN":
        model_params.update(dict(
            num_classes=num_classes,
            channels=num_channels,
            sample_length=sample_length
        ))
    else:
        raise ValueError(f"Model {model_config['name']} not supported.")

    model_params.update(dict(**kwargs))
    model_params.update(dict(optimizer_config))

    if model_config["name"] == "EEGNet":
        model = EEGNetLightning(**model_params)
    elif model_config["name"] == "ConditionalEEGNet":
        model = ConditionalEEGNetLightning(**model_params)
    elif model_config["name"] == "DANN":
        model = DANNLightning(**model_params)
    elif model_config["name"] == "CNN":
        model = CNNLightning(**model_params)
    else:
        raise ValueError(f"Model {model_config['name']} not supported.")

    # check if model_config contains checkpoint
    if base_model_params is not None:
        assert "checkpoint" in base_model_params, "base_model_params must contain checkpoint"
        checkpoint = base_model_params.pop("checkpoint")
        freeze_layers = base_model_params.pop("freeze_layers", {})
        reset_layers = base_model_params.pop("reset_layers", {})
        # checkpoint is the path to a wandb artifact
        # download the artifact and load the model
        artifact = wandb.use_artifact(checkpoint)
        artifact_dir = artifact.download()
        model_path = Path(artifact_dir) / "model.pth"
        state_dict = torch.load(model_path)

        # instead of resetting the model, we only partially load the state dict
        # this way we don't need to reset the layers manually and can be sure that
        # the model is initialized correctly
        updated_state_dict = {
            key: value
            for key, value in state_dict.items()
            if not any([key.startswith(reset_layer_name)
                        for reset_layer_name, reset in reset_layers.items()
                        if reset])
        }

        num_parameters_loaded = 0
        for key, value in updated_state_dict.items():
            print(f"Loading {key} parameters of shape {value.shape}, totalling {value.numel()} parameters.")
            num_parameters_loaded += value.numel()
        print(f"Loaded {num_parameters_loaded} parameters in total.")

        incompatible_keys = model.load_state_dict(updated_state_dict, strict=False)

        # check if there are any unexpectadly incompatible keys
        # 1. there should be no "unexpected" keys
        # 2. there should be no "missing" keys which are not in the reset_layers
        assert len(
            incompatible_keys.unexpected_keys) == 0, f"Unexpected keys: {incompatible_keys.unexpected_keys} when loading model from {checkpoint}"
        assert all([
            any([key.startswith(reset_layer_name)
                 for reset_layer_name, reset in reset_layers.items()
                 if reset])
            for key in incompatible_keys.missing_keys
        ]), f"Found missing keys {incompatible_keys.missing_keys} which are not in the reset_layers {reset_layers} when loading model from {checkpoint}"

        for layer_identifier, freeze in freeze_layers.items():
            if freeze:
                print(f"Freezing layer {layer_identifier}")
            # layer_identifier are strings like "conv_net.0"
            # which are used to access inidivudal layers or submodules of the model
            layer = get_model_layer(layer_identifier, model.eegnet)
            for param in layer.parameters():
                param.requires_grad = not freeze

    # verify model
    if model_summary_input_size:
        summary(model.eegnet, model_summary_input_size, device="cpu")
    return model


def get_model_layer(layer_identifier: str, model: torch.nn.Module) -> torch.nn.Parameter:
    layer_id_parts = layer_identifier.split(".")  # split into list of strings
    layer = model
    for layer_id in layer_id_parts:
        try:
            if isinstance(layer, torch.nn.Sequential):
                layer = layer[int(layer_id)]
            else:
                layer = getattr(layer, layer_id)
        except Exception as e:
            print(f"Error while accessing layer {layer_identifier} in model {model}")
            print(f"Layer {layer} has no attribute {layer_id}")
            raise e
    return layer


def prepare_environment(device: str, device_id: int):
    if device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        # for debugging purposes, will slow down training but report propper stacktraces for CUDA errors
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def create_trainer(trainer_config: Dict[str, Any], device: str, device_id: int, use_wandb_logger: bool = True) -> pl.Trainer:
    print(f"Using {device} with id {device_id}" if device == "gpu" else f"Using {device}")
    prepare_environment(device=device, device_id=device_id)

    # setup WandB logger
    if use_wandb_logger:

        wandb_logger = WandbLogger(save_dir="./wandb-logs/")
        trainer_config = dict(trainer_config)  # make a copy so we don't modify the original
        trainer_config.update(dict(logger=wandb_logger))

    #Track learning rate
    lr_monitor = LearningRateMonitor()
    trainer_config["callbacks"] = [lr_monitor]

    if "early_stop_callback" in trainer_config:
        early_stopping_params = trainer_config.pop("early_stop_callback")
        assert early_stopping_params is not None and isinstance(
            early_stopping_params, dict), "early_stop_callback must be a dict"
        early_stop_callback = EarlyStopping(**early_stop_callback)
        trainer_config["callbacks"].append(early_stop_callback)

    if "model_checkpoint" in trainer_config:
        model_checkpoint_params = trainer_config.pop("model_checkpoint")
        assert model_checkpoint_params is not None and isinstance(
            model_checkpoint_params, dict), "model_checkpoint must be a dict"
        model_checkpoint = ModelCheckpoint(**model_checkpoint_params)
        trainer_config["callbacks"].append(model_checkpoint)

    trainer = pl.Trainer(accelerator=device, **trainer_config)
    return trainer
