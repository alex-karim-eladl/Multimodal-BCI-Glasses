from typing import Dict, Any, Tuple, Union, Optional, Callable, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchsummary import summary
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC
from mind_ml.models.EEGNet import EEGNetLightning

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

class CNN(nn.Module):
    """
    CNN model for EEG data.
    """

    def __init__(self,
                 channels: int,
                 dropout: float,
                 kernel_size: int,
                 sample_rate: int = 256,
                 chan_out: int = 256,
                 pool_out: int = 120, 
                 n_blocks: int = 3,
                 **kwargs):
        """
        Parameters
        ----------
        channels : int
            Number of EEG channels.
        dropout : float
        """

        super().__init__()

        self.channels = channels
        self.sample_rate = sample_rate
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.chan_out = chan_out
        self.pool_out = pool_out
        self.n_blocks = n_blocks

        """
        TO DO:
        """

        self.blocks = dict()
        for n in range(n_blocks):
            self.blocks[f"block_{n}"] = nn.Sequential(
                nn.Conv1d(in_channels=channels,
                        out_channels=chan_out,
                        kernel_size=kernel_size,
                        padding="same",
                        ),
                nn.LazyBatchNorm1d(),
                nn.GELU(),
                nn.AdaptiveMaxPool1d(output_size = pool_out),
                nn.Dropout(p=dropout),
            )
            channels = chan_out
            chan_out = chan_out//2
            pool_out = pool_out//2

        #This is just to make it run with EEGNetLightning
        self.conv_net = nn.Sequential(*[
            block for block in self.blocks.values()
        ])
        self.classifier_head = nn.Sequential(*[
            nn.Flatten(),
            nn.LazyLinear(out_features = 2), 
            nn.GELU()
            ])

    def forward(self, x: TensorType["num_batches", "num_channels", "num_samples"]) -> TensorType["num_batches", "kernal_size", "reduced_channels", 1]:
        # for n in range(self.n_blocks):
        #     x = self.blocks[f"block_{n}"](x)
        x = self.conv_net(x)
        x = self.classifier_head(x)
        return x

class CNNLightning(EEGNetLightning):
    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.eegnet = CNN(**hparams)

    def forward(self, x: TensorType["num_batches", 1, "num_channels", "num_samples"]) -> TensorType["num_batches", "num_classes"]:
        # x is of shape (..., samples, channels) but we need (..., 1, channels, samples)
        #x = torch.swapaxes(x, -2, -1)
        #x = x.unsqueeze(-3)
        x = x.squeeze(axis=1)
        x = torch.swapaxes(x, -2, -1)
        y_hat = self.eegnet(x)
        return y_hat

    def configure_optimizers(self):
        assert hasattr(torch.optim, self.hparams.optimizer_class),\
            f"{self.hparams.optimizer_class} is not a valid optimizer from torch.optim"
        optimizer_class = getattr(torch.optim, self.hparams.optimizer_class)
        optimizer = optimizer_class(
            [
                {  # conv layer parameters
                    "params": filter(lambda p: p.requires_grad, self.eegnet.conv_net.parameters()),
                    "weight_decay": getattr(self.hparams, "conv_weight_decay", 0.0),
                },
                {  # fc layer parameters
                    "params": filter(lambda p: p.requires_grad, self.eegnet.classifier_head.parameters()),
                    "weight_decay": getattr(self.hparams, "fc_weight_decay", 0.0),
                },
            ],
            lr=self.hparams.learning_rate)
        
        # lr = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer, base_lr = self.hparams.learning_rate,
        #     max_lr = 4*self.hparams.learning_rate,
        #     step_size_up = 4*int(self.stepsize),
        #     mode = "triangular",
        #     cycle_momentum = False
        #     )

        lr = torch.optim.lr_scheduler.OneCycleLR(
            optimizer = optimizer,
            max_lr = self.hparams.learning_rate,
            epochs = self.trainer.max_epochs,
            steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs,
            cycle_momentum = True
            )
        
        scheduler = {
            "scheduler": lr,
            "interval": "step",
            "name": "Learning Rate Scheduling"
        }
        return [optimizer], [scheduler]
   