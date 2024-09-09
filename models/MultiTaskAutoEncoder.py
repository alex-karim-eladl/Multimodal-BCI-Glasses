import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import List, Tuple
from typeguard import typechecked
from torchsummary import summary


# Note to self: To reduce all kinds of encoders to the same embedding space simply
# use a flatten and final dense layer to map to a fixed size embedding space.
# This way the encoder architecture can be arbitrary

@typechecked
class MultiTaskAutoEncoder(pl.LightningModule):
    def __init__(self,
                 transforms: List[str],
                 transform_input_shapes: List[Tuple[int, int, int]],
                 embedding_dim: int = 64,
                 num_blocks: int = 3,
                 ):
        super().__init__()
        self.transforms = transforms
        self.transform_input_shapes = transform_input_shapes
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks

        encoders = {}
        decoders = {}

        # the input to each encoder is of shape [n_batches, n_samples, n_channels, n_features]
        # we want to aggressively reduce the dimensionality of the data
        # the embedding should be of shape [n_batches, embedding_dim]
        # the output of each decoder should be of shape [n_batches, n_samples, n_channels, n_features]

        # since n_samples is constant within a given transform we can use conv layers to make our life easier for now

        for transform, (n_samples, n_channels, n_features) in zip(transforms, transform_input_shapes):

            kernal_height = int(math.ceil(n_samples/3))
            kernal_width = int(math.ceil(n_features/2))
            pool_kernal_height = int(math.ceil(kernal_height/2))
            pool_kernal_width = int(math.ceil(kernal_width/2))

            channel_multiplier = embedding_dim / (num_blocks * n_channels)
            print(
                f"transform: {transform}, kernal_height: {kernal_height}, kernal_width: {kernal_width}, channel_multiplier: {channel_multiplier}")
            encoder = []
            in_channels = n_channels
            for i in range(num_blocks):
                out_channels = int(math.floor(channel_multiplier * in_channels))
                encoder_block = nn.Sequential(
                    # input shape [n_channels, n_samples, n_features]
                    nn.Conv2d(in_channels, out_channels, kernel_size=(kernal_height, kernal_width), stride=1, padding="same"),
                    # outout shape [out_channels, n_samples, n_features] since padding is "same"
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(pool_kernal_height, pool_kernal_width), stride=1),
                    # output shape [out_channels, n_samples/2, n_features/2]
                )
                in_channels = out_channels
                encoder.append(encoder_block)
            encoders[transform] = nn.Sequential(*encoder)
        self.encoders = nn.ModuleDict(encoders)

    def forward(self, x):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):

        raise NotImplementedError()

    def configure_optimizers(self):

        raise NotImplementedError()


if __name__ == "__main__":
    model = MultiTaskAutoEncoder(["original", "bandpower", "psd"], [(256, 4, 1), (6, 4, 68), (1, 4, 129)])
    summary(model, (256, 4, 1), device="cpu")
