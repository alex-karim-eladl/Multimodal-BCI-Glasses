from typing import Dict, Any, Tuple, Union, Optional, Callable, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchsummary import summary
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


# Keras offers a DepthwiseConv2D layer as well as a SeparableConv2D layer.
# The DepthwiseConv2D layer performs a depthwise convolution that acts separately on channels,
# while the SeparableConv2D performs a depthwise convolution that acts separately on channels, followed by a pointwise convolution that mixes channels.
# The pytorch equivalent is as follows:
class DepthwiseConv2d(nn.Module):
    """
    From the documentation of torch.nn.Conv2d:
    If groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is also known as a depthwise convolution.
    In pther words, for an input of size (N, C_in, L_in), a depthwise convolution with a depthwise multiplier K,
    can be constructed by providing the arguments (C_in = C_in, C_out = C_in * K, ..., groups = C_in).
    """

    def __init__(self, in_channels, depth_multiplier, **kwargs):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels,
                                   out_channels=in_channels * depth_multiplier,
                                   groups=in_channels,
                                   **kwargs)

    def forward(self, x):
        out = self.depthwise(x)
        return out


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, depth_multiplier=1, **kwargs):
        super(SeparableConv2d, self).__init__()
        self.depthwise = DepthwiseConv2d(in_channels, depth_multiplier, **kwargs)
        self.pointwise = nn.Conv2d(in_channels=in_channels * depth_multiplier,
                                   out_channels=out_channels,
                                   kernel_size=(1, 1))

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class EEGNetBackbone(nn.Module):
    """
    Pytorch implementation of the EEGNet's backbone (convnet) from Lawhern et al. 2018.

    Reference Implementation of EEGNet Version 3 from original authors:
    https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
    """

    def __init__(self,
                 channels: int,
                 sample_length: int,
                 kernel_size: int = 64,
                 dropout: float = 0.0,
                 f1: int = 8,
                 d: int = 2,
                 f2: int = 16,
                 sample_rate: int = 256,
                 adjust_for_sample_length: bool = False,
                 **kwargs):
        """
        Parameters
        ----------
        channels : int
            Number of channels in the input data.
        sample_rate : int
            Sample rate of the input data, the architecture is designed for 256 Hz, but should work for higher sample rates.
        dropout : float
            Dropout rate.
        kernel_size : int
            Length of the temporal convolution kernal in the first layer.
        f1, f2 : int
            Number of temporal filters (F1) and number of pointwise filters (F2) to learn.
            Default: F1 = 8, F2 = F1 * D
        d : int
            Number of spatial filters to learn within each temporal convolution.
            Default: d = 2
        """

        super().__init__()

        self.channels = channels
        self.sample_rate = sample_rate
        self.sample_length = sample_length
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.f1 = f1
        self.d = d
        self.f2 = f2

        # The authors describe their updated model as follows:
        # There are two CNN blocks followed by a fully connected layer.
        # Block 1:
        #  - Vanilla 2D Convolution with same padding and kernal size (1, kernel_size)
        #  - Batch normalization
        #  - Depthwise Convolution with kernal size (channels, 1) and depth multiplier d
        #  - Batch normalization
        #  - ELU activation
        #  - Average pooling with kernal size (1, 4)
        #  - Dropout or Spatial Dropout
        # Block 2:
        #  - Depthwise Separable Convolution with output channels f2, kernal size (1, 16) and same padding
        #  - Batch normalization
        #  - ELU activation
        #  - Average pooling with kernal size (1, 8)
        #  - Dropout or Spatial Dropout
        # Flatten

        """
        The following is the orignal implementation using keras


        input1   = Input(shape = (Chans, Samples, 1))
        block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                    input_shape = (Chans, Samples, 1),
                                    use_bias = False)(input1)
        block1       = BatchNormalization()(block1)
        block1       = DepthwiseConv2D((Chans, 1), use_bias = False,
                                    depth_multiplier = D,
                                    depthwise_constraint = max_norm(1.))(block1)
        block1       = BatchNormalization()(block1)
        block1       = Activation('elu')(block1)
        block1       = AveragePooling2D((1, 4))(block1)
        block1       = dropoutType(dropoutRate)(block1)

        block2       = SeparableConv2D(F2, (1, 16),
                                    use_bias = False, padding = 'same')(block1)
        block2       = BatchNormalization()(block2)
        block2       = Activation('elu')(block2)
        block2       = AveragePooling2D((1, 8))(block2)
        block2       = dropoutType(dropoutRate)(block2)

        flatten      = Flatten(name = 'flatten')(block2)
        """

        # We want the pytorch equivalent of the keras implementation above
        # Note that the input shape for pytorch is (Batch, Channels, Height, Width)
        # for our case height will be the spatial dimension and width will be the temporal dimension
        # spatial dimension is the number of recording channels (not to be confused with "channels" from the pytorch perspective)
        # temporal dimension is the number of samples
        self.pool1 = 4
        # Block 1
        self.block1 = nn.Sequential(
            # shape (batch, 1, channels, samples)
            # first a temporal convolution with kernel size (1, kernel_size) ignoring the spatial dimension
            nn.Conv2d(in_channels=1,
                      out_channels=f1,
                      kernel_size=(1, kernel_size),
                      padding="same",
                      bias=False),
            # shape (batch, f1, channels, samples)
            nn.BatchNorm2d(num_features=f1),
            # shape (batch, f1, channels, samples)
            # next a depthwise convolution over the spatial dimension to learn frequency specific spatial filters
            DepthwiseConv2d(in_channels=f1,
                            depth_multiplier=d,
                            kernel_size=(channels, 1),
                            padding="valid",
                            bias=False),
            # shape (batch, f1 * d, 1, samples)
            nn.BatchNorm2d(num_features=f1 * d),
            # shape (batch, f1 * d, 1, samples)
            nn.ELU(),
            # shape (batch, f1 * d, 1, samples)
            # the pooling is done over the temporal dimension
            nn.AvgPool2d(kernel_size=(1, self.pool1)),
            # shape (batch, f1 * d, 1, samples // 4)
            nn.Dropout(p=dropout),
            # shape (batch, f1 * d, 1, samples // 4)
        )

        # in previous versions the final pooling was hard cored to 8
        # assuming a sample rate of 256 Hz
        self.pool2 = 8

        # but actually if we increase the sample length (more than one second of data)
        # we would want to pool more aggressively to reduce the output size to the same dimension
        # otherwise the fully connected layer will need to be very large
        # we can adjust the pooling size based on ratio of sample length and sample rate
        if adjust_for_sample_length:
            ratio = sample_length / sample_rate
            self.pool2 = int(self.pool2 * ratio)

        # Block 2
        self.block2 = nn.Sequential(
            # shape (batch, f1 * d, 1, samples // 4)
            # again a temporal convolution but this time as a depthwise separable convolution
            SeparableConv2d(in_channels=f1 * d,
                            out_channels=f2,
                            depth_multiplier=1,
                            kernel_size=(1, 16),
                            padding="same",
                            bias=False),
            # shape (batch, f2, samples / 4, 1)
            nn.BatchNorm2d(num_features=f2),
            # shape (batch, f2, samples / 4, 1)
            nn.ELU(),
            # shape (batch, f2, samples / 4, 1)
            # the pooling is done over the temporal dimension
            nn.AvgPool2d(kernel_size=(1, self.pool2)),
            # shape (batch, f2, samples / 32, 1)
            nn.Dropout(p=dropout),
            # shape (batch, f2, samples / 32, 1)
        )

        self.output_shape = (f2, sample_length // (self.pool1*self.pool2), 1)

    def forward(self, x: TensorType["num_batches", 1, "num_channels", "num_samples"]) -> TensorType["num_batches", "kernal_size", "reduced_channels", 1]:
        x = self.block1(x)
        x = self.block2(x)
        return x


@typechecked
class EEGNet(nn.Module):
    """
    Pytorch implementation of the EEGNet model from Lawhern et al. 2018.

    Reference Implementation of EEGNet Version 3 from original authors:
    https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
    """

    def __init__(self,
                 num_classes: int,
                 classifier_hidden_units: Optional[int] = None,
                 classifier_num_layers: int = 1,
                 **backbone_kwargs):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes to predict
        """

        super().__init__()

        self.classifier_hidden_units = classifier_hidden_units
        self.classifier_num_layers = classifier_num_layers
        self.classifier_num_hidden_layers = classifier_num_layers - 1
        self.num_classes = num_classes
        assert self.classifier_num_hidden_layers == 0 or self.classifier_hidden_units is not None, "If classifier_num_layers > 1, classifier_hidden_units must be specified"
        """
        conv_net = ...
        flatten      = Flatten(name = 'flatten')(conv_net)

        dense        = Dense(nb_classes, name = 'dense',
                            kernel_constraint = max_norm(norm_rate))(flatten)
        softmax      = Activation('softmax', name = 'softmax')(dense)
        return Model(inputs=input1, outputs=softmax)
        """
        # for backwards compatibility with previous versions of this EEGNet implementation we need to use nn.Sequential
        # otherwise we couldn't use the pretrained weights / re-evaulate older models
        # self.conv_net = EEGNetBackbone(**backbone_kwargs)
        backbone = EEGNetBackbone(**backbone_kwargs)
        self.dropout = backbone.dropout

        self.conv_net = nn.Sequential(*[
            backbone.block1,
            backbone.block2,
        ])
        # output shape of conv_net is (batch, f2, samples / 32, 1)
        classifier_input_units = np.array(backbone.output_shape).prod()
        input_units = [classifier_input_units] + [classifier_hidden_units] * self.classifier_num_hidden_layers
        output_units = [classifier_hidden_units] * self.classifier_num_hidden_layers + [num_classes]

        classifier_layers = [nn.Flatten()]
        for in_features, out_features in zip(input_units, output_units):
            classifier_layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            classifier_layers.append(nn.ELU())

        # remove the last activation layer
        classifier_layers = classifier_layers[:-1]

        self.classifier_head = nn.Sequential(*classifier_layers)
        # output shape of classifier_head is (batch, num_classes)

    def forward(self, x: TensorType["num_batches", 1, "num_channels", "num_samples"]) -> TensorType["num_batches", "num_classes"]:
        x = self.conv_net(x)
        x = self.classifier_head(x)
        return x

    def embed(self, x: TensorType["num_batches", 1, "num_channels", "num_samples"]) -> TensorType["num_batches", "classifier_input_units"]:
        x = self.conv_net(x)
        x = x.view(x.shape[0], -1)
        return x


class ConditionalEEGNet(nn.Module):
    """
    A model based on EEGNet but with the ability to condition it's prediction for any given sample on samples from a calibration sequence 

    The architecture changes as follows:
    - the input is now a tuple (x_calib, x_sample) where the shapes are
        - x_calib: (num_batches, num_calib_samples, 1, num_channels, num_samples)
        - x_sample: (num_batches, 1, num_channels, num_samples)
        Note: for technical reasons the input to the forward method is a single tensor with shape (num_batches, num_sequences, 1, num_channels, num_samples)
        where num_sequences = num_calib_samples + 1
    - both x_calib and x_sample are passed through the EEGNet backbone
    - the outputs for x_calib are aggregated (various aggregation methods are possible)
    - the output for x_sample is then aggregated with the x_calib_aggregated (again various aggregation methods are possible)
    - the aggregated output is passed through a classifier head
    """

    def __init__(self,
                 num_classes: int,
                 calibration_aggregation_method: str,
                 pre_classifier_aggregation_method: str,
                 classifier_hidden_units: Optional[int] = None,
                 classifier_num_layers: int = 1,
                 **backbone_kwargs):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes to predict
        calibration_aggregation_method : str
            The method used to aggregate the outputs of the EEGNet backbone for the calibration sequence
        pre_classifier_aggregation_method : str
            The method used to aggregate the outputs of the EEGNet backbone for the sample and the aggregated calibration sequence
        """

        super().__init__()

        self.num_classes = num_classes
        self.calibration_aggregation_method = calibration_aggregation_method
        self.pre_classifier_aggregation_method = pre_classifier_aggregation_method

        self.classifier_hidden_units = classifier_hidden_units
        self.classifier_num_layers = classifier_num_layers
        self.classifier_num_hidden_layers = classifier_num_layers - 1
        self.num_classes = num_classes
        assert self.classifier_num_hidden_layers == 0 or self.classifier_hidden_units is not None, "If classifier_num_layers > 1, classifier_hidden_units must be specified"

        self.conv_net = EEGNetBackbone(**backbone_kwargs)
        # output shape is (batch, f2, samples / 32, 1)
        conv_net_output_shape = self.conv_net.output_shape

        # two times the backbone output shape
        # because we concatenate the aggregated calibration sequence with the sample's output
        # classifier input shape might differ depending on pre_classifier_aggregation_method
        if pre_classifier_aggregation_method == "concat":
            classifier_input_shape = (2, *conv_net_output_shape)
        elif pre_classifier_aggregation_method in ["mean", "max", "min"]:
            classifier_input_shape = (1, *conv_net_output_shape)
        elif "difference" in pre_classifier_aggregation_method:
            classifier_input_shape = (1, *conv_net_output_shape)
        else:
            raise ValueError(f"Unknown pre_classifier_aggregation_method: {pre_classifier_aggregation_method}")

        classifier_input_units = np.array(classifier_input_shape).prod()
        input_units = [classifier_input_units] + [classifier_hidden_units] * self.classifier_num_hidden_layers
        output_units = [classifier_hidden_units] * self.classifier_num_hidden_layers + [num_classes]

        classifier_layers = [nn.Flatten(start_dim=-4)]
        for in_features, out_features in zip(input_units, output_units):
            classifier_layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            classifier_layers.append(nn.ELU())

        # remove the last activation layer
        classifier_layers = classifier_layers[:-1]

        self.classifier_head = nn.Sequential(*classifier_layers)
        # output shape of classifier_head is (batch, num_classes)

    def forward(self, x: TensorType["num_batches", "num_sequences", 1, "num_channels", "num_samples"]) -> TensorType["num_batches", "num_classes"]:
        # first pass all samples through the conv_net backbone
        # the conv_net backbone operates on 4D (batched) tensors so we need to flatten the first two dimensions and then reshape the output back to the original shape
        num_batches, num_sequences = x.shape[:2]
        x = x.view(num_batches * num_sequences, *x.shape[2:])
        x = self.conv_net(x)
        x = x.view(num_batches, num_sequences, *x.shape[1:])

        # split the input tensor into calibration and sample tensors
        x_calib, x_sample = x[:, :-1], x[:, -1:]

        # aggregate the outputs of the conv_net backbone for the calibration sequence
        if self.calibration_aggregation_method == "mean":
            x_calib = x_calib.mean(dim=1).unsqueeze(1)
        elif self.calibration_aggregation_method == "max":
            x_calib = x_calib.max(dim=1)[0].unsqueeze(1)
        elif self.calibration_aggregation_method == "min":
            x_calib = x_calib.min(dim=1)[0].unsqueeze(1)
        elif self.calibration_aggregation_method == "none":
            # x_calib stays the same
            # but for this case we need to expand the dimension of x_sample instead so that the concatenation below works
            x_sample = x_sample.expand(x_calib.shape).unsqueeze(1)
            x_calib = x_calib.unsqueeze(1)
        else:
            raise ValueError(f"Unknown calibration_aggregation_method: {self.calibration_aggregation_method}")

        # aggregate the outputs of the conv_net backbone for the sample and the aggregated calibration sequence
        x_pre_classifier = torch.cat([x_calib, x_sample], dim=1)
        assert x_pre_classifier.shape[
            1] == 2, f"Expected x_pre_classifier to have shape (batch, 2, ...), but got {x_pre_classifier.shape}"

        # since we operate over axis=1 where we will always have 2 elements, we can also use other aggregation methods here
        if self.pre_classifier_aggregation_method == "mean":
            x_pre_classifier = x_pre_classifier.mean(dim=1).unsqueeze(1)
        elif self.pre_classifier_aggregation_method == "max":
            x_pre_classifier = x_pre_classifier.max(dim=1)[0].unsqueeze(1)
        elif self.pre_classifier_aggregation_method == "min":
            x_pre_classifier = x_pre_classifier.min(dim=1)[0].unsqueeze(1)
        elif self.pre_classifier_aggregation_method == "difference":
            x_pre_classifier = (x_pre_classifier[:, 0] - x_pre_classifier[:, 1]).unsqueeze(1)
        elif self.pre_classifier_aggregation_method == "abs_difference":
            x_pre_classifier = torch.abs(x_pre_classifier[:, 0] - x_pre_classifier[:, 1]).unsqueeze(1)
        elif self.pre_classifier_aggregation_method == "square_difference":
            x_pre_classifier = torch.square(x_pre_classifier[:, 0] - x_pre_classifier[:, 1]).unsqueeze(1)
        elif self.pre_classifier_aggregation_method == "concat":
            x_pre_classifier = x_pre_classifier
        # even various distance metrics are possible here
        elif self.pre_classifier_aggregation_method == "cosine_similarity":
            x_pre_classifier = F.cosine_similarity(x_pre_classifier[:, 0], x_pre_classifier[:, 1])
        elif self.pre_classifier_aggregation_method == "euclidean_distance":
            x_pre_classifier = F.pairwise_distance(x_pre_classifier[:, 0], x_pre_classifier[:, 1])
        else:
            raise ValueError(f"Unknown pre_classifier_aggregation_method: {self.pre_classifier_aggregation_method}")

        # TODO: This might be 6 dimensional in the case of "none" for the calibration_aggregation_method
        assert len(x_pre_classifier.shape) >= 5, \
            f"Expected x_pre_classifier to be at least 5-dimensional, but got {x_pre_classifier.shape}"

        if len(x_pre_classifier.shape) == 6:
            # switch axis 1 and 2, since the last 4 dimensions are operated on by the classifier head
            x_pre_classifier = x_pre_classifier.permute(0, 2, 1, 3, 4, 5)
        # pass the aggregated output through the classifier head
        x = self.classifier_head(x_pre_classifier)

        if len(x.shape) == 3:
            # average over the second dimension
            x = x.mean(dim=1)

        return x


# pytorch lightning module of EEGNet
@typechecked
class EEGNetLightning(pl.LightningModule):

    metric_name_replacements = {
        "MulticlassAccuracy": "acc",
        "MulticlassPrecision": "prec",
        "MulticlassRecall": "rec",
        "MulticlassF1Score": "f1",
        "MulticlassROC": "roc",
        "MulticlassAUC": "auc",
        "MulticlassAUROC": "auroc",
        "BinaryAccuracy": "acc",
        "BinaryPrecision": "prec",
        "BinaryRecall": "rec",
        "BinaryF1Score": "f1",
        "BinaryROC": "roc",
        "BinaryAUC": "auc",
        "BinaryAUROC": "auroc",
    }

    @staticmethod
    def shorten_metric_name(name: str) -> str:
        for k, v in EEGNetLightning.metric_name_replacements.items():
            name = name.replace(k, v)
        return name

    @staticmethod
    def shorten_metric_names(metrics: Dict[str, Any]) -> Dict[str, float]:
        return {
            EEGNetLightning.shorten_metric_name(k): (v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in metrics.items()
        }

    def __init__(self, **hparams):
        super().__init__()
        self.use_class_weights = hparams.get("use_class_weights", False)
        class_weights = hparams.pop("class_weights", None)
        self.class_weights = class_weights if self.use_class_weights else None

        self.save_hyperparameters(hparams, ignore=["class_weights"])
        self.eegnet = EEGNet(**hparams)

        # check that learning_rate and optimizer_class exists
        assert hasattr(self.hparams, "learning_rate"), "learning_rate must be specified in hparams"
        assert hasattr(self.hparams, "optimizer_class"), "optimizer_class must be specified in hparams"

        #### metrics ####
        if self.hparams.num_classes == 2:
            metric_params = dict(task="binary")
        else:
            metric_params = dict(num_classes=self.hparams.num_classes, task="multiclass", average='macro')
        metrics = MetricCollection([
            Accuracy(**metric_params),
            Precision(**metric_params),
            Recall(**metric_params),
            F1Score(**metric_params),
            AUROC(**metric_params),
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, x: TensorType["num_batches", 1, "num_samples", "num_channels"]) -> TensorType["num_batches", "num_classes"]:
        x = x.squeeze(1)
        # x is of shape (batch, samples, channels) but we need (batch, 1, channels, samples)
        x = torch.swapaxes(x, -2, -1)
        x = x.unsqueeze(-3)
        y_hat = self.eegnet(x)
        return y_hat

    def embed(self, x: TensorType["num_batches", 1, "num_samples", "num_channels"]) -> TensorType["num_batches", "num_embedding_dims"]:
        x = x.squeeze(1)
        # x is of shape (batch, samples, channels) but we need (batch, 1, channels, samples)
        x = torch.swapaxes(x, -2, -1)
        x = x.unsqueeze(-3)
        x = self.eegnet.embed(x)
        return x

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch[:2]
        batch_size = x.shape[0]
        y_hat = self(x)
        if self.class_weights is not None and not isinstance(self.class_weights, torch.Tensor):
            self.class_weights = torch.tensor(self.class_weights, device=y_hat.device, dtype=torch.float)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        self.log("train_loss", loss, batch_size=batch_size)
        # calculate metrics
        preds = y_hat.argmax(-1) if self.hparams.num_classes == 2 else y_hat
        y_metrics = y.clone()
        y_metrics[y!=0] = 1
        output = self.train_metrics(preds, y_metrics)
        # shorten the names of the metrics
        output = {
            EEGNetLightning.shorten_metric_name(k): v
            for k, v in output.items()
        }
        # calculate balance of classes
        # convert classes to onehot
        y_onehot = F.one_hot(y, num_classes=self.hparams.num_classes)
        # calculate class balance
        class_balance = y_onehot.sum(dim=0) / len(y_onehot)
        # log class balance
        for i, b in enumerate(class_balance):
            self.log(f"train_class_balance_{i}", b, on_step=False, on_epoch=True, prog_bar=False)

        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch[:2]
        batch_size = x.shape[0]
        y_hat = self(x)
        if self.class_weights is not None and not isinstance(self.class_weights, torch.Tensor):
            self.class_weights = torch.tensor(self.class_weights, device=y_hat.device, dtype=torch.float)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        self.log("val_loss", loss, batch_size=batch_size)
        # calculate metrics
        preds = y_hat.argmax(-1) if self.hparams.num_classes == 2 else y_hat
        y_metrics = y.clone()
        y_metrics[y!=0] = 1
        output = self.val_metrics(preds, y_metrics)
        # shorten the names of the metrics
        output = {
            EEGNetLightning.shorten_metric_name(k): v
            for k, v in output.items()
        }

        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch[:2]
        batch_size = x.shape[0]
        y_hat = self(x)
        if self.class_weights is not None and not isinstance(self.class_weights, torch.Tensor):
            self.class_weights = torch.tensor(self.class_weights, device=y_hat.device, dtype=torch.float)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        self.log("test_loss", loss, batch_size=batch_size)
        # calculate metrics
        preds = y_hat.argmax(-1) if self.hparams.num_classes == 2 else y_hat
        y_metrics = y.clone()
        y_metrics[y!=0] = 1
        output = self.test_metrics(preds, y_metrics)
        # shorten the names of the metrics
        output = {
            EEGNetLightning.shorten_metric_name(k): v
            for k, v in output.items()
        }

        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch[:2]
        y_hat = self(x)
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
        return optimizer


class ConditionalEEGNetLightning(EEGNetLightning):
    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.eegnet = ConditionalEEGNet(**hparams)

    def forward(self, x: TensorType["num_batches", "sequence_length", "num_samples", "num_channels"]) -> TensorType["num_batches", "num_classes"]:
        # x is of shape (..., samples, channels) but we need (..., 1, channels, samples)
        x = torch.swapaxes(x, -2, -1)
        x = x.unsqueeze(-3)
        y_hat = self.eegnet(x)
        return y_hat
