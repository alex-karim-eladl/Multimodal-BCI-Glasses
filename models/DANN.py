from typing import Dict, Any, Tuple, Union, Optional, Callable, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import torch.optim as optim
import pytorch_lightning as pl
from torchsummary import summary
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC

from mind_ml.models.EEGNet import EEGNetBackbone, EEGNetLightning

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


class AlphaScheduler:
    def __init__(self, final_alpha, total_steps, mode="dann_original", temperature_scaling=25):
        """
        Parameters
        ----------
        final_alpha : float
            The final alpha value after the total number of steps
        total_steps : int
            The total number of steps to reach the final alpha value
        mode : str
            The mode of the scheduler. Supported are
            - linear: linearly increase alpha from 0 to final_alpha
            - dann_original: the original alpha scheduler used in the DANN paper
            - sigmoid: sigmoid function to increase alpha from 0 to final_alpha, with a temperature parameter to control the steepness
            - linear_warmup: linearly increase alpha from 0 to final_alpha over the first 33% of the total steps
        temperature_scaling : float
            The temperature parameter for the sigmoid scheduler
            This is relative to the total number of steps, so that the temperature is independent of the total number of steps

        """
        self.final_alpha = final_alpha
        self.total_steps = total_steps
        self.mode = mode
        self.temperature_scaling = temperature_scaling

    def step(self, current_step):
        # first calculates alpha in range of 0 to 1
        # then multiplies it with the final alpha value
        if self.mode == "dann_original":
            p = float(current_step) / self.total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
        elif self.mode == "linear":
            alpha = current_step / self.total_steps
        elif self.mode == "sigmoid":
            # make sure is shifted along the x axis to center around 0.5 * total_steps
            # include a temperature/scaling parameter to control the steepness of the sigmoid
            temperature = self.temperature_scaling / self.total_steps
            alpha = 1 / (1 + np.exp(-temperature * (current_step - 0.5 * self.total_steps)))
        elif self.mode == "linear_warmup":
            if current_step < 0.33 * self.total_steps:
                alpha = current_step / (0.33 * self.total_steps)
            else:
                alpha = 1
        else:
            raise ValueError(f"Unknown alpha scheduler mode {self.mode}")

        return self.final_alpha * alpha


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class MLPClassifier(nn.Module):
    """
    A simple MLP classifier.
    """

    def __init__(self,
                 num_classes: int,
                 input_units: int,
                 hidden_units: Optional[int] = None,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 ):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes to classify.
        input_units : int
            Number of input units to the classifier.
        hidden_units : Optional[int], optional
            Number of hidden units in the classifier, by default None
        num_layers : int, optional
            Number of layers in the classifier, by default 1
        dropout : float, optional
            Dropout rate in the classifier, by default 0.0

        Raises
        ------
        AssertionError
            If num_layers > 1 and hidden_units is not specified.
        """

        super().__init__()

        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.num_hidden_layers = num_layers - 1
        self.num_classes = num_classes
        assert self.num_hidden_layers == 0 or self.hidden_units is not None, "If num_layers > 1, hidden_units must be specified"

        input_units = [input_units] + [hidden_units] * self.num_hidden_layers
        output_units = [hidden_units] * self.num_hidden_layers + [num_classes]

        layers = []
        for in_features, out_features in zip(input_units, output_units):
            layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))

        # remove the last activation layer and dropout layer
        layers = layers[:-2]

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: TensorType["num_batches", "input_units"]) -> TensorType["num_batches", "num_classes"]:
        x = self.classifier(x)
        return x


@typechecked
class DANN(nn.Module):
    """
    A vaiartion of Domain Adversarial Neural Network (DANN) 
    """

    def __init__(self,
                 feature_extractor_kwargs: Dict[str, Any],
                 task_classifier_kwargs: Dict[str, Any],
                 discriminator_classifier_kwargs: Dict[str, Any],
                 **kwargs
                 ):
        """

        """

        super().__init__()

        self.feature_extractor = EEGNetBackbone(**feature_extractor_kwargs)
        self.feature_shape = self.feature_extractor.output_shape

        classifier_input_units = np.array(self.feature_shape).prod()

        task_classifier_kwargs["input_units"] = classifier_input_units
        self.task_classifier = MLPClassifier(**task_classifier_kwargs)

        # there may be more than one discriminator classifier
        # for backward compatibility, we check if there is only one discriminator classifier
        discriminator_classifier_kwargs["input_units"] = classifier_input_units
        if len(discriminator_classifier_kwargs["num_classes"]) == 1:
            self.multi_discriminator = False
            discriminator_classifier_kwargs["num_classes"] = discriminator_classifier_kwargs["num_classes"][0]
            self.discriminator_classifier = MLPClassifier(**discriminator_classifier_kwargs)
        else:
            self.multi_discriminator = True
            self.discriminator_classifier = nn.ModuleList()
            for num_classes in discriminator_classifier_kwargs["num_classes"]:
                # create a copy of the discriminator classifier kwargs
                discriminator_kwargs = discriminator_classifier_kwargs.copy()
                discriminator_kwargs["num_classes"] = num_classes
                self.discriminator_classifier.append(MLPClassifier(**discriminator_kwargs))

    def forward(self, x: TensorType["num_batches", 1, "num_channels", "num_samples"], alpha: float = 1.0) -> Tuple[TensorType["num_batches", "num_classes_task"],
                                                                                                                   List[torch.Tensor]]:
        feature = self.feature_extractor(x)

        # flatten the feature
        feature = feature.view(feature.size(0), -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)

        class_output = self.task_classifier(feature)
        if self.multi_discriminator:
            discriminator_output = [discriminator(reverse_feature) for discriminator in self.discriminator_classifier]
        else:
            discriminator_output = [self.discriminator_classifier(reverse_feature)]

        return class_output, discriminator_output

    def embed(self, x: TensorType["num_batches", 1, "num_channels", "num_samples"]) -> TensorType["num_batches", "classifier_input_units"]:
        feature = self.feature_extractor(x)

        # flatten the feature
        feature = feature.view(feature.size(0), -1)

        return feature

# pytorch lightning module of DANN


class DANNLightning(pl.LightningModule):

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
        self.class_weights = hparams.pop("class_weights", None)
        # reset to None only after popping it from hparams,
        # otherwise it will be saved as a hyperparameter (and throw an error since it's an array)
        if not self.use_class_weights:
            self.class_weights = None
        self.alpha_scheduler_kwargs = hparams.pop("alpha_scheduler_kwargs", {})
        self.save_hyperparameters(hparams, ignore=["class_weights"])
        self.eegnet = DANN(**hparams)

        assert hasattr(self.hparams, "learning_rate"), "learning_rate must be specified in hparams"
        assert hasattr(self.hparams, "optimizer_class"), "optimizer_class must be specified in hparams"
        assert hasattr(self.hparams, "final_alpha"), "final_alpha must be specified in hparams"

        # initialize alpha with default value, since it will only be first updated in training_step
        # but lightning will run one validation step before training starts for sanity checking
        self.alpha = 0.0

        # add metrics for task classifier
        self.task_num_classes = self.hparams.task_classifier_kwargs["num_classes"]
        if self.task_num_classes == 2:
            metric_params = dict(task="binary")
        else:
            metric_params = dict(num_classes=self.task_num_classes, task="multiclass", average='macro')

        metrics = MetricCollection([
            Accuracy(**metric_params),
            Precision(**metric_params),
            Recall(**metric_params),
            F1Score(**metric_params),
            AUROC(**metric_params),
        ])
        # if we collect metrics in a dict, pytorch lightning will *not* move them to the correct device....
        # so we need to keep them separate and as class attributes
        # self.metrics_task = {
        #     stage: metrics.clone(prefix=f"{stage}_")
        #     for stage in ["train", "val", "test"]
        # }
        self.metrics_task_train = metrics.clone(prefix="train_")
        self.metrics_task_val = metrics.clone(prefix="val_")
        self.metrics_task_test = metrics.clone(prefix="test_")

        # add metrics for discriminator classifier

        self.discriminator_num_classes = self.hparams.discriminator_classifier_kwargs["num_classes"]
        if not isinstance(self.discriminator_num_classes, list):
            self.discriminator_num_classes = [self.discriminator_num_classes]
        for discriminator_id, discriminator_num_classes in enumerate(self.discriminator_num_classes):
            if discriminator_num_classes == 2:
                metric_params = dict(task="binary")
            else:
                metric_params = dict(num_classes=discriminator_num_classes, task="multiclass", average='micro')

            metrics_discriminator = MetricCollection(
                [
                    Accuracy(**metric_params),
                    Precision(**metric_params),
                    Recall(**metric_params),
                    F1Score(**metric_params),
                ] + ([AUROC(**metric_params)] if metric_params["task"] == "binary" else [])  # because AUROC doesn't support micro averaging
            )
            postfix = f"_discriminator_{discriminator_id}"
            for stage in ["train", "val", "test"]:
                setattr(self, f"metrics_discriminator_{discriminator_id}_{stage}",
                        metrics_discriminator.clone(prefix=f"{stage}_", postfix=postfix))

    def _get_metrics(self, stage: str) -> Tuple[MetricCollection, List[MetricCollection]]:
        metrics_task = getattr(self, f"metrics_task_{stage}")
        metrics_discriminators = []
        for discriminator_id, _ in enumerate(self.discriminator_num_classes):
            metrics_discriminators.append(getattr(self, f"metrics_discriminator_{discriminator_id}_{stage}"))
        return metrics_task, metrics_discriminators

    def forward(self, x: TensorType["num_batches", 1, "num_samples", "num_channels"], alpha: float) -> Tuple[TensorType["num_batches", "num_classes_task"],
                                                                                                             TensorType["num_batches", "num_classes_discriminator"]]:
        x = x.squeeze(1)
        # x is of shape (batch, samples, channels) but we need (batch, 1, channels, samples)
        x = torch.swapaxes(x, -2, -1)
        x = x.unsqueeze(-3)
        y_hat = self.eegnet(x, alpha)
        return y_hat

    def embed(self, x: TensorType["num_batches", 1, "num_samples", "num_channels"]) -> TensorType["num_batches", "num_embedding_dims"]:
        x = x.squeeze(1)
        # x is of shape (batch, samples, channels) but we need (batch, 1, channels, samples)
        x = torch.swapaxes(x, -2, -1)
        x = x.unsqueeze(-3)
        x = self.eegnet.embed(x)
        return x

    def train_val_test_step(self, batch, batch_idx, stage: str) -> torch.Tensor:
        assert stage in ["train", "val", "test"], "stage must be one of train, val, test"
        x, targets = batch[:2]
        targets_task = targets[..., 0]
        targets_discriminator = []
        for discriminator_id in range(1, targets.shape[-1]):
            targets_discriminator.append(targets[..., discriminator_id])
        batch_size = x.shape[0]

        logits_task, logits_discriminator = self(x, self.alpha)

        # calculate task loss
        loss_task = F.cross_entropy(logits_task, targets_task)
        # logging this here because for some reason the variable is changed when computing the combined loss
        self.log(f"{stage}_loss", loss_task, batch_size=batch_size)
        # calculate discriminator loss
        # to make sure the task and discriminator losses are approximately of the same magnitude, we scale the discriminator losses
        # we do this by considering what the losses would be for completely random predictions
        # for cross entropy across C classes, the loss is log(C)
        # so we scale the discriminator loss by log(C_task) / log(C_discriminator)
        loss_discriminator = [
            (np.log(self.task_num_classes) / np.log(num_classes)) * F.cross_entropy(logits, targets)
            for logits, targets, num_classes in zip(logits_discriminator, targets_discriminator, self.discriminator_num_classes)
        ]

        # combine losses
        # loss = torch.tensor([loss_task] + loss_discriminator).mean()
        # the above formulation breaks gradient tracking, so we use the following instead
        loss = loss_task
        for loss_discriminator_i in loss_discriminator:
            loss += loss_discriminator_i
        loss /= (1 + len(loss_discriminator))
        self._step_metrics_logs(logits_task,
                                logits_discriminator,
                                loss,
                                loss_task,
                                loss_discriminator,
                                targets_task,
                                targets_discriminator,
                                batch_size, stage)
        return loss

    def _step_metrics_logs(self,
                           logits_task,
                           logits_discriminator,
                           loss,
                           loss_task,
                           loss_discriminator,
                           targets_task,
                           targets_discriminator,
                           batch_size, stage: str):
        # get metrics
        metrics_task, metrics_discriminator = self._get_metrics(stage)

        # calculate task metrics
        preds_task = logits_task.argmax(-1) if self.task_num_classes == 2 else logits_task
        output_task = metrics_task(preds_task, targets_task)
        # shorten the names of the metrics
        output_task = EEGNetLightning.shorten_metric_names(output_task)

        # calculate discriminator metrics
        output_discriminator = [
            metrics(logits.argmax(-1) if num_classes == 2 else logits, targets)
            for metrics, logits, targets, num_classes in zip(metrics_discriminator, logits_discriminator, targets_discriminator, self.discriminator_num_classes)
        ]
        # flatten list of dicts into one dict
        output_discriminator = {k: v for d in output_discriminator for k, v in d.items()}
        # shorten the names of the metrics
        output_discriminator = EEGNetLightning.shorten_metric_names(output_discriminator)
        output = {**output_task, **output_discriminator}

        # self.log(f"{stage}_loss", loss_task, batch_size=batch_size)
        self.log(f"{stage}_loss_combined", loss, batch_size=batch_size)
        # log discriminator losses
        for discriminator_id, discriminator_loss in enumerate(loss_discriminator):
            self.log(f"{stage}_loss_discriminator_{discriminator_id}", discriminator_loss, batch_size=batch_size)
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        # calculate balance of classes
        # convert classes to onehot and calculate class balance
        targets_task_onehot = F.one_hot(targets_task, num_classes=self.task_num_classes)
        targets_task_class_balance = targets_task_onehot.sum(dim=0) / len(targets_task_onehot)
        # log class balance
        for i, b in enumerate(targets_task_class_balance):
            self.log(f"{stage}_targets_task_class_balance_{i}", b, on_step=True, on_epoch=True, prog_bar=False)

        # same for discriminator classes
        for discriminator_id, num_classes in enumerate(self.discriminator_num_classes):
            targets_discriminator_onehot = F.one_hot(targets_discriminator[discriminator_id], num_classes=num_classes)
            targets_discriminator_class_balance = targets_discriminator_onehot.sum(dim=0) / len(targets_discriminator_onehot)
            for i, b in enumerate(targets_discriminator_class_balance):
                self.log(f"{stage}_targets_discriminator_class_balance_{discriminator_id}_{i}",
                         b, on_step=False, on_epoch=True, prog_bar=False)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        self.alpha = self.alpha_scheduler.step(self.trainer.global_step)
        self.log(f"alpha", self.alpha, on_step=True, on_epoch=True)
        return self.train_val_test_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self.train_val_test_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self.train_val_test_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        x, targets = batch[:2]

        logits_task, logits_discriminator = self(x, self.alpha)
        return logits_task

    def configure_optimizers(self):
        assert hasattr(torch.optim, self.hparams.optimizer_class),\
            f"{self.hparams.optimizer_class} is not a valid optimizer from torch.optim"
        optimizer_class = getattr(torch.optim, self.hparams.optimizer_class)
        optimizer = optimizer_class(
            [
                {  # feature_extractor layer parameters
                    "params": filter(lambda p: p.requires_grad, self.eegnet.feature_extractor.parameters()),
                    "weight_decay": getattr(self.hparams, "feature_extractor_weight_decay", 0.0),
                },
                {  # task_classifier layer parameters
                    "params": filter(lambda p: p.requires_grad, self.eegnet.task_classifier.parameters()),
                    "weight_decay": getattr(self.hparams, "task_classifier_weight_decay", 0.0),
                },
                {  # discriminator_classifier layer parameters
                    "params": filter(lambda p: p.requires_grad, self.eegnet.discriminator_classifier.parameters()),
                    "weight_decay": getattr(self.hparams, "discriminator_classifier_weight_decay", 0.0),
                },
            ],
            lr=self.hparams.learning_rate)

        assert hasattr(self.hparams, "final_alpha"), "final_alpha is not set"
        self.alpha_scheduler = AlphaScheduler(final_alpha=self.hparams.final_alpha,
                                              #total_steps=self.trainer.max_epochs * len(self.trainer.train_dataloader)
                                              total_steps=self.trainer.estimated_stepping_batches,
                                              **self.alpha_scheduler_kwargs
                                              )

        return optimizer
