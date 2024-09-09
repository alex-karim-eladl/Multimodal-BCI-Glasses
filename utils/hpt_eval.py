from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
import warnings

import wandb


from data import create_data_loaders
from models import create_model, create_trainer
from models.EEGNet import EEGNetLightning
from models.DANN import DANNLightning
from utils import segment_mask
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC

from tqdm import tqdm

import matplotlib.pyplot as plt


def load_model(config, artifact_dir, model_summary_input_size, train_dataloader):

    model_config = config["model"]
    optimizer_config = config["optimizer"]
    # for backwards compatibility fallback to sample_rate
    sample_length = config["sample_length"] if "sample_length" in config else config["sample_rate"]
    num_channels = config["num_channels"]
    batch_size = config["data"]["batch_size"]

    model = create_model(model_config=model_config,
                         optimizer_config=optimizer_config,
                         batch_size=batch_size,
                         sample_length=sample_length,
                         model_summary_input_size=model_summary_input_size,
                         num_channels=num_channels,
                         num_classes=train_dataloader.dataset.n_classes,
                         num_classes_discriminator=train_dataloader.dataset.n_classes_discriminator,
                         class_weights=train_dataloader.dataset.get_class_weights())

    
    config["num_classes"] = train_dataloader.dataset.n_classes
    config["num_classes_discriminator"] = train_dataloader.dataset.n_classes_discriminator
    
    model_file_path = Path(artifact_dir) / "model.pth"
    print(f"Loading model from {model_file_path}")
    model.load_state_dict(torch.load(model_file_path))

    return model


def load_data(config):
    data_config = config["data"]
    seed = config["seed"]
    pl.seed_everything(seed)
    data_config["shuffle_train"] = False
    train_dataloader, val_dataloader, test_dataloader, sampling_rate = create_data_loaders(data_config=data_config,
                                                                                           num_workers=0,
                                                                                           return_meta=True,
                                                                                           skip_test_dataset=False,
                                                                                           return_sampling_rate=True)
    return train_dataloader, val_dataloader, test_dataloader, sampling_rate


def load_data_and_model(project_name: str, run_id: str):
    api = wandb.Api()
    run = api.run(f"{project_name}/{run_id}")
    artifact = run.logged_artifacts()[0]
    artifact_dir = artifact.download()
    config = artifact.metadata['config']

    trainer_config = config["trainer"]
    # create trainer
    device = "gpu"
    device_id = 1
    # check if gpu is available
    if not torch.cuda.is_available():
        device = "cpu"

    trainer = create_trainer(trainer_config=trainer_config,
                             device=device,
                             device_id=device_id,
                             use_wandb_logger=False)

    train_dataloader, val_dataloader, test_dataloader, sample_rate = load_data(config)
    batch_size, sequence_length, sample_length, num_channels = next(iter(train_dataloader))[0].shape
    if config["model"]["name"] == "EEGNet":
        model_summary_input_size = (1, num_channels, sample_length)
    elif config["model"]["name"] == "ConditionalEEGNet":
        model_summary_input_size = (sequence_length, 1, num_channels, sample_length)
    elif config["model"]["name"] == "DANN":
        model_summary_input_size = (1, num_channels, sample_length)
    else:
        model_summary_input_size = None

    model = load_model(config, artifact_dir, model_summary_input_size, train_dataloader)
    if "sample_rate" not in config:
        config["sample_rate"] = sample_rate
    return trainer, model, train_dataloader, val_dataloader, test_dataloader, config, run


def gather_dataset(dataloader):
    dataset = {}
    for batch in dataloader:
        for i, batch_element in enumerate(batch):
            if i not in dataset:
                dataset[i] = []
            dataset[i].append(batch_element.cpu().numpy() if isinstance(batch_element, torch.Tensor) else batch_element)

    # np.concatenate everything
    for key in dataset:
        dataset[key] = np.concatenate(dataset[key])

    key_mapping = {
        0: "data",
        1: "label",
        2: "recording_ids",
        3: "recording_order",
        4: "recording_names",
    }
    # replace the keys with the correct names
    dataset = {key_mapping[key]: dataset[key] for key in dataset}

    # DANN has labels for the discriminators as well, but we only want to keep the labels for the task classifier
    # check if label has shape (..., 2) and if so, only keep the first column
    if len(dataset["label"].shape) > 1:
        for i in range(1, dataset["label"].shape[-1]):
            dataset[f"label_discriminator_{i-1}"] = dataset["label"][:, 0]
        dataset["label"] = dataset["label"][:, 0]

    return dataset


def get_dataset_and_predictions(train_dataloader, val_dataloader, test_dataloader, trainer, model, run, return_embeddings=False):
    train_dataset = gather_dataset(train_dataloader)
    val_dataset = gather_dataset(val_dataloader)
    test_dataset = gather_dataset(test_dataloader) if test_dataloader is not None else {}

    # validate data and model loading worked correctly by rerunning eval on validation set
    test_results = trainer.test(model, val_dataloader)
    if run is not None:
        expected_loss = run.summary["test_loss"] if "test_loss" in run.summary else run.summary["val_loss"]
        expected_acc = run.summary["test_acc"] if "test_acc" in run.summary else run.summary["val_acc"]
        if not np.isclose(test_results[0]["test_loss"], expected_loss, atol=1e-3):
            warnings.warn(f"Expected val loss: {expected_loss}, but val loss: {test_results[0]['test_loss']}")
        if not np.isclose(test_results[0]["test_acc"], expected_acc, atol=1e-3):
            warnings.warn(f"Expected val accuracy: {expected_acc}, but val accuracy: {test_results[0]['test_acc']}")

    train_prediction_results, val_prediction_results, test_prediction_results = predict(
        trainer, model, train_dataloader, val_dataloader, test_dataloader, run, return_embeddings=return_embeddings)

    train_dataset.update(
        train_prediction_results
    )
    val_dataset.update(
        val_prediction_results
    )
    test_dataset.update(
        test_prediction_results
    )

    dataset = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }

    if test_dataset["logits"] is None:
        del dataset["test"]

    return dataset


def is_inter_recording_split(dataset):
    # check whether the dataset split was an inter or intra recording split
    unique_recording_ids = np.unique(dataset["recording_ids"])
    # checking whether this is an inter or intra recording split is tricky
    # if the train and val set have the same number of unique recording ids, then it is most likely an intra recording split,
    # but it could also be an inter recording split with a 50/50 split of recordings
    # so instead we are going to look at the recording_order and see if it contains as many conitguous recordings as there are unique recording ids

    # get the number of contiguous recordings in the train and val set
    # we do this by checking if the difference between the recording_order is 1 (i.e within each recording the recording_order is 0, 1, 2, 3, ..., n_i, 0, 1, 2, 3, ..., n_j)
    # if it is not 1, then we have a new recording or a gap in the recording_order.
    # A gap in the recording_order would indicate that the recording_order is not continuous which would indicate that the split is an intra recording
    train_num_contiguous_recording_ids = (np.diff(dataset["recording_order"][:, -1].flatten()) != 1).sum() + 1

    is_inter_recording_split = train_num_contiguous_recording_ids == len(unique_recording_ids)
    return is_inter_recording_split


def prepare_run_for_evaluation(project_name: str, run_id: str, return_embeddings: bool = False):
    trainer, model, train_dataloader, val_dataloader, test_dataloader, config, run = load_data_and_model(project_name, run_id)

    dataset = get_dataset_and_predictions(train_dataloader, val_dataloader, test_dataloader,
                                          trainer, model, run, return_embeddings=return_embeddings)

    return dataset, run, config


def get_embeddings(model, dataloader):
    embeddings = []
    for batch in tqdm(dataloader):
        model.eval()
        with torch.no_grad():
            # get the device the model is on
            device = next(model.parameters()).device
            # move the tensors in the batch to the device the model is on (only torch tensors can be moved to a device)
            batch = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch]
            # get the embeddings
            embedding = model.embed(batch[0])
            embeddings.append(embedding.cpu().numpy())
    return np.concatenate(embeddings)


def predict(trainer, model, train_dataloader, val_dataloader, test_dataloader, run, return_embeddings=False):
    # now get the predictions for the train and val set and append them to the dataset dicts
    logits_val = trainer.predict(model, val_dataloader)

    num_classes = train_dataloader.dataset.n_classes
    if num_classes == 2:
        metric_params = dict(task="binary")
    else:
        metric_params = dict(num_classes=num_classes, task="multiclass", average='macro')
    acc_metric = Accuracy(**metric_params)

    for batch, l in zip(val_dataloader, logits_val):
        acc_metric(
            l.argmax(-1),
            batch[1][:, 0] if isinstance(model, DANNLightning) else batch[1]
        )
    val_accuracy = acc_metric.compute()

    expected_val_acc = run.summary["test_acc"] if "test_acc" in run.summary else run.summary["val_acc"]

    print(f"Expected val accuracy: {expected_val_acc}, actual val accuracy: {val_accuracy}") if run is not None else None
    # assert np.isclose(val_accuracy, run.summary["val_acc"], atol=1e-4)

    logits_train = trainer.predict(model, train_dataloader)
    acc_metric = Accuracy(**metric_params)
    # we need to calculate the weighted average of the accuracys (weighted by batch size)
    for batch, l in zip(train_dataloader, logits_train):
        acc_metric(
            l.argmax(-1),
            batch[1][:, 0] if isinstance(model, DANNLightning) else batch[1]
        )
    train_accuracy = acc_metric.compute()
    print(
        f"Expected train accuracy: {run.summary['train_acc']}, actual train accuracy: {train_accuracy}") if run is not None else None
    # assert np.isclose(train_accuracy, run.summary["train_acc"], atol=2e-2)

    
    logits_train = torch.cat(logits_train)
    logits_val = torch.cat(logits_val)

    probs_train = torch.nn.functional.softmax(logits_train, dim=1)
    preds_train = torch.argmax(probs_train, dim=1)

    probs_val = torch.nn.functional.softmax(logits_val, dim=1)
    preds_val = torch.argmax(probs_val, dim=1)


    if test_dataloader is not None:
        logits_test = trainer.predict(model, test_dataloader)
        acc_metric = Accuracy(**metric_params)
        # we need to calculate the weighted average of the accuracys (weighted by batch size)
        for batch, l in zip(test_dataloader, logits_test):
            acc_metric(
                l.argmax(-1),
                batch[1][:, 0] if isinstance(model, DANNLightning) else batch[1]
            )
        test_accuracy = acc_metric.compute()
        print(f"Test accuracy: {test_accuracy}")
        
        logits_test = torch.cat(logits_test)
        probs_test = torch.nn.functional.softmax(logits_test, dim=1)
        preds_test = torch.argmax(probs_test, dim=1)
    else:
        logits_test, probs_test, preds_test = None, None, None


    train_res, val_res, test_res = (
        {
            "logits": logits_train.cpu().numpy(),
            "probs": probs_train.cpu().numpy(),
            "preds": preds_train.cpu().numpy()
        },
        {
            "logits": logits_val.cpu().numpy(),
            "probs": probs_val.cpu().numpy(),
            "preds": preds_val.cpu().numpy()
        },
        {
            "logits": logits_test.cpu().numpy() if logits_test is not None else None,
            "probs": probs_test.cpu().numpy() if probs_test is not None else None,
            "preds": preds_test.cpu().numpy() if preds_test is not None else None
        }
    )

    if return_embeddings:
        train_embeddings = get_embeddings(model, train_dataloader)
        val_embeddings = get_embeddings(model, val_dataloader)
        test_embeddings = get_embeddings(model, test_dataloader)
        train_res["embeddings"] = train_embeddings
        val_res["embeddings"] = val_embeddings
        test_res["embeddings"] = test_embeddings

    return train_res, val_res, test_res


# %%


def evaluate_recording(recording_id: int, dataset: dict, config: dict):
    data_config = config["data"]
    sample_rate = config["sample_rate"]
    num_classes = config["num_classes"]
    # sample_length is a new parameter, for backwards compatibility fallback to sample_rate
    window_size = config["sample_length"] if "sample_length" in config else sample_rate

    subject = data_config["dataset_filters"]["subject"]
    if isinstance(subject, list):
        assert len(subject) == 1
        subject = subject[0]
    print(f"Subject: {subject}")

    orig_data_recording_ids = dataset["recording_ids"][:, -1]
    recording_id_mask = orig_data_recording_ids == recording_id

    recording = {
        k: v[recording_id_mask] for k, v in dataset.items()
    }

    # now sort all the recording data by the recording order
    sort_idx = recording["recording_order"][:, -1].argsort()
    recording = {
        k: v[sort_idx] for k, v in recording.items()
    }

    # the data from the dataloader had been windowed, we need to undo this and reconstruct the original data
    # default behavior of the stride parameter was changed such that stride==None means stride_samples = sample_rate
    # this is equivalent to data_config["window_stride"] = sample_rate / window_size
    if data_config["window_stride"] is None:
        data_config["window_stride"] = sample_rate / window_size
    #int(round(stride * self.window_size)) if stride is not None else sampling_rate
    window_stride_samples = int(round(data_config["window_stride"] * window_size))

    # reconstructing the original data positions since we know the recording order and the window stride
    # recording order gives us the position of the window in the recording
    # multiplying this with the window stride gives us the position of the first sample in the window in the recording
    # we can then add the sample ids of the window to get the position of all samples in the window
    window_sample_ids = np.arange(0, window_stride_samples)
    original_data_positions = (((recording["recording_order"][:, -1][:, None]) *
                               window_stride_samples) + window_sample_ids[None, :]).flatten()

    # make sure we didn't mess anything up
    assert original_data_positions.shape == np.unique(original_data_positions).shape

    # this assert statment only works when we have all samples of a single recording,
    # but not if we did an intra recording split (e.g. some of the samples of this recording are in another split)
    if is_inter_recording_split(dataset):
        assert original_data_positions.max() == len(original_data_positions) - 1

    # no we also need to reconstruct the original data
    # first we will select the correct windows (across the sequence axis). This should always be the last element of the sequence axis
    recording_data = recording["data"][:, -1]
    # then we select only the first window_stride_samples samples of each window (so we get rid of all overlapping samples)
    # and finally we reshape the data to the original shape
    original_data = recording_data[:, :window_stride_samples].reshape(-1, recording_data.shape[-1])
    assert original_data.shape[0] == original_data_positions.shape[0]

    # Additionally, we reduced the labels to one label for each window, we need to take this into account to.
    # We want the label time to be the time of the last sample in the window
    # again we calculate the position of the last sample in the window based on the recording order and the window stride
    label_time = ((recording["recording_order"][:, -1] + 1) * window_stride_samples) - 1

    recording["original_data"] = original_data
    recording["original_data_positions"] = original_data_positions
    recording["label_time"] = label_time

    # this also only applies for inter recording splits, but not for intra recording splits
    if is_inter_recording_split(dataset):
        assert (np.diff(label_time) == window_stride_samples).all()

    # next we take moving average over the probabilities to smooth them out
    # using the same window size and stride as the dataloader
    # the labels are already based on windows, so know we just need 1/stride number of labels to average over

    # pad the probabilities with zeros at the beginning to make sure we can take the moving average
    # and get the same number of labels as the original predictions
    num_windows_to_average = int(1/data_config["window_stride"])
    # probs_padded = np.pad(recording["probs"][:, 1], (num_windows_to_average-1, 0), mode="linear_ramp")
    probs_padded = []
    for i in range(num_classes):
        probs_padded.append(np.pad(recording["probs"][:, i], (num_windows_to_average-1, 0), mode="linear_ramp"))

    probs_padded = np.array(probs_padded).T
    # probs_averaged = np.lib.stride_tricks.sliding_window_view(
    #     recording["probs"][:, 1], int(1/data_config["window_stride"]), axis=0).mean(axis=1)
    probs_averaged = np.lib.stride_tricks.sliding_window_view(
        probs_padded, num_windows_to_average, axis=0).mean(axis=-1)
    pred_averaged = (probs_averaged >= 0.5).astype(float).argmax(axis=-1)

    # time_averaged = np.lib.stride_tricks.sliding_window_view(
    #     recording["label_time"], int(1/data_config["window_stride"]), axis=0)[:, -1]
    # label_averaged = (np.lib.stride_tricks.sliding_window_view(recording["label"], int(
    #     1/data_config["window_stride"]), axis=0).mean(axis=1) >= 0.5).astype(float)
    # label_averaged = np.lib.stride_tricks.sliding_window_view(recording["label"], int(
    #     1/data_config["window_stride"]), axis=0)[:, -1].astype(float)

    # since we padded the probabilities, we don't need to change the labels or time
    time_averaged = recording["label_time"]
    label_averaged = recording["label"]

    recording["probs_averaged"] = probs_averaged
    recording["pred_averaged"] = pred_averaged
    recording["time_averaged"] = time_averaged
    recording["label_averaged"] = label_averaged

    # calculate metrics for the recording
    if num_classes == 2:
        metric_params = dict(task="binary")
    else:

        metric_params = dict(num_classes=num_classes, task="multiclass", average="micro")

    metrics = MetricCollection(
        [
            Accuracy(**metric_params),
            Precision(**metric_params),
            Recall(**metric_params),
            F1Score(**metric_params),
        ] + ([AUROC(**metric_params)] if num_classes == 2 else [])  # because AUROC doesn't support micro averaging
    )
    metrics_smoothed = metrics.clone()

    preds = torch.from_numpy(recording["preds"]) if num_classes == 2 else torch.from_numpy(recording["probs"])
    metrics(preds, torch.from_numpy(recording["label"]))
    preds_smoothed = torch.from_numpy(recording["pred_averaged"]
                                      ) if num_classes == 2 else torch.from_numpy(recording["probs_averaged"])
    metrics_smoothed(preds_smoothed, torch.from_numpy(recording["label_averaged"]))

    improvements = {
        m_name: (metrics_smoothed[m_name].compute() - metrics[m_name].compute()).item()
        for m_name in metrics.keys()
    }
    # print(f"improvements through averaging predictions for recording {recording_id}\n", json.dumps(improvements, indent=4))
    # print(f"metrics (smoothed predictions) for recording {recording_id}\n",
    #       json.dumps({k: round(v.item(), 4) for k, v in metrics_smoothed.compute().items()}, indent=4))

    recording["metrics"] = {
        "original": EEGNetLightning.shorten_metric_names(dict(metrics.compute())),
        "smoothed": EEGNetLightning.shorten_metric_names(dict(metrics_smoothed.compute())),
        "improvements": EEGNetLightning.shorten_metric_names(improvements),
    }

    # calculate statistics over probabilities
    math_probs = recording["probs"][:, 1]

    # first in general
    prob_stats = numpy_statistics(math_probs)

    # then grouped by label
    prob_stats_by_label = {
        label: numpy_statistics(math_probs[recording["label"] == label])
        for label in np.unique(recording["label"])
    }

    # grouped by prediction
    prob_stats_by_pred = {
        pred: numpy_statistics(math_probs[recording["preds"] == pred])
        for pred in np.unique(recording["preds"])
    }

    # class margins are the difference between the probability of the positive class and the negative class
    # A margin of 0.0 means that the model is completely uncertain
    # A margin of 1.0 means that the model is completely certain of the positive class
    # A margin of -1.0 means that the model is completely certain of the negative class
    # In general the absolute value of the margin is the certainty of the model
    margins = recording["probs"][:, 1] - recording["probs"][:, 0]
    margin_stats = numpy_statistics(margins)
    margin_stats_by_label = {
        label: numpy_statistics(margins[recording["label"] == label])
        for label in np.unique(recording["label"])
    }
    margin_stats_by_pred = {
        pred: numpy_statistics(margins[recording["preds"] == pred])
        for pred in np.unique(recording["preds"])
    }

    recording["stats"] = {
        "probs": prob_stats,
        "probs_by_label": prob_stats_by_label,
        "probs_by_pred": prob_stats_by_pred,
        "margins": margin_stats,
        "margins_by_label": margin_stats_by_label,
        "margins_by_pred": margin_stats_by_pred,
        "preds": numpy_statistics(recording["preds"]),
    }

    # add some statistics about the recording data itself
    recording["stats"]["data"] = {
        "num_samples": len(original_data),
        "num_windows": len(recording["label"]),
        "duration": len(original_data) / sample_rate,
        "sampling_rate": sample_rate,
    }

    # and about the labels
    recording["stats"]["labels"] = {
        "num_occurrences": {
            label: (recording["label"] == label).sum()
            for label in np.unique(recording["label"])
        },
        # segments are contiguous chunks of the same label
        # math is labeled as 1, non-math as 0
        "num_math_segments": (np.diff(recording["label"]) == 1).sum(),
        # +1 because the first segment is always non-math
        "num_non_math_segments": (np.diff(recording["label"]) == -1).sum() + 1,
    }

    recording_name = np.unique(recording["recording_names"])
    if len(recording_name) == 1:
        recording["recording_name"] = str(recording_name[0])

    recording_label = np.unique(recording["label"])
    if len(recording_label) == 1:
        recording["recording_label"] = recording_label[0]

    del recording["recording_names"]
    return recording


def numpy_statistics(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": x.mean(),
        "std": x.std(),
        "var": x.var(),
        "min": x.min(),
        "max": x.max(),
        "median": np.median(x),
        "percentile_25": np.percentile(x, 25),
        "percentile_50": np.percentile(x, 50),
        "percentile_75": np.percentile(x, 75),
    }


def plot_recording_predictions(recording: dict, subject: str, recording_id: int, figure_dir: str | Path):
    # get recording name
    recording_name = recording["recording_name"] if "recording_name" in recording else f"{subject} recording #{recording_id}"
    print(f"plotting predictions for {recording_name}")
    fig, ax = plt.subplots(4, 1, figsize=(20, 10), sharex=True)
    fig.suptitle(f"{recording_name}", fontsize=16)

    ax[0].plot(recording["original_data_positions"], recording["original_data"])
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title("Data")

    ax[1].plot(recording["label_time"], recording["label"])
    ax[1].set_ylabel("Label")
    ax[1].set_title("Labels")

    ax[2].plot(recording["label_time"], recording["preds"], label="prediction")
    ax[2].plot(recording["label_time"], recording["probs"][:, 1], label="probability true")
    ax[2].set_ylabel("Label")
    ax[2].set_title("Predictions")
    ax[2].legend()

    ax[3].plot(recording["time_averaged"], recording["pred_averaged"], label="prediction averaged")
    ax[3].plot(recording["time_averaged"], recording["probs_averaged"], label="probability true averaged")
    ax[3].set_ylabel("Label")
    ax[3].set_title("Predictions averaged over overlapping-windows")

    # the plot should have rectangular boxes to indicate correctly and incorrectly predicted areas
    # we'll do this by calculating the difference between the prediction and the label

    correct_mask = (recording["pred_averaged"] == recording["label_averaged"])
    correct_mask_segments = segment_mask(correct_mask.astype(int))

    # recording["time_averaged"]

    # add the rectangles to the plot
    for (start, end) in correct_mask_segments:
        start_time = recording["time_averaged"][start]
        end_time = recording["time_averaged"][end]
        color = "green" if correct_mask[start] == 1 else "red"
        assert (np.diff(correct_mask[start:end])).sum() == 0, "incorrect mask segment"
        ax[1].axvspan(start_time, end_time, alpha=0.2, color=color)
        ax[3].axvspan(start_time, end_time, alpha=0.2, color=color)

    # add label legend
    ax[3].legend()

    # plot the validation/train split for the recording
    # do it in the same way as the correct/incorrect rectangles

    # val_mask = recording["val_mask"]
    # val_mask_segments = segment_mask(val_mask.astype(int))

    # # add the rectangles to the plot
    # for start, end in val_mask_segments:
    #     start_time = recording["label_time"][start]
    #     end_time = recording["label_time"][end]
    #     color = "green" if val_mask[start] == 1 else "red"
    #     assert (np.diff(val_mask[start:end])).sum() == 0, "incorrect mask segment"
    #     ax[4].axvspan(start_time, end_time, alpha=0.3, color=color)

    # ax[4].set_title("Split into train (red) and validation (green)")

    # save plot as png
    figure_dir.mkdir(parents=True, exist_ok=True)
    filename = figure_dir / f"recording_{recording_name}.png"
    # print(f"Saving figure to {filename}")
    fig.savefig(filename)
    plt.close(fig)
