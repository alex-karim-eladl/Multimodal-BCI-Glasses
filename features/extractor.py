"""
Author:			Alex Karim El Adl
Project:		Multimodal-BCI-Glasses
File:			/features/extractor.py

Description: custom feature extraction
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy
import scipy.signal
from tqdm import tqdm

from features.bands import band_power_from_psd, psd_estimator_welch

log = logging.getLogger(__name__)


_FEATURE_EXTRACTOR: Dict[str, Any] = {}  # registry
DEFAULT_FEATURES = []


def register_feature(default: bool = False):
    def decorator(fn):
        _FEATURE_EXTRACTOR[fn.__name__.lower()] = fn
        if default:
            DEFAULT_FEATURES.append(fn.__name__.lower())
        return fn

    return decorator


class Extractor:
    """Extract statistical features from raw data"""

    def __init__(
        self,
        x: np.ndarray,  # (num_frames, nsamples, num_channels)
        y_true: List[int],  # [0, 1]  {0: Rest, 1: Math}
        features: List[str] = DEFAULT_FEATURES,
    ):
        self.x = x
        self.y_true = y_true
        self.features = features

    def extract(self) -> Tuple[np.ndarray, List[str]]:
        results, feat_names = None, []
        pbar = tqdm(enumerate(zip(self.x, self.y_true)), total=len(self.y_true))
        for i, (frame, event) in pbar:
            pbar.set_description(f"Epoch({i})")
            try:
                ret, feat_names = self.calculate_features_from_frame(frame, event=event)
            except Exception as e:
                log.error(f"Error {e} at frame({i}) {frame}", e)
                continue

            if results is None:
                results = ret
            else:
                results = np.vstack([results, ret])
        return results, feat_names

    def calculate_features_from_frame(
        self, frame: np.ndarray, event: int
    ) -> Tuple[np.ndarray, List[str]]:
        """Calculate features for a single frame"""
        # Extract the half- and quarter-windows
        # nsamples = frame.shape[0]
        h1, h2 = np.split(frame, 2)
        q1, q2, q3, q4 = np.split(frame, 4)

        results = []
        feature_names_list = []
        covariance_matrix = np.cov(frame.T)
        for feature in self.features:
            feature_fn = get_feature_fn(feature)
            if "quarter" in feature:
                feature_result, feature_names = feature_fn(q1, q2, q3, q4)
            elif "half" in feature:
                feature_result, feature_names = feature_fn(h1, h2)
            elif "covariance" in feature:
                feature_result, feature_names, covariance_matrix = feature_fn(frame)
            elif "eigenvalues" in feature or "logcov" in feature:
                feature_result, feature_names = feature_fn(covariance_matrix)
            else:
                feature_result, feature_names = feature_fn(frame)
            results.extend(feature_result)
            feature_names_list.extend(feature_names)
        # Note: Remember to drop
        if event is not None:
            results.append(np.array([event]))
            feature_names_list.extend(["event"])

        return np.hstack(results), feature_names_list


def get_feature_fn(name: str):
    return _FEATURE_EXTRACTOR[name]


@register_feature(default=True)
def feature_mean(frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Compute the mean value of each signal for the frame"""
    channels_mean = np.nanmean(frame, axis=0).flatten()
    feature_names = [f"mean_ch_{i}" for i in range(channels_mean.shape[0])]
    return channels_mean, feature_names


@register_feature(default=True)
def feature_stddev(frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Computes the standard deviation of each signal for the full time window"""
    channels_stddev = np.std(frame, axis=0, ddof=1).flatten()
    feature_names = [f"stddev_ch_{i}" for i in range(channels_stddev.shape[0])]
    return channels_stddev, feature_names


@register_feature(default=True)
def feature_moments(frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Computes the 3rd and 4th standardised moments about the mean (i.e., skewness
    and kurtosis) of each signal, for the full time window."""

    skw = scipy.stats.skew(frame, axis=0, bias=False)
    krt = scipy.stats.kurtosis(frame, axis=0, bias=False)
    channels_moments = np.append(skw, krt)

    feature_names = [f"skew_{i}" + str(i) for i in range(frame.shape[1])]
    feature_names.extend([f"kurt_{i}" + str(i) for i in range(frame.shape[1])])
    return channels_moments, feature_names


@register_feature(default=True)
def feature_max(frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Returns the maximum value of each signal for the full time window"""

    channels_max = np.max(frame, axis=0).flatten()
    feature_names = [f"max_{i}" for i in range(channels_max.shape[0])]
    return channels_max, feature_names


@register_feature(default=True)
def feature_min(frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Returns the maximum value of each signal for the full time window"""

    channels_max = np.min(frame, axis=0).flatten()
    feature_names = [f"min_{i}" for i in range(channels_max.shape[0])]
    return channels_max, feature_names


@register_feature(default=True)
def feature_covariance_matrix(
    frame: np.ndarray,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Computes the elements of the covariance matrix of the signals."""

    covariance_matrix = np.cov(frame.T)
    indx = np.triu_indices(covariance_matrix.shape[0])
    result = covariance_matrix[indx]

    feature_names = []
    for i in np.arange(0, covariance_matrix.shape[1]):
        for j in np.arange(i, covariance_matrix.shape[1]):
            feature_names.extend([f"covariance_matrix_{i}_{j}"])

    return result, feature_names, covariance_matrix


@register_feature(default=True)
def feature_eigenvalues(covariance_matrix: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Computes the eigenvalues of the covariance matrix passed as the function
    argument."""

    eigenvalues = np.linalg.eigvals(covariance_matrix).flatten()
    feature_names = [f"eigenval_{i}" for i in range(covariance_matrix.shape[0])]
    return eigenvalues, feature_names


@register_feature(default=True)
def feature_logcov(covariance_matrix: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Computes the matrix logarithm of the covariance matrix of the signals."""
    log_cov = scipy.linalg.logm(covariance_matrix)
    indx = np.triu_indices(log_cov.shape[0])
    result = np.abs(log_cov[indx])

    feature_names = []
    for i in np.arange(0, log_cov.shape[1]):
        for j in np.arange(i, log_cov.shape[1]):
            feature_names.extend([f"logcovM_{i}_{j}"])

    return result, feature_names


@register_feature(default=True)
def feature_fft(
    frame: np.ndarray,
    period: float = 7.0,
    mains_f: float = 50.0,
    top_n: int = 10,
    filter_mains: bool = True,
    filter_DC: bool = False,
    normalize_signals: bool = False,
    get_power_spectrum: bool = False,
    resolution: int = 9,
) -> Tuple[np.ndarray, List[str]]:
    """Computes the FFT of each signal."""

    # Signal properties
    nsamples = frame.shape[0]
    ts = period / nsamples

    # Scale all signals to interval [-1, 1]
    if normalize_signals:
        frame = -1 + 2 * (frame - np.min(frame)) / (np.max(frame) - np.min(frame))

    # Compute the (absolute values of the) FFT
    freq, psd = psd_estimator_welch(frame, fs=1 / ts)

    # Remove DC component
    if filter_DC:
        psd = psd[1:]
        freq = freq[1:]

    # Remove mains frequency component(s)
    if filter_mains:
        indx = np.where(np.abs(freq - mains_f) <= 1)
        psd = np.delete(psd, indx, axis=0)
        freq = np.delete(freq, indx)

    # Extract top N frequencies for each signal
    indx = np.argsort(psd, axis=0)[::-1]
    indx = indx[:top_n]

    result = freq[indx].flatten(order="F")

    feature_names = []
    C = psd.shape[1]
    for i in np.arange(C):
        feature_names.extend([f"top_freq_{j}_ch_{i}" for j in np.arange(1, 11)])

    band_powers = band_power_from_psd(psd, freq)
    result = np.hstack([result, band_powers.flatten(order="F")])
    BW, _ = band_powers.shape
    for i in np.arange(C):
        feature_names.extend([f"band_{j}_ch_{i}" for j in np.arange(BW)])

    if get_power_spectrum:
        result = np.hstack([result, psd.flatten(order="F")])

        for i in np.arange(psd.shape[1]):
            feature_names.extend(
                [f"freq_{int(j):03d}_ch_{i}" for j in 10 * np.round(freq, 1)]
            )

    return result, feature_names


def compute_half_window_feature(
    h1: np.ndarray, h2: np.ndarray, feature_name: str, fn_name: str
) -> Tuple[np.ndarray, List[str]]:
    feature_fn = get_feature_fn(fn_name)
    result = (feature_fn(h2)[0] - feature_fn(h1)[0]).flatten()
    feature_names = [f"{feature_name}_h2h1_ch{i}" for i in range(result.shape[0])]
    return result, feature_names


def compute_quarter_window_feature(
    q1: np.ndarray,
    q2: np.ndarray,
    q3: np.ndarray,
    q4: np.ndarray,
    feature_name: List[str],
    fn_name: str,
) -> Tuple[np.ndarray, List[str]]:
    qs = (q1, q2, q3, q4)
    feature_fn = get_feature_fn(fn_name)
    feature_result = v1, v2, v3, v4 = [feature_fn(q)[0] for q in qs]
    diff_qs_result = [v1 - v2, v1 - v3, v1 - v4, v2 - v3, v2 - v4, v3 - v4]
    result = np.hstack([*feature_result, *diff_qs_result]).flatten()

    feature_names = []
    for ch in range(4):
        feature_names.extend(
            [f"{feature_name[0]}_q{q+1}_ch{ch+1}" for q in range(len(feature_result))]
        )

    for i in range(3):  # for quarter-windows 1-3
        for j in range((i + 1), 4):  # and quarter-windows (i+1)-4
            feature_names.extend(
                [
                    "mean_d_q" + str(i + 1) + "q" + str(j + 1) + "_" + str(k)
                    for k in range(len(v1))
                ]
            )
    return result, feature_names


@register_feature(default=True)
def feature_mean_half_window_difference(
    h1: np.ndarray, h2: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """Mean difference between half-windows-frames"""
    return compute_half_window_feature(h1, h2, "mean_diff", "feature_mean")


@register_feature(default=True)
def feature_mean_quarter_difference(
    q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, q4: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """Mean difference between half-windows-frames plus the mean difference between
    quarter-windows-frames"""
    return compute_quarter_window_feature(
        q1, q2, q3, q4, ["mean", "mean_diff"], "feature_mean"
    )


@register_feature(default=True)
def feature_stddev_half_window_difference(
    h1: np.ndarray, h2: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """Std difference between half-windows-frames"""
    return compute_half_window_feature(h1, h2, "std_diff", "feature_stddev")


@register_feature(default=True)
def feature_max_half_window_difference(
    h1: np.ndarray, h2: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """Max difference between half-windows-frames"""
    return compute_half_window_feature(h1, h2, "max_diff", "feature_max")


@register_feature(default=True)
def feature_max_quarter_difference(
    q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, q4: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """Max difference between half-windows-frames plus the max difference between
    quarter-windows-frames"""
    return compute_quarter_window_feature(
        q1, q2, q3, q4, ["max", "max_diff"], "feature_max"
    )


@register_feature(default=True)
def feature_min_half_window_difference(
    h1: np.ndarray, h2: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """Min difference between half-windows-frames"""
    return compute_half_window_feature(h1, h2, "min_diff", "feature_min")


@register_feature(default=True)
def feature_min_quarter_difference(
    q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, q4: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """Min difference between half-windows-frames plus the min difference between
    quarter-windows-frames"""
    return compute_quarter_window_feature(
        q1, q2, q3, q4, ["min", "min_diff"], "feature_min"
    )
