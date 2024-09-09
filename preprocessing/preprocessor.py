from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple, Union

from functools import partial
import numpy as np
from pywt import wavedec, waverec, swt, threshold
from scipy.signal import welch, filtfilt, butter, firwin, kaiserord

from typeguard import typechecked
import warnings
from tqdm import tqdm

import copy
from data.datasets.dataset import EEGDataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from tslearn.clustering import TimeSeriesKMeans, KShape
from sklearn.decomposition import PCA
# from preprocessing.filters.artifact_removal import band_pass_filter, notch_filter

# import the defaultdict
from collections import defaultdict


CLUSTERING_METHODS = {
    "kmeans": TimeSeriesKMeans,
    "kshape": KShape
}


@typechecked
class Preprocessor:

    def __init__(self, dataset: EEGDataset, reduced_memory: bool = False):
        self.dataset = EEGDataset.from_preprocessed_data(data=dataset.data,
                                                         metadata=dataset.metadata,
                                                         old_dataset=dataset,
                                                         reduced_memory=reduced_memory)
        self.data = self.dataset.data
        self.metadata = self.dataset.metadata
        self.reduced_memory = reduced_memory

    def to_dataset(self) -> EEGDataset:
        """
        Returns the preprocessed dataset

        Returns:
            EEGDataset: EEGDataset

        """
        return EEGDataset.from_preprocessed_data(data=self.data,
                                                 metadata=self.metadata,
                                                 old_dataset=self.dataset,
                                                 reduced_memory=self.reduced_memory)

    def filter_by(self, by: str = "subjects", value: str = "S1") -> Preprocessor:
        """
        Filters the data by a specific value of a specific metadata field

        Args:
            by (str, optional): Metadata field to filter by. Defaults to "subject".
            value (str, optional): Value of the metadata field to filter by. Defaults to "S1".

        Returns:
            Preprocessor: Preprocessor

        """

        if by == "subjects":
            indices = np.where(np.isin(self.dataset.subjects, value))[0]
        elif by == "sessions":
            indices = np.where(np.isin(self.dataset.sessions, value))[0]
        elif by == "recordings":
            indices = np.where(np.isin(self.dataset.recordings, value))[0]
        elif by in self.dataset.metadata[0].keys():
            meta = np.arrqay([m[by] for m in self.dataset.metadata])
            assert len(meta.shape) == 1, f"Metadata field to filter by must be scalar, but is {meta.shape[1:]} dimensional"
            indices = np.where(np.isin(meta, value))[0]
        else:
            raise ValueError(f"Invalid value for by: {by}")

        self.data = [self.data[i] for i in indices]
        self.metadata = [self.metadata[i] for i in indices]
        self.dataset.subjects = [self.dataset.subjects[i] for i in indices]
        self.dataset.sessions = [self.dataset.sessions[i] for i in indices]
        self.dataset.recordings = [self.dataset.recordings[i] for i in indices]
        self.dataset._file_paths = [self.dataset._file_paths[i] for i in indices]

        return self

    def truncate(self,
                 valid_types: Optional[List[str]] = None,
                 invalid_types: Optional[List[str]] = ["break", "unknown"],
                 delete_invalid: bool = True,
                 keep_beginning: bool = False) -> Preprocessor:
        """
        Truncates the data according to the rage of event times.
        Data that is recorded before the first event started
        or after the last event finishes is removed.

        Only considers events of specific types.
        Either valid_types or invalid_types can be specified.

        Parameters
        ----------
        valid_types : Optional[List[str]], optional
            List of valid event types, takes precedence over invalid_types, by default None
        invalid_types : Optional[List[str]], optional
            List of invalid event types, by default ["break", "unknown"]
        delete_invalid : bool, optional
            Whether to delete invalid data, e.g recordings which dont have event information
            or which would be trunkated to contain no samples, by default True
        Returns:
            Preprocessor: Preprocessor

        """
        assert valid_types is not None or invalid_types is not None, "Either valid_types or invalid_types must be specified"

        deletion_indices = []
        for i, metadata in tqdm(enumerate(self.metadata)):
            if "event_time" in metadata.keys():
                # we only want to keep data occuring during *interesting* events
                if valid_types is not None:
                    event_mask = np.isin(metadata["event_type"], valid_types)
                else:
                    event_mask = ~np.isin(metadata["event_type"], invalid_types)

                if keep_beginning:
                    start_time = metadata["relative_time"][0]
                else:
                    start_time = metadata["event_relative_time"][event_mask][0]

                end_time = metadata["event_relative_end_time"][event_mask][-1]
                trunk_mask = ((metadata["relative_time"] >= start_time)
                              & (metadata["relative_time"] <= end_time))

                if trunk_mask.sum() <= 0:
                    warnings.warn(
                        f"Truncation resulted in empty data for subject {self.dataset.subjects[i]}, mark this data for deletion")
                    # we can't delete the data right now, because that would mess up the loop
                    # therefore we mark the index for deletion and delete it later
                    deletion_indices.append(i)
                else:
                    # apply trunk mask to all columns of original data and metadata
                    # note for the metadata, the mask only applies to the columns that came with the original raw data
                    # e.g. not the event data columns

                    trunkated_data = self.data[i][trunk_mask]
                    trunkated_metadata = {k: v[trunk_mask]
                                          for k, v in metadata.items() if isinstance(v, np.ndarray) and len(v) == len(trunk_mask)}

                    # trunkate event meta data by applying the event mask
                    trunkated_metadata.update(
                        {k: v[event_mask]
                         for k, v in metadata.items() if isinstance(v, np.ndarray) and len(v) == len(event_mask)}
                    )
                    trunkated_metadata["event_duration"] = np.diff(trunkated_metadata["event_relative_end_time"])

                    self.data[i] = trunkated_data
                    self.metadata[i].update(trunkated_metadata)  # overwrite the old metadata with the trunkated metadata
            else:
                # warning if no event_time is found
                warnings.warn(
                    f"No event information found for recording {self.dataset.subjects[i]} cannot trunkate data, mark this data for deletion")
                deletion_indices.append(i)

        if delete_invalid:
            warnings.warn(
                f"Deleting {len(deletion_indices)} recordings out of {len(self.data)} which are without event information or which would be truncated to contain no samples")
            # delete all marked indices
            # we have to do this in reverse order, because otherwise the indices would change
            for i in deletion_indices[::-1]:
                del self.data[i]
                del self.metadata[i]
                del self.dataset.subjects[i]
                del self.dataset.sessions[i]
                del self.dataset.recordings[i]
            del self.dataset._file_paths[i]
        return self

    def derive_label_from_event_type(self) -> Preprocessor:
        """
        Creates a label assignment (one label for each sample in the data) based on the event types.
        Uses the event_type.split("_")[-1] as label.
        """
        for i, data in tqdm(enumerate(self.data)):
            metadata = self.metadata[i]
            if "event_type" in metadata.keys():
                # we need to reduce the event_types to only the last part of the event_type to get the label
                # e.g. "stimulus_name" -> "name"
                event_types = metadata["event_type"]
                label_rel_start_time = metadata["event_relative_time"]
                label_rel_end_time = np.concatenate([metadata["event_relative_time"][1:], [metadata["relative_time"][-1]]])
                labels = [event_type.split("_")[-1] for event_type in event_types]
                label_assignment = np.empty(len(data), dtype='U32')
                for (label, rel_time_start, rel_time_end) in zip(labels, label_rel_start_time, label_rel_end_time):
                    label_assignment[(metadata["relative_time"] >= rel_time_start)
                                     & (metadata["relative_time"] <= rel_time_end)] = label
                self.metadata[i]["label"] = label_assignment

        return self

    def normalize(self, method: str = "zscore") -> Preprocessor:
        """
        Normalizes the data according to the specified method

        Args:
            method (str, optional): Normalization method. Defaults to "zscore".

        Returns:
            Preprocessor: Preprocessor

        """
        if method == "zscore":
            for i, data in tqdm(enumerate(self.data), total=len(self.data)):
                # three dimensional data means [n_sequences, n_samples, n_channels]
                # two dimensional data means [n_samples, n_channels]
                # normalizing over n_sampels axis
                normalzation_constant = np.expand_dims(data.std(axis=-2), axis=-2)
                normalzation_constant[normalzation_constant == 0] = 1
                data_mean = np.expand_dims(data.mean(axis=-2), axis=-2)
                normed_data = (data - data_mean) / normalzation_constant
                self.data[i] = normed_data
        else:
            raise ValueError(f"Normalization method {method} not implemented")
        return self

    def scale(self, method: str = "minmax") -> Preprocessor:
        """
        Scales the data according to the specified method

        Args:
            method (str, optional): Scaling method. Defaults to "minmax".

        Returns:
            Preprocessor: Preprocessor

        """
        if method == "minmax":
            for i, data in tqdm(enumerate(self.data), total=len(self.data)):
                if data is not None:
                    normed_data = TimeSeriesScalerMinMax().fit_transform(data)
                    self.data[i] = normed_data
        else:
            raise ValueError(f"Scaling method {method} not implemented")
        return self

    def band_pass_filter(self, low: Optional[float] = 0.1, high: Optional[float] = 30, **filter_kwargs) -> Preprocessor:
        """
        Applies a band pass filter to the data

        Args:
            low (float, optional): Low cutoff frequency. Defaults to 0.1.
            high (float, optional): High cutoff frequency. Defaults to 30.

        Returns:
            Preprocessor: Preprocessor

        """
        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            if data is not None:
                filtered_data = band_pass_filter(data, range=(
                    low, high), sample_frequency=self.dataset._sampling_rate, **filter_kwargs)
                self.data[i] = filtered_data
        return self

    def kaiser_band_pass_filter(
        self, low: Optional[float] = 1.5, high: Optional[float] = 42.0,
        width_low: Optional[float] = 2.0, width_high: Optional[float] = 6.0,
        ripple_low: Optional[float] = 70.0, ripple_high: Optional[float] = 20.0
    ) -> Preprocessor:
        """
        Applies a band pass filter to the data
        Args:
            low (float, optional): Low cutoff frequency. Defaults to 1.0.
            high (float, optional): High cutoff frequency. Defaults to 42.
            width_low (float, optional): Width of transition from pass at low cutoff (rel. to Nyq).
            width_high (float, optional): Width of transition from pass at high cutoff (rel. to Nyq).
            ripple_low (float, optional): Desired attenuation in the stop band at low cutoff (dB).
            ripple_high (float, optional): Desired attenuation in the stop band at high cutoff (dB).
        Returns:
            Preprocessor: Preprocessor
        """
        nyq = self.dataset._sampling_rate/2.0

        # Compute order and Kaiser parameter for the FIR filter.
        N_high, beta_high = kaiserord(ripple_high, width_high/nyq)
        N_low, beta_low = kaiserord(ripple_low, width_low/nyq)

        # Compute coefficients for FIR filters
        taps_high = firwin(N_high, high/nyq, window=('kaiser', beta_high))
        taps_low = firwin(N_low, low/nyq, window=('kaiser', beta_low), pass_zero=False)

        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            if data is not None:
                self.data[i] = filtfilt(taps_low, 1.0, self.data[i], axis = -2) #highpass
                self.data[i] = filtfilt(taps_high, 1.0, self.data[i], axis = -2) #lowpass
                
        return self

    def notch_filter(self, frequency: Union[float, List[float]] = 50.0, **filter_kwargs) -> Preprocessor:
        """
        Applies a notch filter to the data at 50Hz and 60Hz. The reason is that recordings 
        may come from EU or US, yielding line noise at 50 or 60Hz, respectively.

        Parameters
        ----------
        frequency : Union[float, List[float]]
            Frequency to filter out, defaults to 50.0
            Can be a list of frequencies to filter out multiple frequencies or a single frequency, e.g. 50.0 or [50.0, 60.0]
        filter_kwargs : dict
            Keyword arguments to pass to the filter function, 
            see mind_ml.data.artifact_removal.notch_filter for details

        Returns:
            Preprocessor: Preprocessor

        """
        if isinstance(frequency, int):
            frequency = [frequency]
        assert len(frequency) > 0, "frequency must be a list of integers"

        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            if data is not None:
                filtered_data = data
                for f in frequency:
                    filtered_data = notch_filter(filtered_data,
                                                 frequency=f,
                                                 sample_frequency=self.dataset._sampling_rate,
                                                 **filter_kwargs)
                self.data[i] = filtered_data
        return self

    def dwt(self, type="dwt", wavelet="db8", mode="sym",  level: int = 4) -> Preprocessor:
        """
        Applies a discrete wavelet decomposition or stationary wavelet decomposition of the data into 
        approximate and detailed coefficients containing low- and high-frequency components, respectively. 

        Note: the obtained coefficients may be used for denoising by thresholding the
        coefficients and inverting the filtered space back to the time domain. 

        Parameters
        ----------
        type : string
            Choose between discrete wavelet transformation ("dwt") and static wavelet transformation ("swt").
            SWT is a translation-invariance modification of the discrete wavelet transform (DWT).
            Default: "dwt"
        wavelet : string
            Wavelet type to use in decomposition 
            An exhaustive list of the wavelet families can be obtained by calling pywt.families()
            Default: "db8" - Daubechies-8 wavelet is used as suggested in Asaduzzaman etl al. (2010)
        mode : string
            Signal extension mode. Extrapolates the input data to extend the signal before computing the dwt.
            Depending on the extrapolation method, significant artifacts at the signal's borders can be introduced during that process.
            Default: "sym" - symmetric window 
            Check: https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html#ref-modes for further explanation
        level : int
            Number of decompositions to be done (yields 1 approximate, n-1 detailed coefficients)
            If None, four-level decomposition is done as suggested in Aliyu & Lim (2023)

        Returns:
            Preprocessor: Preprocessor
                Adds the following columns to the metadata:
                    - approx_coeffs (array): approximate coefficients 
                    - detail_coeffs (list of arrays): detailled coefficients
                    Note: Contains level-1 arrays of detailled coefficients
        """
        types = ["dwt", "swt"]
        if type not in types:
            raise ValueError("Invalid type. Expected one of: %s" % types)

        if type == "dwt":
            transform = partial(wavedec, wavelet = wavelet, level = level, axis = -2)
        else:
            transform = partial(swt, wavelet = wavelet, level = level, axis = -2, trim_approx = True)

        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            if data is None:
                warnings.warn(f"No data found for recording {self.dataset.subjects[i]}, skipping")
                continue

            #n_sequences, window_size, n_channels = data.shape
            #assert num_channels == self.dataset.num_channel, "Data must have 3 channels"

            coeffs = transform(self.data[i])

            # updates metadata with approximate and detailed coefficients
            self.metadata[i].update(
                {
                    "approx_coeffs": coeffs[0],
                    "detail_coeffs": coeffs[1:],
                    "wavelet": wavelet
                }
            )

        return self
    
    def simple_threshold_denoising(self, mode="soft") -> Preprocessor:
        """
        dwt-based denoising after Donoho & Johnstone (1994) with a simple set threshold.
        Note: requires to run dwt to use coefficients for denoising!
        Parameters
        ----------
        mode : string
            Determines which type of thresholding is used.
            For a list of thresholding methods, check:
            https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html
            Default: "soft"
        Returns:
            Preprocessor: Preprocessor
        """
        thresholding = partial(threshold, mode=mode, substitute=0)
        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            if data is not None:
                res = []
                for detail in self.metadata[i]["detail_coeffs"]:
                    sig = np.median(np.abs(detail), axis=-2)/0.6745
                    thresh = sig * np.sqrt(2*np.log(len(detail)))
                    if np.all(thresh) == True:
                        thresholded_detail = thresholding(data=detail, value=thresh)
                        res.append(thresholded_detail)
                    else:
                        res.append(detail)
                self.data[i] = waverec([self.metadata[i]["approx_coeffs"]] + res, wavelet = self.metadata[i]["wavelet"], axis=-2)
        return self

    def neigh_block_denoising(self, sigma: int = 2.0) -> Preprocessor:
        """
        dwt-based denoising using the NeighBlock method (Cai & Silverman, 2001). Data is inverted back to time domain. 
        NeighBlock estimates wavelet coefficients simultaneously in groups and uses neighboring coefficients outside 
        the block of current interest in fixing the threshold.  
        Note: requires to run dwt to use coefficients for denoising!
        Parameters
        ----------
        sigma : int
            Noise level. Defaults to 2.0.
        Returns:
            Preprocessor: Preprocessor
        """
        assert "approx_coeffs" in self.metadata[0].keys(), "No coefficients - first run dwt."

        # helper function to compute beta shrinkage
        def beta(sigma, L, detail, lmbd):
            S2 = np.sum(detail ** 2)
            beta = (1 - lmbd * L * sigma**2 / S2)
            return max(0, beta)

        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            if data is not None:
                res = []
                L0 = int(np.log2(len(self.data[i])) // 2)
                L1 = max(1, L0 // 2)
                L = L0 + 2 * L1
                lmbd = 4.50524  # explanation in Cai & Silverman (2001)
                for detail in self.metadata[i]["detail_coeffs"]:
                    d2 = detail.copy()
                    # group wavelet into disjoint blocks of length L0
                    for start_b in range(0, len(d2), L0):
                        end_b = min(len(d2), start_b + L0)  # if len(d2) < start_b + L0 -> last block
                        start_B = start_b - L1
                        end_B = start_B + L
                        if start_B < 0:
                            end_B -= start_B
                            start_B = 0
                        elif end_B > len(d2):
                            start_B -= end_B - len(d2)
                            end_B = len(d2)
                        assert end_B - start_B == L  # sanity check
                        d2[start_b:end_b] *= beta(sigma, L, d2[start_B:end_B], lmbd)
                    res.append(d2)
                self.data[i] = waverec([self.metadata[i]["approx_coeffs"]] + res, wavelet = self.metadata[i]["wavelet"], axis = -2)
        return self

    def difference(self, order: int = 1) -> Preprocessor:
        """
        Calculates the difference of the data

        Args:
            order (int, optional): Order of difference. Defaults to 1.

        Returns:
            Preprocessor: Preprocessor

        """
        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            diff_data = np.diff(data, n=order, axis=-2)
            self.data[i] = diff_data
            # when differencing the first 'order' samples are lost
            # we need to adjust the meta data accordingly by removing the first 'order' samples

            self.metadata[i].update(
                {k: v[order:]
                 for k, v in self.metadata[i].items() if isinstance(v, np.ndarray) and len(v) == len(data)}
            )

        return self

    def downsample(self, factor: int = 2) -> Preprocessor:
        """
        Downsamples the data

        Args:
            factor (int, optional): Downsampling factor. Defaults to 2.

        Returns:
            Preprocessor: Preprocessor

        """
        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            if data is not None:
                downsampled_data = data[::factor]
                self.data[i] = downsampled_data
        return self

    def bandpower(self,
                  window_size: Optional[float] = 1,
                  window_stride: float = 0.25,
                  max_freq: Optional[int] = 50,
                  to_db: bool = False) -> Preprocessor:
        """
        Calculates the band power as well as the PSD (Power Spectral Density) of the data using a sliding window approach.
        The PSD and therby the band power is calculated independently for each window making the resulting data a time series of PSDs and band powers.

        The PSD is calculated using a Welch method.

        Parameters
        ----------
        window_size : float
            Size of the sliding window in seconds.
            If None, the window size is set to the sampling frequency, by default 1
        window_stride : float
            Window stride (relative step size) of the sliding window in relation to window size. Must be in the range [0, 1], by default 0.25
        to_db : bool
            If True, the band power is converted to decibel (dB) values, by default False

        Returns:
            Preprocessor: Preprocessor
                Adds the following columns to the metadata:
                    - psd: Power Spectral Density
                    - psd_freqs: Frequency bins of the PSD
                    - bandpower: Band power

        """
        assert window_stride > 0 and window_stride <= 1, "Window stride must be between 0 and 1"
        if not window_size == None:
            window_size = int(window_size * self.dataset._sampling_rate)
            if window_size <= self.dataset._bands.max() * 2:
                raise ValueError(
                    f"Window size must be smaller than the Nyquist frequency {self.dataset._bands.max() * 2} Hz but got window_size={window_size} Hz")

        initial_window_size = window_size
        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            if data is None:
                warnings.warn(f"No data found for recording {self.dataset.subjects[i]}, skipping")
                continue
            
            if len(data.shape) == 2:
                num_samples = data.shape[0]
            else:
                num_sequences, num_samples, num_channels = data.shape
            #assert num_channels == self.dataset.num_channel, "Data must have 3 channels"

            # We want to compute in the following steps:

            # 1. Create a sliding window to sample from each recording with overlap (unless window_stride = 1)
            # sliding_window_view will yield n_sequences, trimmed_data_shape, n_channels, window_size
            # The step_size is adjusted depending on how long a sample should be (e.g. 10sec)
            # This results in n_sequences, n_samples, n_channels, window_size
            if initial_window_size == None:  
                window_size = num_samples
            assert window_size <= num_samples, f"Window sample size [{window_size}] must be smaller than number of samples [{num_samples}]"

            step_size = max(int(window_size * window_stride), 1)
            if len(self.data) == 1: #real-time processing -> self.data is a list containing the data array as entry
                sliding_view = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=-2)[:,::step_size]
            else: #offline preprocessing
                sliding_view = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=-2)[::step_size]

            metadata_update = {
                f"bandpower_{k}": np.lib.stride_tricks.sliding_window_view(v, window_size, axis=0)[::step_size, -1]
                for k, v in self.metadata[i].items() if isinstance(v, np.ndarray) and len(v.shape) != 0 and len(v) == num_samples
            }

            # 2. Compute psds on the samples, the time-data is axis=-1 
            f, psd = welch(sliding_view, fs=self.dataset._sampling_rate, axis=-1)

            if to_db:
                psd = 10 * np.log10(psd)

            # 3. Create bandpowers from psds
            # vectorized version of band_power_from_psd
            f = f[:max_freq]
            if len(psd.shape) == 4: #Real-Time
                psd = psd[:, :, :, :max_freq]
                band_power = np.zeros((psd.shape[0], psd.shape[1], psd.shape[2], len(self.dataset._bands)))
                for j, band in enumerate(self.dataset._bands):
                    band_lower, band_upper = band
                    freq_mask = (f >= band_lower) & (f < band_upper)
                    band_power[:, :, :, j] = np.mean(psd[:, :, :, freq_mask], axis=-1)
            else: #Offline
                psd = psd[:, :, :max_freq]
                band_power = np.zeros((psd.shape[0], psd.shape[1], len(self.dataset._bands)))
                for j, band in enumerate(self.dataset._bands):
                    band_lower, band_upper = band
                    freq_mask = (f >= band_lower) & (f < band_upper)
                    band_power[:, :, j] = np.mean(psd[:, :, freq_mask], axis=-1)

            # updates metadata with bandpower information
            self.metadata[i].update(
                {
                    "bandpowers": band_power,
                    "psd": psd,
                    "psd_freqs": f,
                    "bandpower_sampling_rate": 1 / window_stride,
                }
            )
            self.metadata[i].update(metadata_update)

        return self

    def normalize_bandpowers(self) -> Preprocessor:
        """

        """
        for i, metadata in tqdm(enumerate(self.metadata), total=len(self.metadata)):
            # calculate mean over all bandpowers and subtract it from each bandpower
            bandpowers = metadata["bandpowers"] - np.mean(metadata["bandpowers"], axis=-1, keepdims=True)
            self.metadata[i]["bandpowers"] = bandpowers

        return self

    def bandpower_ratios(self, band_names: List[Tuple[str, str]]) -> Preprocessor:
        """"
        Calculates the band power ratios for the given bandpowers.

        Parameters
        ----------
            band_names (List[Tuple[str, str]]):
                List of tuples containing the names of the bands to calculate the ratio for e.g. [("alpha", "beta"), ("theta", "delta")]

        Returns:
            Preprocessor: Preprocessor
                Adds the following columns to the metadata:
                    - bandpower_ratios: Band power ratios
                    - bandpower_ratio_names: Names of the band power ratios (the band_names parameter)
        """
        eps = 1e-10
        for i, metadata in tqdm(enumerate(self.metadata), total=len(self.metadata)):
            if metadata is None:
                warnings.warn(f"No metadata found for recording {self.dataset.subjects[i]}, skipping")
                continue

            if "bandpowers" in metadata.keys():
                bandpowers = metadata["bandpowers"]
                bandpower_ratios = np.zeros((bandpowers.shape[0], bandpowers.shape[1], len(band_names)))
                for j, (band_1, band_2) in enumerate(band_names):
                    band_1_idx = self.dataset._band_idx[band_1]
                    band_2_idx = self.dataset._band_idx[band_2]
                    bandpower_ratios[:, :, j] = bandpowers[:, :, band_1_idx] / (bandpowers[:, :, band_2_idx]+eps)
                self.metadata[i].update(
                    {
                        "bandpower_ratios": bandpower_ratios,
                        "bandpower_ratio_names": band_names,
                    }
                )
            else:
                raise ValueError("Bandpowers not calculated yet. Please call .bandpower() first")
        return self

    def bandpower_ratio_as_data(self, ratio_name: Tuple[str, str]) -> Preprocessor:
        """
        Replaces the data with the band power such that the new data for each recording
        is a time series of band powers with shape [n_samples, n_bands, n_channels]

        Returns:
            Preprocessor: Preprocessor
        """

        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            assert "bandpower_ratios" in self.metadata[i], "Bandpower not calculated, please run Preprocessor.bandpower() first"
            assert "bandpower_ratio_names" in self.metadata[i], "Bandpower not calculated, please run Preprocessor.bandpower() first"

            ratio_names = self.metadata[i]["bandpower_ratio_names"]
            ratios = self.metadata[i]["bandpower_ratios"]

            assert ratio_name in ratio_names, f"Ratio {ratio_name} not found in ratio names {ratio_names}, available ratios are {ratio_names}"
            # get idx
            ratio_idx = ratio_names.index(ratio_name)
            self.data[i] = ratios[:, :, ratio_idx]
            self.metadata[i]["relative_time"] = self.metadata[i]["bandpower_relative_time"]
            self.dataset._sampling_rate = self.metadata[i]["bandpower_sampling_rate"]

        return self

    def rolling_mean(self, window_size: float, window_stride: float) -> Preprocessor:
        """
        Calculates the rolling mean of the data

        Parameters
        ----------
            window_size (float):
                Size of the rolling window, in seconds.
                Can not be larger than the length of the data.
            window_stride (float):
                Stride of the window in relation to the window size. Must be between 0 and 1.
                Small values mean a larger overlap between the windows and more windows.
                Greater values mean a smaller overlap and less windows. A value of 1 means no overlap at all.

        Returns:
            Preprocessor: Preprocessor

        """
        assert window_stride > 0 and window_stride <= 1, "window_stride must be between 0 and 1"
        window_size *= self.dataset._sampling_rate
        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            if data is not None:
                num_samples = data.shape[0]
                assert window_size <= num_samples, "Window size must be smaller than number of samples"
                step_size = max(int(window_size * window_stride), 1)
                sliding_view = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=-2)[::step_size]
                mean = np.mean(sliding_view, axis=-1)
                self.data[i] = mean

        return self

    def mean_channels(self, channel_mask: np.ndarray, channel_names: List[str]) -> Preprocessor:
        """
        Calculates the means of the channels specified by the channel_mask.

        `code example`
        ```python
        # calculate the mean of the first and second channel
        preprocessor.mean_channels(channel_mask=np.array(
            [[True, True, False], [False, True, True]]), channel_names=["ch1_2", "ch2_3"])
        ```
        wil result in the data to have shape [num_windows, 2] and the dataset._channel_names to be ["ch1_2", "ch2_3"]


        Parameters
        ----------
            channel_mask (np.ndarray):
                Boolean mask of the channels to be averaged. Must be of shape [num_new_channels, num_old_channels]
            channel_names (List[str]):
                Names of the new channels, must be of length num_new_channels

        Returns:
            Preprocessor: Preprocessor
        """

        assert len(
            channel_names) == channel_mask.shape[0], "Number of channel names must be equal to the number of new channels"
        assert (channel_mask.sum(-1) > 0).all(), "At least one old channel must be selected for each new channel"
        for i, data in tqdm(enumerate(self.data), total=len(self.data)):

            assert data.shape[-1] == channel_mask.shape[1], "Number of channels in data must be equal to the number of old channels"
            nan_channel_mask = np.isnan(data).any(0)
            # in case one of the channels is nan, we want to ignore this channel when calculating the mean
            # this is done by setting the channel mask to False
            channel_mask = channel_mask * ~nan_channel_mask
            # and setting the data to zero for the nan channel
            # this is also overwritting self.data but that's fine since we are overwriting it with the mean anyways
            data[:, nan_channel_mask] = 0
            mean = (np.expand_dims(data, axis=-2) * channel_mask).sum(-1) / channel_mask.sum(-1)[None, ...]

            assert mean.shape[-1] == len(
                channel_names), "Number of channels in averaged data must be equal to the number of new channels"
            self.data[i] = mean

        self.dataset._channel_names = channel_names
        return self

    def mean_subjects(self, groups: List[str], skip_data=False) -> Preprocessor:
        """
        Averages the data and bandpower across subjects, according to the groups provided.
        Only averages for those recording that have the same metadata["event_type"]

        Parameters
        ----------
            groups (List[str]):
                List of subject prefixes to be grouped together. E.g. ["con", "asd"] will group together all subjects with the prefix "con" and "asd".

        Returns:
            Preprocessor: Preprocessor
        """

        assert len(groups) > 0, "At least one group must be provided"
        assert len(groups) == len(set(groups)), "Groups must be unique"

        # get all subjects
        subjects = self.dataset.subjects

        # group the data according to subject and event type
        grouped_data = defaultdict(list)
        grouped_metadata = defaultdict(list)
        for i, subject in enumerate(subjects):
            group = None
            for g in groups:
                if subject.startswith(g):
                    group = g
                    break
            assert group is not None, f"Subject {subject} does not belong to any group"

            event_type = self.metadata[i]["event_type"] if "event_type" in self.metadata[i] else ["default"]
            assert len(event_type) == 1, "Only one event type per recording is supported"
            event_type = event_type[0]
            grouped_data[(group, event_type)].append(self.data[i])
            grouped_metadata[(group, event_type)].append(self.metadata[i])

        # average the data
        new_data = []
        new_metadata = []
        new_subjects = []
        new_sessions = []
        new_recordings = []

        for i, (group, event_type) in enumerate(grouped_data):

            # average the data
            if skip_data:
                data_shape = grouped_data[(group, event_type)][0].shape
                data = np.zeros(data_shape)
            else:
                group_np_data = np.stack(grouped_data[(group, event_type)])
                data = np.mean(group_np_data, axis=-2)

            # averge the metadata
            metadata = {}
            metadata_keys_mean = [
                "psd",
                "bandpowers",
                "bandpower_relative_time"
                "event_relative_time",
                "event_relative_end_time",
            ]
            metadata_keys_keep_first = [
                "psd_freqs",
                "event_type",
                "label",
                "bandpower_label",
            ]
            for key in metadata_keys_mean:
                if key in grouped_metadata[(group, event_type)][0]:
                    stacked = np.stack([m[key] for m in grouped_metadata[(group, event_type)]])
                    metadata[key] = np.mean(stacked, axis=-2)

            for key in metadata_keys_keep_first:
                if key in grouped_metadata[(group, event_type)][0]:
                    metadata[key] = grouped_metadata[(group, event_type)][0][key]

            new_data.append(data)
            new_metadata.append(metadata)
            new_subjects.append(group)
            new_sessions.append(None)
            new_recordings.append(None)

        self.data = new_data
        self.metadata = new_metadata
        self.dataset.subjects = new_subjects
        self.dataset.sessions = new_sessions
        self.dataset.recordings = new_recordings

        return self

    def explode_events_into_recordings(self) -> Preprocessor:
        """
        Each recording has multiple sections/windows defined by the events.
        This function explodes these windows into their own recordings.

        E.g. if there are 5 subjects, each with two recordings and each recording has 10 events,
            then the dataset will have 5 subjects with 20 recordings after calling this function.

        """

        new_data = []
        new_metadata = []
        new_subjects = []
        new_sessions = []
        new_recordings = []
        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            metadata = self.metadata[i]
            subject = self.dataset.subjects[i]
            session = self.dataset.sessions[i]
            recording = self.dataset.recordings[i]
            recording = f"{recording}_" if recording is not None else ""

            for j, event_type in enumerate(metadata["event_type"]):
                start_time = metadata["event_relative_time"][j]
                end_time = metadata["event_relative_end_time"][j]
                event_window_mask = ((metadata["relative_time"] >= start_time)
                                     & (metadata["relative_time"] <= end_time))

                if event_window_mask.sum() <= 0:
                    raise ValueError(f"No data found for event {event_type} in recording {subject}/{session}/{recording}")
                else:
                    # apply event_window_mask to the original data
                    event_window_data = data[event_window_mask]

                    # apply event_window_mask to metadata as well
                    event_window_metadata = copy.deepcopy(metadata)
                    event_window_metadata.update(
                        {k: v[event_window_mask]
                         for k, v in metadata.items() if isinstance(v, np.ndarray) and len(v) == len(event_window_mask)}
                    )
                    # trunkate event meta data by applying the event mask
                    event_window_metadata.update(
                        {k: v[j:j+1]
                         for k, v in metadata.items() if isinstance(v, np.ndarray) and len(v) == len(metadata["event_type"])}
                    )
                    event_window_metadata["event_duration"] = event_window_metadata["event_relative_end_time"] - \
                        event_window_metadata["event_relative_time"]

                    # the relative time is not relative to the start of the recording anymore
                    # fix this by subtracting the start time of the new
                    event_window_metadata["relative_time"] -= event_window_metadata["relative_time"][0]
                    event_window_metadata["event_relative_time"] -= event_window_metadata["event_relative_time"][0]
                    event_window_metadata["event_relative_end_time"] -= event_window_metadata["event_relative_time"][0]

                    # add new data, metadata, subjects, sessions and recordings
                    assert len(event_window_data.shape) == 2, "Data must be 2 dimensional"
                    assert event_window_data.shape[1] == self.dataset.num_channel, "Number of channels must stay equal to the number of channels in the dataset"
                    new_data.append(event_window_data)
                    new_metadata.append(event_window_metadata)
                    new_subjects.append(subject)
                    new_sessions.append(session)
                    new_recordings.append(f"{recording}{event_type}_{j}")

        self.data = new_data
        self.metadata = new_metadata
        self.dataset.subjects = new_subjects
        self.dataset.sessions = new_sessions
        self.dataset.recordings = new_recordings

        # assert no None values in data or metadata
        assert all([d is not None for d in self.data]), "None values in data"
        assert all([d is not None for d in self.metadata]), "None values in metadata"

        return self

    def explode_recording_into_windows(self, window_size: float = 1, window_stride: float = 1) -> Preprocessor:
        """
        Each recording is split into multiple windows of length window_size with a stride of window_stride.
        This function explodes these windows into their own recordings.

        The number of resulting recordings is (num_recordings * num_windows_per_recording).
        Note that num_windows_per_recording may vary between recordings.


        Parameters
        ----------
            window_size (float):
                Size of the window in seconds, must be smaller than the shortest recording. By default 1.
            window_stride (float):
                Stride of the window (relative step size) in relation to the window size. By default 1.
                Values smaller than 1 will result in overlapping windows. Values larger than 1 will result in gaps between windows.

        Returns:
            Preprocessor: Preprocessor
        """

        assert window_size > 0, "Window size must be larger than 0"
        assert window_stride > 0, "Window stride must be larger than 0"

        window_sample_size = int(window_size * self.dataset._sampling_rate)
        window_sample_stride = int(window_stride * window_sample_size)

        new_data = []
        new_metadata = []
        new_subjects = []
        new_sessions = []
        new_recordings = []
        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            metadata = self.metadata[i]
            subject = self.dataset.subjects[i]
            session = self.dataset.sessions[i]
            recording = self.dataset.recordings[i]
            recording = f"{recording}_" if recording is not None else ""

            # calculate number of windows
            num_windows = int(np.floor(((data.shape[0] - window_sample_size) / window_sample_stride) + 1))
            assert num_windows > 0, f"Check window size and stride. Unpossible to extract window from {subject}/{session}/{recording}"
            for j in range(num_windows):
                start_sample = j * window_sample_stride
                end_sample = start_sample + window_sample_size

                # apply window mask to the original data
                if end_sample > data.shape[0]:
                    print("asd")
                assert end_sample <= data.shape[0], "Window must not exceed the length of the recording"
                window_data = data[start_sample:end_sample]
                assert window_data.shape[0] == window_sample_size, "Window size must be equal to window_sample_size"

                # apply window mask to metadata as well
                window_metadata = copy.deepcopy(metadata)
                window_metadata.update(
                    {k: v[start_sample:end_sample]
                     for k, v in metadata.items() if isinstance(v, np.ndarray) and len(v) == len(data)}
                )

                # the relative time is not relative to the start of the recording anymore
                # fix this by subtracting the start time of the new
                window_metadata["relative_time"] -= window_metadata["relative_time"][0]

                # add new data, metadata, subjects, sessions and recordings
                assert len(window_data.shape) == 2, "Data must be 2 dimensional"
                assert window_data.shape[1] == self.dataset.num_channel, "Number of channels must stay equal to the number of channels in the dataset"
                new_data.append(window_data)
                new_metadata.append(window_metadata)
                new_subjects.append(subject)
                new_sessions.append(session)
                new_recordings.append(f"{recording}win{j}")

        self.data = new_data
        self.metadata = new_metadata
        self.dataset.subjects = new_subjects
        self.dataset.sessions = new_sessions
        self.dataset.recordings = new_recordings

        # assert no None values in data or metadata
        assert all([d is not None for d in self.data]), "None values in data"
        assert all([d is not None for d in self.metadata]), "None values in metadata"

        return self

    def extract_event_related_windows(self, window_size: float = 1) -> Preprocessor:
        """
        Each recording is first reduced to a fixed size window of length window_size placed after the event onset.


        Parameters
        ----------
            window_size (float):
                Size of the window in seconds, must be smaller than the shortest recording. By default 1.

        """

        assert window_size > 0, "Window size must be larger than 0"

        window_sample_size = int(window_size * self.dataset._sampling_rate)

        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            metadata = self.metadata[i]
            subject = self.dataset.subjects[i]
            session = self.dataset.sessions[i]
            recording = self.dataset.recordings[i]

            assert len(metadata["event_type"]
                       ) == 1, "Expects only one event per recording, call explode_recording_into_events first"

            new_data = data[:window_sample_size]
            update_metadata = {
                k: v[:window_sample_size]
                for k, v in metadata.items() if isinstance(v, np.ndarray) and len(v) == len(data)
            }

            self.metadata[i].update(update_metadata)
            self.data[i] = new_data

        return self

    def event_related_potential_extraction(self, groupby: str = "subject", group_prefixes: Optional[List[str]] = None) -> Preprocessor:
        """
        """

        new_data = []
        new_metadata = []
        new_subjects = []
        new_sessions = []
        new_recordings = []

        # get all the event types in the dataset
        unique_events = np.unique(np.array([m["event_type"] for m in self.metadata]))
        if group_prefixes is None:
            # initialize group prefixes
            if groupby == "subject":
                # with subject names
                group_prefixes = np.unique(np.array([s for s in self.dataset.subjects]))
            else:
                raise ValueError(f"Groupby {groupby} not supported")
        # make sure group_prefixes is lower case
        group_prefixes = [g.lower() for g in group_prefixes]

        for i, group_prefix in enumerate(group_prefixes):
            for j, event_type in enumerate(unique_events):
                group_data = []
                grouped_metadata = {}
                for k, data in tqdm(enumerate(self.data), total=len(self.data)):
                    if groupby == "subject":
                        groupby_element: str = self.dataset.subjects[k].lower()
                    else:
                        raise ValueError(f"Groupby {groupby} not supported")

                    if groupby_element.startswith(group_prefix) and self.metadata[k]["event_type"][0] == event_type:
                        group_data.append(data)
                        for key, value in self.metadata[k].items():
                            if isinstance(value, np.ndarray) and len(value) == len(data):
                                if key not in grouped_metadata:
                                    grouped_metadata[key] = []
                                grouped_metadata[key].append(value)
                if len(group_data) == 0:
                    continue
                new_data.append(np.mean(np.stack(group_data, axis=-2), axis=-2))
                new_metadata.append({
                    k: np.mean(np.stack(v, axis=-2), axis=-2)
                    for k, v in grouped_metadata.items()
                    if v[0].dtype == np.float64
                })
                new_metadata[-1]["event_type"] = np.array([event_type])
                new_metadata[-1]["event_relative_time"] = np.array([0.0])
                new_subjects.append(group_prefix)
                new_sessions.append(event_type)
                new_recordings.append(None)

        self.data = new_data
        self.metadata = new_metadata
        self.dataset.subjects = new_subjects
        self.dataset.sessions = new_sessions
        self.dataset.recordings = new_recordings

        return self

    def explode_recording_into_event_related_potentials(self, window_size: float = 1.0) -> Preprocessor:
        """
        This function explodes the dataset into event related potentials (ERP).
        It operates on the original recordings, not on the windows.
        For each recording it extracts a fixed size window of length window_size placed before and after the event onset.
        Which means the total size of each extracted window is 2 * window_size with the event onset at the center of the window

        Parameters
        ----------
            window_size (float):
                Size of the window in seconds. By default 1.

        Returns:
            Preprocessor: Preprocessor
        """

        assert window_size > 0, "Window size must be larger than 0"

        window_sample_size = int(window_size * self.dataset._sampling_rate)

        new_data = []
        new_metadata = []
        new_subjects = []
        new_sessions = []
        new_recordings = []

        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            metadata = self.metadata[i]
            subject = self.dataset.subjects[i]
            session = self.dataset.sessions[i]
            recording = self.dataset.recordings[i]

            # iterate over all events and extract the window
            for j, event_time in enumerate(metadata["event_relative_time"]):
                # get the index of the first sample after the event based on data (relative) time
                event_sample = np.where(metadata["relative_time"] >= event_time)[0][0]
                # get the index of the first & last sample of the window
                first_sample = event_sample - window_sample_size
                last_sample = event_sample + window_sample_size
                if first_sample < 0 or last_sample >= len(data):
                    # warn the user
                    print(
                        f"Event {j} of recording {recording} is too close to the start or end of the recording, skipping, first sample: {first_sample}, last sample: {last_sample}")
                    continue
                # check if the window is within the recording
                # if last_sample < len(data):
                new_data.append(data[first_sample:last_sample])

                new_metadata.append({
                    k: v[first_sample:last_sample]
                    for k, v in metadata.items()
                    if isinstance(v, np.ndarray) and len(v) == len(data)
                })
                # add event related metadata
                new_metadata[-1].update({
                    k: v[j:j+1]
                    for k, v in metadata.items()
                    if isinstance(v, np.ndarray) and len(v) == len(metadata["event_relative_time"])
                })
                # recalculate relative time metadata
                # get the start time of the window to recalculate the relative time w.r.t. this fixed timestamp
                start_time = new_metadata[-1]["relative_time"][0]
                new_metadata[-1].update({
                    k: v - start_time
                    for k, v in new_metadata[-1].items()
                    if "relative_time" in k and isinstance(v, np.ndarray)
                })
                new_subjects.append(subject)
                new_sessions.append(session)
                new_recordings.append(recording)
                # else:
                #     # if the window is not within the recording, ignore it but warn the user
                #     warnings.warn(
                #         f"Event {j} of type {metadata['event_type'][j]} in recording {subject} {session} is too close to the end of the recording. Ignoring it.")

        self.data = new_data
        self.metadata = new_metadata
        self.dataset.subjects = new_subjects
        self.dataset.sessions = new_sessions
        self.dataset.recordings = new_recordings

        return self

    def flatten_recordings(self) -> Preprocessor:
        """
        Flattens the list of recordings into a single recording where the data is one big numpy array of shape [n_sequences, n_samples, n_channels].
        This is useful (and required) for training a model, doing PCA, cluster, etc.

        Doing this requires that all recordings have the same number of samples.
        Call explode_recording_into_windows to ensure this.

        Returns:
            Preprocessor: Preprocessor
        """

        # assert all recordings have the same number of samples
        assert all([d.shape == self.data[0].shape for d in self.data]
                   ), "All recordings must have the same shape"

        # flatten data
        self.data = [np.stack(self.data, axis=-2)]

        # flatten metadata
        new_metadata = {}
        # implicitly assume that all metadata is the same for all recordings, let's hope this is true
        for k in self.metadata[0].keys():
            try:
                new_metadata[k] = np.stack([m[k] for m in self.metadata], axis=-2)
            except:
                warnings.warn(f"Could not stack metadata {k} into a numpy array")

        new_metadata["subject"] = np.array(self.dataset.subjects)
        new_metadata["session"] = np.array(self.dataset.sessions)
        new_metadata["recording"] = np.array(self.dataset.recordings)

        self.metadata = [new_metadata]

        # assert all metadata has length n_sequences
        assert all([len(v) == self.data[0].shape[0] for v in self.metadata[0].values()]),\
            "All metadata must have length n_sequences after flattening"

        # flatten subjects, sessions and recordings
        self.dataset.subjects = ["all"]
        self.dataset.sessions = [None]
        self.dataset.recordings = [None]
        return self

    def cluster(self, method: str = "kshape", **cluster_kwargs) -> Preprocessor:
        """
        Assumes that the recordings have been flattened into a single recording using .flatten_recordings()
        Clusters the data using the given method and adds the following metadata:
            - cluster: cluster label for each sequence
            - cluster_center: cluster center for each sequence

        Parameters
        ----------
            method (str):
                Clustering method. By default "kshape".
                - "kshape": KShape clustering
                - "kmeans": KMeans clustering
            cluster_kwargs (dict):
                Keyword arguments for the clustering method

        Returns:
            Preprocessor: Preprocessor
        """
        assert len(self.data) == 1, "Data must be flattened into a single recording"
        assert method in CLUSTERING_METHODS.keys(), f"Method must be one of {list(CLUSTERING_METHODS.keys())}"

        # get data
        data = self.data[0]

        # cluster data
        cluster_model = CLUSTERING_METHODS[method](**cluster_kwargs)
        cluster_labels = cluster_model.fit_predict(data)
        cluster_centers = cluster_model.cluster_centers_

        # add cluster and cluster center to metadata
        self.metadata[0]["cluster_label"] = cluster_labels
        self.metadata[0]["cluster_center"] = cluster_centers
        self.metadata[0]["cluster_model"] = cluster_model

        return self

    def pca(self, n_components: int = 2, **pca_kwargs) -> Preprocessor:
        """
        Assumes that the recordings have been flattened into a single recording using .flatten_recordings()
        and the data is of shape [n_sequences, n_samples, n_channels]
        Performs PCA on each channel independently and adds the following metadata:
            - explained_variance_ratio: explained variance ratio for each component, shape [n_components, n_channels]
            - pca_components: pca components for each channel, shape [n_components, n_samples, n_channels]
            - original_data: original data, shape [n_sequences, n_samples, n_channels]
        Overwrites the data with the PCA transformed data of shape [n_sequences, n_components, n_channels]


        Parameters
        ----------
            n_components (int):
                Number of components to keep. By default 2.
            pca_kwargs (dict):
                Keyword arguments for the PCA method

        Returns:
            Preprocessor: Preprocessor
        """

        assert len(self.data) == 1, "Data must be flattened into a single recording"

        # get data
        data = self.data[0]

        # iterate over channels

        pca_components = []
        explained_variance_ratio = []
        projections = []
        pca_models = []

        n_sequences, n_samples, n_channels = data.shape
        for i in range(n_channels):
            # perform pca
            data_traces = data[:, :, i]
            pca = PCA(n_components=n_components, **pca_kwargs)
            pca.fit(data_traces)
            pca_components.append(pca.components_)
            explained_variance_ratio.append(pca.explained_variance_ratio_)
            projections.append(pca.transform(data_traces))
            pca_models.append(pca)

        # add pca metadata
        self.metadata[0]["explained_variance_ratio"] = np.stack(explained_variance_ratio, axis=-1)
        self.metadata[0]["pca_components"] = np.stack(pca_components, axis=-1)
        self.metadata[0]["original_data"] = data[:, :, :]
        self.metadata[0]["pca_model"] = pca_models

        self.data[0] = np.stack(projections, axis=-1)

        # assert shapes
        assert self.metadata[0]["explained_variance_ratio"].shape == (n_components, n_channels),\
            "explained_variance_ratio must have shape [n_components, n_channels]"
        assert self.metadata[0]["pca_components"].shape == (n_components, n_samples, n_channels),\
            "pca_components must have shape [n_components, n_samples, n_channels]"
        assert self.metadata[0]["original_data"].shape == (n_sequences, n_samples, n_channels),\
            "original_data must have shape [n_sequences, n_samples, n_channels]"
        assert self.data[0].shape == (n_sequences, n_components, n_channels),\
            "projected data must have shape [n_sequences, n_components, n_channels]"

        return self

    def combine_features(self) -> Preprocessor:
        """
        Combines the features extracted by dwt and psd into a 2D array of [n_sequences, n_features]
        and applies z-normalization.

        Returns:
            Preprocessor: Preprocessor

        """
        assert len(self.data[0].shape) == 3, "Only works with realtime - 2D data suggests you are doing offline."
        assert "approx_coeffs" in self.metadata[0].keys() and "psd" in self.metadata[0].keys(), "No metadata - first run dwt & psd!."
        n_sequences = self.data[0].shape[0]
        for i, data in tqdm(enumerate(self.data), total=len(self.data)):
            if data is not None:

                #z-normalize approx_coeffs
                self.metadata[i]["approx_coeffs"] = (self.metadata[i]["approx_coeffs"]-np.mean(self.metadata[i]["approx_coeffs"], axis=0))/np.std(self.metadata[i]["approx_coeffs"], axis=0)

                #z-normalize list of detail coefficients 
                detail_coeffs = self.metadata[i]["detail_coeffs"]
                for j, detail_coeff in enumerate(detail_coeffs):
                    #avoid division by zero
                    coeff_std = np.std(detail_coeff, axis=0)
                    coeff_std[coeff_std == 0] = 1 #avoid division by zero
                    detail_coeffs[j] = (detail_coeff-np.mean(detail_coeff, axis=0))/coeff_std
                self.metadata[i]["detail_coeffs"] = detail_coeffs

                #combine approx_coeffs and detail_coeffs into a single array
                vec = np.hstack((self.metadata[i]["approx_coeffs"], np.concatenate(self.metadata[i]["detail_coeffs"], axis=-2))).reshape(n_sequences, -1)
                
                #z-normalize psd and combine with dwt features
                psd_std = np.expand_dims(np.std(self.metadata[i]["psd"],axis=-1), axis=-1) #psd standard deviation
                psd_std[psd_std == 0] = 1 #avoid division by zero
                psd = np.divide((self.metadata[i]["psd"]-np.expand_dims(np.mean(self.metadata[i]["psd"],axis=-1), axis=-1)), psd_std) #z-normalize psd
                self.data[i] = np.hstack((vec, psd.reshape(n_sequences, -1))) #combine dwt and psd features
        return self

    def apply(self, preprocessing_params: Dict[str, Any]) -> Preprocessor:
        """
        Applies the preprocessing pipeline defined in preprocessing_params
        where each key must match a function name in the Preprocessor class
        and the value must be a dictionary of keyword arguments for the function.
        """
        preprocessor = self
        order = preprocessing_params["order"] if "order" in preprocessing_params else list(preprocessing_params.keys())
        for function_name in order:
            kwargs = preprocessing_params[function_name]
            preprocessor = getattr(preprocessor, function_name)(**kwargs)

        return preprocessor
