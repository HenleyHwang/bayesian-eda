import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import firwin, group_delay, lfilter


class SignalData:
    """
    Array-like class to preprocess and store signal data.
    The preprocesed data is stored in the `data` attribute as 1D ndarray.
    """

    def __init__(
        self,
        raw_data: np.ndarray,
        original_frequency: int = 200,
        downsampled_frequency: int = 4,
        lowpass_cutoff_frequency: int = 2,
        lowpass_filter_order: int = 4096,
        outlier_window_duration: float = 3,
    ):
        self.raw_data = raw_data
        self.original_frequency = original_frequency
        self.downsampled_frequency = downsampled_frequency

        self.preprocess(lowpass_cutoff_frequency, lowpass_filter_order, outlier_window_duration)

    def preprocess(self, lowpass_cutoff_frequency, lowpass_filter_order, outlier_window_duration):
        self.data = self.raw_data.copy()
        # 1. Low-pass filter to remove high-frequency noise
        self.data = low_pass_filter(self.data, self.original_frequency, lowpass_cutoff_frequency, lowpass_filter_order)
        # 2. Downsample the signal to reduce computational cost
        self.data = downsample(self.data, self.original_frequency, self.downsampled_frequency)
        # 3. Fill outliers to remove artifacts
        self.data = fill_outliers(self.data, self.downsampled_frequency, outlier_window_duration)

    @property
    def t(self):
        """
        Discrete time ticks corresponding to the signal data.
        """
        return np.arange(len(self)) * self.dt

    @property
    def T(self):
        """
        Total duration of the signal data in seconds.
        """
        return len(self) * self.dt

    @property
    def dt(self):
        """
        Time step between each signal data point in seconds.
        """
        return 1 / self.f

    @property
    def f(self):
        """
        Frequency of the signal data in Hz.
        """
        return self.downsampled_frequency

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]


class InputsData(SignalData):
    def __init__(self, raw_data, original_frequency=200, downsampled_frequency=4):
        self.raw_data = raw_data
        self.original_frequency = original_frequency
        self.downsampled_frequency = downsampled_frequency

        self.preprocess()

    def preprocess(self):
        self.data = self.raw_data.copy()
        # 1. Downsample the inputs to match the signal
        self.data = maxpool(self.data, self.original_frequency, self.downsampled_frequency)


def low_pass_filter(signal, signal_frequency, cutoff_frequency, filter_order):
    fir_filter = firwin(filter_order + 1, cutoff_frequency, fs=signal_frequency)
    mean_group_delay = group_delay((fir_filter, 1.0))[1].mean().round().astype(int)
    padded_signal = np.pad(signal, mean_group_delay, mode="edge")
    filtered_signal = lfilter(fir_filter, 1.0, padded_signal)
    filtered_signal = filtered_signal[2 * mean_group_delay :]
    return filtered_signal


def downsample(signal, original_frequency, downsampled_frequency):
    if original_frequency % downsampled_frequency != 0:
        raise ValueError("Downsampled frequency must be a divisor of the original frequency")
    downsampling_rate = original_frequency // downsampled_frequency
    downsampled_signal = signal[::downsampling_rate]
    return downsampled_signal


def fill_outliers(signal, signal_frequency, window_duration):
    window_size = int(window_duration * signal_frequency)
    padded_signal = np.pad(signal, (window_size // 2, (window_size - 1) // 2), mode="edge")
    windows = sliding_window_view(padded_signal, window_shape=window_size)
    window_medians = np.median(windows, axis=1)
    window_mads = np.median(np.abs(windows - window_medians[:, None]), axis=1)
    outlier_indices = np.abs(signal - window_medians) > 3 * window_mads
    signal[outlier_indices] = window_medians[outlier_indices]
    return signal


def maxpool(signal, original_frequency, downsampled_frequency):
    if original_frequency % downsampled_frequency != 0:
        raise ValueError("Downsampled frequency must be a divisor of the original frequency")

    window_size = original_frequency // downsampled_frequency
    num_pools = len(signal) // window_size

    output = np.zeros(num_pools)
    for i in range(num_pools):
        window = signal[i * window_size : (i + 1) * window_size]
        output[i] = np.max(window)

    # Handle the last window
    if len(signal) % window_size != 0:
        last_window = signal[num_pools * window_size :]
        output = np.append(output, np.max(last_window))

    return output
