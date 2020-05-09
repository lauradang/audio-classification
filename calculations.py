"""The calculations that produce the values in the visualization plots."""

import numpy as np 
import pandas as pd

def calc_fft(samples, sampling_rate):
    """
    Calculates magnitude and frequency of the given samples using fast Fourier transform.

    Arguments:
        samples {np.array} -- The samples retrieved from the wavfile.
        sampling_rate {int} -- The sampling rate in seconds from the wavfile.

    Returns:
        (np.array, np.array) -- A tuple of the magnitudes and frequencies of the given samples.
    """
    num_samples = len(samples)
    frequencies = np.fft.rfftfreq(num_samples, d=1/sampling_rate)
    magnitudes = abs(np.fft.rfft(samples) / num_samples)

    return (magnitudes, frequencies)    

def calc_envelope_signal(samples, sampling_rate, threshold):
    """
    Creates a boolean mask of the given samples that is determined using a threshold value.

    Arguments:
        samples {np.array} -- The samples retrieved from the wavfile.
        sampling_rate {int} -- The sampling rate in seconds from the wavfile.
        threshold {float} -- The threshold value that determines which sample values are dead sound space.

    Returns:
        np.array -- A boolean mask that represents which part of the wavfile should be kept.
    """
    samples = pd.Series(samples).apply(np.abs) # Convert numpy array to Series for easier data manipulation
    samples_mean = samples.rolling(int(sampling_rate / 10), min_periods=1, center=True).mean()

    return [True if mean > threshold else False for mean in samples_mean]
