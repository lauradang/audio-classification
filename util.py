"""The utility functions used in the clenaing/visualization notebook."""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_data(data, title, cmap=None, interpolation=None):
    """
    Plots the given data.

    Arguments:
        data {dict} -- A dictionary with keys that contain the classes and values with numpy arrays.
        title {str} -- The title of the plot.

    Keyword Arguments:
        cmap {str} -- Defines the cmap type of the plot (default: {None}).
        interpolation {str} -- Defines the interpolation type of the plot (default: {None}).
    """
    num_rows = 2
    num_columns = 5

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(20, num_columns))
    fig.suptitle(title, size=16)
    class_idx = 0

    for x in range(num_rows):
        for y in range(num_columns):
            axes[x,y].set_title(list(data.keys())[class_idx])
            values = list(data.values())[class_idx]
            
            if cmap:
                axes[x,y].imshow(values, cmap=cmap, interpolation=interpolation)
            else:
                if len(values) == 2:
                    Y, freq = values[0], values[1]
                    axes[x,y].plot(freq, Y)
                else:
                    axes[x,y].plot(values)

            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            class_idx += 1
            