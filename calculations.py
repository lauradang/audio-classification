import numpy as np 
import pandas as pd

def calc_fft(samples, sampling_rate):
    window_length = len(samples)
    frequency = np.fft.rfftfreq(window_length, d=1/sampling_rate)
    magnitude = abs(np.fft.rfft(samples) / window_length)

    return (magnitude, frequency)

def calc_envelope_signal(samples, sampling_rate, threshold):
    # Convert numpy array to Series for easier data manipulation
    samples = pd.Series(samples).apply(np.abs)
    samples_mean = samples.rolling(
        int(sampling_rate/10), 
        min_periods=1,
        center=True
    ).mean()

    return [True if mean > threshold else False for mean in samples_mean]

