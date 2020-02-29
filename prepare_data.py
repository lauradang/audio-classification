import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
from python_speech_features import mfcc
import librosa
import pickle
from plot import df_insert_length
from model_config import ModelConfig

config1 = ModelConfig(mode="conv")
config2 = ModelConfig(mode="recurrent")
configs = [config1, config2]

def build_features(config):
    df = df_insert_length("instruments.csv", "clean")
    categories = list(np.unique(df.label))
    categories_dist = df.groupby(["label"])["length"].mean()

    num_samples = 2 * int(df.length.sum() / config.sample_size)
    prob_dist = categories_dist / categories_dist.sum()

    X = []
    y = []

    for i in tqdm(range(num_samples)):
        rand_category = np.random.choice(categories, p=prob_dist)

        file = np.random.choice(df[df.label == rand_category].index)
        rate, wav = wavfile.read(os.path.join("clean", file))

        if wav.shape[0] < config.step:
            continue
        
        random_idx = np.random.randint(0, wav.shape[0] - config.step)
        sample = wav[random_idx:random_idx + config.step]

        X_sample = mfcc(
            sample, 
            rate, 
            numcep=config.nfeat,
            nfilt=config.nfilt,
            nfft=config.nfft
        ).T 
    
        config.min = min(np.amin(X_sample), config.min)
        config.max = max(np.amax(X_sample), config.max)

        X.append(X_sample if config.mode == "conv" else X_sample.T)
        y.append(categories.index(rand_category))
    
    X_array, y_array = np.array(X), np.array(y)
    X = (X_array - config.min) / (config.max - config.min)

    if config.mode == "conv":
        X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == "recurrent":
        X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        
    y = to_categorical(y, num_classes=10)

    config.data = (X_reshaped, y)
    dump_pickle(config)

    return X_reshaped, y

def dump_pickle(config):
    pickle_out = open(f"pickles/{config.mode}_config.pickle", "wb")
    pickle.dump(config, pickle_out)
    pickle_out.close()

def main():
    for config in configs:
        build_features(config)


if __name__ == "__main__":
    main()
