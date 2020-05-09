"""The script that prepares the data by building the proper features for the model to be trained."""

import os
import argparse
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
from scipy.io import wavfile
from keras.utils import to_categorical
from python_speech_features import mfcc
from model_config import Model

def build_features(df, clean_directory, output_dir, model):
    """
    Building the MFCC features to train the model.

    Arguments:
        df {pd.DataFrame} -- The dataframe of the training data.
        clean_directory {str} -- The directory that contains the clean wavfiles.
        output_dir {str} -- The directory to output the pickle files.
        model {Model} -- An object from the Model() class.

    Returns:
        (np.array, np.array) -- A tuple of the numpy arrays of the X and y features.
    """
    categories = list(np.unique(df.label))
    categories_dist = df.groupby(["label"])["length"].mean()

    num_samples = 2 * int(df.length.sum() / model.sample_size)
    prob_dist = categories_dist / categories_dist.sum()

    X = []
    y = []

    for i in tqdm(range(num_samples)):
        rand_category = np.random.choice(categories, p=prob_dist)

        file = np.random.choice(df[df.label == rand_category].fname)
        rate, wav = wavfile.read(os.path.join(clean_directory, file))

        if wav.shape[0] < model.step:
            continue
        
        random_idx = np.random.randint(0, wav.shape[0] - model.step)
        sample = wav[random_idx:random_idx + model.step]

        X_sample = mfcc(sample, rate, numcep=model.nfeat, nfilt=model.nfilt, nfft=model.nfft).T 
    
        model.min = min(np.amin(X_sample), model.min)
        model.max = max(np.amax(X_sample), model.max)

        X.append(X_sample if model.mode == "conv" else X_sample.T)
        y.append(categories.index(rand_category))
    
    X_array, y_array = np.array(X), np.array(y)
    X = (X_array - model.min) / (model.max - model.min)

    X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = to_categorical(y, num_classes=10)

    model.data = (X_reshaped, y)
    dump_pickle(model, output_dir)

    return X_reshaped, y

def dump_pickle(model, output_dir):
    """
    Saves the Model() object in a pickle file.

    Arguments:
        model {Model} -- An object from the Model() class.
        output_dir {str} -- The directory where the pickle files are outputted.
    """
    pickle_out = open(f"{output_dir}/{model.mode}_config.pickle", "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()

def main(csv_filename, clean_directory, output_dir):
    """The main entrypoint of the script."""
    df = pd.read_csv(csv_filename)
    model = Model()
    build_features(df, clean_directory, output_dir, model)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="A script to prepare the training data for the model.")
    arg_parser.add_argument("--csv_filename", type=str, default="data/train.csv", help="A comma-separated value file that contains the raw training data.")
    arg_parser.add_argument("--clean_directory", type=str, default="clean_wavfiles", help="The directory that contains the cleaned wavfiles.")    
    arg_parser.add_argument("--output_dir", type=str, default="pickles", help="The directory to output the pickle files that contain the training data.")    
    args = arg_parser.parse_args()

    main(args.csv_filename, args.clean_directory, args.output_dir)
