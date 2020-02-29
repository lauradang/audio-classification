import pickle
import os
import numpy as np 
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score

def build_predictions(clean_audio_dir, config, model, categories, labeled_category_file):
    y_actual = []
    y_pred = []
    file_category_prob = {}

    for _file in tqdm(os.listdir(clean_audio_dir)):
        rate, wav = wavfile.read(os.path.join(clean_audio_dir, _file))
        category = labeled_category_file[_file]
        category_idx = categories.index(category) # Get index of the category

        y_prob = []

        # wav.shape[0] is the length of the audio file
        for i in range(0, wav.shape[0] - config.step, config.step):
            sample = wav[i:i+config.step]

            X_sample = mfcc(
                sample, 
                rate, 
                numcep=config.nfeat,
                nfilt=config.nfilt,
                nfft=config.nfft
            ).T

            X_sample = (X_sample - config.min) / (config.max - config.min) # denominator is the range

            if config.mode == "conv":
                X = X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], 1)
            elif config.mode == "recurrent":
                X = X_sample.reshape(1, X_sample.shape[1], X_sample.shape[0])

            y_hat = model.predict(X)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_actual.append(category_idx)

        file_category_prob[_file] = np.mean(y_prob, axis=0).flatten()

    return y_actual, y_pred, file_category_prob

def main():
    conv_config = pickle.load(open("pickles/conv_config.pickle", "rb"))
    recurrent_config = pickle.load(open("pickles/recurrent_config.pickle", "rb"))

    configs = [conv_config, recurrent_config]

    for config in configs:        
        df = pd.read_csv("instruments.csv")
        categories = list(np.unique(df.label))
        labeled_category_file = dict(zip(df.fname, df.label))

        model = load_model(f"saved_models/{config.mode}_model")
        y_actual, y_pred, file_category_prob = build_predictions("clean", config, model, categories, labeled_category_file)
        acc_score = accuracy_score(y_true=y_actual, y_pred=y_pred)

        y_probs = []

        for i, row in df.iterrows():
            y_prob = file_category_prob[row.fname]
            y_probs.append(y_prob)
            for category, prob in zip(categories, y_prob):
                df.at[i, category] = prob

        y_pred = [categories[np.argmax(prob)] for prob in y_probs]
        df.insert(2, "predicted", y_pred)

        df.to_csv(f"predictions/predictions_{config.mode}.csv")

if __name__ == "__main__":
    main()
