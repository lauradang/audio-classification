"""The script that defines and trains the model."""

import os
import pickle
import argparse
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential

def get_conv_model(input_shape):
    """
    The convolutional neural network's model architecture.

    Arguments:
        input_shape {(int, int, int)} -- The shape of the input to the model.

    Returns:
        Keras model -- The compiled CNN.
    """
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation="relu", strides=(1, 1), padding="same", input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation="relu", strides=(1, 1), padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", strides=(1, 1), padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", strides=(1, 1), padding="same"))

    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["acc"])

    return model

def main(pickle_dir, epochs, batch_size, output_dir):
    """The main entrypoint of the script."""
    model_pkl = pickle.load(open(f"{pickle_dir}/conv_config.pickle", "rb"))

    X = model_pkl.data[0]
    y = model_pkl.data[1]

    y_flat = np.argmax(y, axis=1)
    class_weight = compute_class_weight("balanced", np.unique(y_flat), y_flat)

    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model(input_shape)

    model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=True, class_weight=class_weight)
    model.save(os.path.join(output_dir, f"conv_model")) 


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="A script that defines and trains the convolutional neural network.")
    arg_parser.add_argument("--pickle_dir", type=str, default="pickles", help="The directory to that contains the pickle files of training data.")    
    arg_parser.add_argument("--epochs", type=int, default=10, help="The number of epochs the model is trained for.")    
    arg_parser.add_argument("--batch_size", type=int, default=32, help="The batch size that the model is trained with.")    
    arg_parser.add_argument("--output_dir", type=str, default="saved_models", help="The directory where the trained model is outputted.")    
    args = arg_parser.parse_args()

    main(args.pickle_dir, args.epochs, args.batch_size, args.output_dir)
