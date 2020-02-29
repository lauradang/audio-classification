import os
import pickle
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from prepare_data import configs

def get_conv_model(input_shape):
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
    model.add(Dense(10, activation="softmax")) # softmax layer outputs 1x10 array (All these values add up to 1)
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["acc"])

    return model

def get_rec_model(input_shape):
    # shape of RNN is (n, time, feat) - which is why we transposed
    model = Sequential()
    # import pdb; pdb.set_trace()
    # Costly to add many LSTM layers (backpropogation is more costly than convolutional layer)
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape)) # Comparable to a Dense layer
    model.add(LSTM(128, return_sequences=True)) # Comparable to a Dense layer
    
    # Keep consistent with convolutional
    model.add(Dropout(0.5))

    # We have a time component so we can add time distributed layer
    # Must be more careful with how we flatten/dense these layers
    # it may not look like much (64, 32, 16, 8..) but these numbers are multipled by the time dimension in the input shape
    model.add(TimeDistributed(Dense(64, activation="relu"))) 
    model.add(TimeDistributed(Dense(32, activation="relu"))) 
    model.add(TimeDistributed(Dense(16, activation="relu"))) 
    model.add(TimeDistributed(Dense(8, activation="relu"))) 

    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))

    model.summary
    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["acc"])

    return model

def main():
    X_conv = conv_config.data[0]
    y_conv = conv_config.data[1]

    X_recurrent = recurrent_config.data[0]
    y_recurrent = recurrent_config.data[1]

    for config in configs:
        if config.mode == "conv":
            X = X_conv
            y = y_conv
        elif config.mode == "recurrent":
            X = X_recurrent
            y = y_recurrent

        y_flat = np.argmax(y, axis=1)
        class_weight = compute_class_weight(
            "balanced",
            np.unique(y_flat),
            y_flat
        )

        if config.mode == "conv":
            input_shape = (X.shape[1], X.shape[2], 1)
            model = get_conv_model(input_shape)
        elif config.mode == "recurrent":
            X = X_recurrent
            input_shape = (X.shape[1], X.shape[2])
            model = get_rec_model(input_shape)

        model.fit(X, y, epochs=10, batch_size=32, shuffle=True, class_weight=class_weight)
        model.save(os.path.join("saved_models", f"{config.mode}_model")) 

if __name__ == "__main__":
    conv_config = pickle.load(open("pickles/conv_config.pickle", "rb"))
    recurrent_config = pickle.load(open("pickles/recurrent_config.pickle", "rb"))

    main()
