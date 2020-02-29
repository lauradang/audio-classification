# Intrument Classification

A machine learning classification project that predicts the instrument being played in a particular wavfile. Specifically, we visualize, clean, and format the data to feed them through either a convolutional or recurrent neural network for predictions. 

## Steps to Run:
1. Install dependencies using `pip install -r requirements.txt`.
2. Run `visualizing.ipynb` Jupyter Notebook to produce a `clean` directory which contains the cleaned wavfiles.
2. Run `prepare_data.py` to produce a `pickles` directory with two `config` objects (one for the CNN and one for the RNN).
3. Run `model.py` to train the data on the models and produces a `saved_models` directory with the CNN and the RNN saved.
4. Run `predict.py` to predict what instrument is being played in the respective wavfile and produces a `predictions` directory containing 2 CSVs of predictions.

## Mainly Built With

* [Librosa](https://librosa.github.io/librosa/) - For retrieving audio sample rates
* [Tensorflow - Keras](https://www.tensorflow.org/guide/keras) - Used to create CNN and RNN and train data
* [Pandas](https://pandas.pydata.org/) - Cleaning and formatting data

## Author

Laura Dang

