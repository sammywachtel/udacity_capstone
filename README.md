# Using Deep Neural Network to Predict Musical Instrument Family

Goal: Use the NSynth Dataset by Google Inc. data to train a deep neural net to predict the instrument family playing a single note.

## Status: 
This project is the first iteration of a Udacity Capstone Project. It is submitted, but will continue to evolve. Currently, the best model predicts the instrument family of audio samples twice as well as random.

## Files:
- instrument_identification.ipynb: The main implementation steps for this project. Each of the steps above are run from this file.
- visualizations.ipynb: Extra visualizations required for this report.
- models.py: The various versions of the CNN labeled, create_model_v1, create_model_v2, etc.
- generate_spectrograms.py​: A standalone script to generate spectrograms.
- data/: Data directory, but not included in the project. See NSynth link below to download directly from their page.
- saved_models/: The saved weights from the models in models.py, such as
weights.best.v1_1.hdf5, weights.best.v2_1.hdf5, etc
- utils/audio_utils.py: utility methods for working with and processing audio
- utils/general_utils.py: convenience methods

## Libraries Used
- sklearn
- Google Tensorflow
- Keras
- Numpy
- tqdm
- wave, pyaudio
- librosa (a python package for music and audio analysis)

## Dataset
NSynth: https://magenta.tensorflow.org/datasets/nsynth
