# Using Deep Neural Network to Predict Musical Instrument Family

Goal: Use the NSynth Dataset by Google Inc. data to train a deep neural net to label the instrument playing a single note at any pitch or velocity.

## Status: 
This project is the first iteration of a Udacity Capstone Project. It is not complete and the best model tested predicts the instrument family of audio samples twice as well as random.

## Files:
- identify_instrument.ipynb: The main Jupyter notebook.
- models.py: Various versions (numbered) of the DNN models.
- utils/audio_utils.py: methods to work with audio, like converting audio files to spectrograms
- utils/general_utils.py: convenience methods to encapsulate functionality that surrounds working with tfrecord files and loading and manipulating data and running predictions.

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
