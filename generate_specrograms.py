from utils import audio_utils

test_spectrogram_folder = 'data/nsynth-test-mel-spectrograms-064'
valid_spectrogram_folder = 'data/nsynth-valid-mel-spectrograms-064'
train_spectrogram_folder = 'data/nsynth-train-mel-spectrograms-064'

#regex_filter = None
regex_filter = '.*-064-.*'


audio_utils.write_spectograms_parallelized('data/nsynth-test.tfrecord', test_spectrogram_folder, 
                                           batch_size=200, mel=True, overwrite=False, regex_filter=regex_filter)

audio_utils.write_spectograms_parallelized('data/nsynth-valid.tfrecord', valid_spectrogram_folder, 
                                           batch_size=200, mel=True, overwrite=False, regex_filter=regex_filter)

## following took about 4 or 5 hours with full dataset
audio_utils.write_spectograms_parallelized('data/nsynth-train.tfrecord', train_spectrogram_folder, 
                                           batch_size=200, mel=True, overwrite=False, regex_filter=regex_filter)