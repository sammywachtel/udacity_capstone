import warnings
warnings.filterwarnings('ignore')
import numpy as np
import wave, sys, pyaudio
import math
import matplotlib.pyplot as plt
import librosa
import librosa.display
from librosa.core import load
import tensorflow as tf
import multiprocessing as mp
import time
from utils import general_utils
import importlib
import os

def play_wav(wf):
    paud = pyaudio.PyAudio()
    chunk = 1024
    stream = paud.open(format = paud.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)
    data = wf.readframes(chunk)
    while data != '':
        stream.write(data)
        data = wf.readframes(chunk)
    
def graph_waveform(wave_data, wav_name, sample_rate):
    fig, axs = plt.subplots(1, 1, figsize=(10, 2))
    axs.set_title('Waveform for: {}'.format(wav_name))
    librosa.display.waveplot(wave_data, sr=sample_rate)

def graph_stft(wave_data, wav_name):
    fig, axs = plt.subplots(1, 1, figsize=(10, 2))
    axs.set_title('Short-time Fourier Transform for: {}'.format(wav_name))
    n_fft = 2048
    data = np.abs(librosa.stft(wave_data[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
    plt.plot(data)

def graph_time_stft(wave_data, wav_name, sample_rate):
    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    axs.set_title('Short-time Fourier Transform over Time for: {}'.format(wav_name))
    n_fft = 2048
    hop_length = 512
    stft = np.abs(librosa.stft(wave_data, n_fft=n_fft, hop_length=hop_length))
    db = librosa.amplitude_to_db(stft, ref=np.max)
    librosa.display.specshow(db, sr=sample_rate, x_axis='time', y_axis='linear')
    plt.colorbar()

#def create_spectrogram_non_mel(wave_data, wav_name, sample_rate):
#    fig, axs = plt.subplots(1, 1, figsize=[5,4])
#    axs.set_title('Spectrogram for: {}'.format(wav_name))
#    hop_length = 512
#    n_fft = 2048
#    data = np.abs(librosa.stft(wave_data, n_fft=n_fft, hop_length=hop_length))
#    db = librosa.amplitude_to_db(data, ref=np.max)
#    librosa.display.specshow(db, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='hz')
#    plt.colorbar(format='%+2.0f dB')

def create_spectrogram(wave_data, wav_name, sample_rate, mel=False, spect_only=False, figsize=[5,4]):
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    
    if mel:
        spect_type = 'Mel Spectrogram'
        spect = librosa.feature.melspectrogram(y=wave_data, sr=sample_rate)
        db = librosa.power_to_db(spect, ref=np.max)
    else:
        spect_type = 'Spectrogram'
        #spect = np.abs(librosa.stft(wave_data))
        spect = np.abs(librosa.stft(wave_data, hop_length=512))
        db = librosa.amplitude_to_db(spect, ref=np.max)
    
    if spect_only:
        plt.axis('off')
        axs.set_frame_on(False)
        librosa.display.specshow(db, sr=sample_rate, x_axis='time', y_axis='hz')
    else:
        axs.set_title('{} for: {}'.format(spect_type,wav_name))
        librosa.display.specshow(db, sr=sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
    
    return plt
    
def create_mel_to_hz_plot(sample_rate):
    hop_length = 512
    n_fft = 2048
    n_mels = 20
    mels = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    plt.subplots(1, 1, figsize=[5,4])
    librosa.display.specshow(mels, sr=sample_rate, hop_length=hop_length, x_axis='linear')
    plt.ylabel('Mel filter')
    plt.colorbar()
    plt.title('Hz to Mels.')


def create_spectrogram_parallelized(audio, sample_rate, audioname, directory,
                                    batch_num, batch_i_num, batch_size, num_samples, mel, overwrite=False):
    # Since we are potentially returning in more than one place
    ret_val = batch_num * batch_size + batch_i_num+1, batch_num, num_samples
    
    # put together file name (will need the subdirectory below, so break it out)
    subdirectory = directory + '/' + audioname.split('_')[0] + '/'
    filename = subdirectory + audioname + '.jpg'
    
    # Skip the file if overwrite is False
    if not overwrite:
        if os.path.exists(filename):
            return ret_val
    
    # Create the subsirectory for the instrument family if it doesn't exist
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)
        
    plt = create_spectrogram(audio, audioname, sample_rate, mel=mel, spect_only=True, figsize=[0.72,0.72])
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    #fig.clf()
    #plt.close(fig)
    #plt.close('all')
    del plt
    return ret_val

def create_spectrogram_parallelized_callback(ret):
    if ret[0]%50 == 0 or ret[0] == ret[2]:
        print('\rprocessed {} files out of {} in {} batches'.format(ret[0], ret[2], ret[1]), end="")
        
def write_spectograms_parallelized(tfrecord_file_name, directory, batch_size = 50, mel=False, overwrite=False):
    num_tfrecords = general_utils.get_tfrecord_count(tfrecord_file_name)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Read data

            dataset = tf.data.TFRecordDataset(tfrecord_file_name)

            parse_func = lambda example_proto: tf.parse_single_example(example_proto, general_utils.feats)
            dataset = dataset.map(parse_func)

            #set to different number to take fewer samples for testing
            num_samples = num_tfrecords #50
            
            # no need to shuffle as we are only generating images
            #dataset = dataset.shuffle(buffer_size=num_tfrecords).take(num_samples)
            dataset = dataset.batch(batch_size=batch_size)
            itr = dataset.make_one_shot_iterator()

            batch_itr = itr.get_next()
                        
            #loop through all batches
            for batch_num in range(math.ceil(num_samples/batch_size)):
                # sess.run returns a dict
                batch = sess.run(batch_itr)

                item_cnt_in_batch = len(batch[list(batch.keys())[0]])
                #[pool.apply(create_spectrogram, args=(row['audio'], 64000, row['note_str'], directory)) for row in batch]
                #[print(type(row)) for row in batch]
                
                pool = mp.Pool(mp.cpu_count())
                [pool.apply_async(create_spectrogram_parallelized, args=(batch['audio'][i], batch['sample_rate'][i],
                                                                   batch['note_str'][i].decode('utf-8'), directory,
                                                                   batch_num, i, batch_size, num_samples, mel, overwrite),
                                  callback=create_spectrogram_parallelized_callback
                                  ) for i in range(item_cnt_in_batch)]
                pool.close()
                pool.join()

def get_audio_sample_by_name_from_tfrecord(note_str, tfrecord_file_name='data/nsynth-test.tfrecord'):
    ret = general_utils.get_data_from_tfrecord_by_note_str(note_str, tfrecord_file_name)    
    return ret['audio'][0], ret['sample_rate'], ret['note_str']

#works with any audio format
def get_sound_file_data(file_path):
    data, sample_rate = load(file_path)
    return data, sample_rate
                    