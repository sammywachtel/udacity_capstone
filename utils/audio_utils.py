import numpy as np
import wave, sys, pyaudio
import math
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
import multiprocessing as mp
import time
from utils import general_utils

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

def graph_wav(wave_data, wav_name, wav_num):
    fig, axs = plt.subplots(1, 1, figsize=(10, 2))
    print('plotting...')
    axs.plot(wave_data);
    axs.set_title('Audio Signal for image {}: {}'.format(wav_num, wav_name))
    #axs[1].plot(encoding[0]);
    #axs[1].set_title('NSynth Encoding')
    plt.show()
    #plt.close(fig)
    
def create_spectrogram(audio, sample_rate, name, directory):
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    spect = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(spect, ref=np.max))
    #plt.colorbar(format='%+2.0f dB')
    #plt.title('Mel Spectrogram')
    filename  = directory + '/' + name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,fig,ax,spect
    
def write_spectograms(tfrecord_file_name, directory, batch_size = 50):
    num_tfrecords = general_utils.get_tfrecord_count(tfrecord_file_name)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Read data

            dataset = tf.data.TFRecordDataset(tfrecord_file_name)

            # make the tensor structure
            feats = {
                "note_str": tf.FixedLenFeature([], dtype=tf.string),
                "audio": tf.FixedLenFeature([64000], dtype=tf.float32)
            }

            parse_func = lambda example_proto: tf.parse_single_example(example_proto, feats)
            dataset = dataset.map(parse_func)

            num_samples = 60#num_tfrecords #50 #set to not take all samples
            dataset = dataset.shuffle(buffer_size=num_tfrecords).take(num_samples)
            dataset = dataset.batch(batch_size=batch_size)
            itr = dataset.make_one_shot_iterator()

            batch_itr = itr.get_next()
            cnt = 0

            #loop through all batches
            for batch_num in range(math.ceil(num_samples/batch_size)):
                # sess.run returns a dict
                batch = sess.run(batch_itr)

                item_cnt_in_batch = len(batch[list(batch.keys())[0]])
                for i in range(item_cnt_in_batch):
                    name = batch['note_str'][i].decode('utf-8')
                    print('\rfile count: {} out of {}, batch: {}, batch_item: {}, note_str: {}'.format(cnt, num_samples, batch_num, i, name), end="")
                    #graph_wav(batch['audio'][i], name, cnt)
                    create_spectrogram(batch['audio'][i], 64000, name, directory)
                    cnt += 1

def create_spectrogram_parallelized(audio, sample_rate, audioname, directory,
                                    batch_num, batch_i_num, batch_size, num_samples):
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    spect = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(spect, ref=np.max))
    #plt.colorbar(format='%+2.0f dB')
    #plt.title('Mel Spectrogram')
    filename  = directory + '/' + audioname + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,fig,ax,spect
    return batch_num * batch_size + batch_i_num+1, batch_num

def create_spectrogram_parallelized_callback(files):
    if files[0]%50 == 0:
        print('\rprocessed {} files in {} batches'.format(files[0], files[1]), end="")
    
def write_spectograms_parallelized(tfrecord_file_name, directory, batch_size = 50):
    num_tfrecords = general_utils.get_tfrecord_count(tfrecord_file_name)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Read data

            dataset = tf.data.TFRecordDataset(tfrecord_file_name)

            # make the tensor structure
            feats = {
                "note_str": tf.FixedLenFeature([], dtype=tf.string),
                "audio": tf.FixedLenFeature([64000], dtype=tf.float32)
            }

            parse_func = lambda example_proto: tf.parse_single_example(example_proto, feats)
            dataset = dataset.map(parse_func)

            num_samples = num_tfrecords #50 #set to not take all samples
            dataset = dataset.shuffle(buffer_size=num_tfrecords).take(num_samples)
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
                [pool.apply_async(create_spectrogram_parallelized, args=(batch['audio'][i], 64000,
                                                                   batch['note_str'][i].decode('utf-8'), directory,
                                                                   batch_num, i, batch_size, num_samples),
                                  callback=create_spectrogram_parallelized_callback
                                  ) for i in range(item_cnt_in_batch)]
                pool.close()
                pool.join()
            
def test_func(note_str, audio, batch_num, batch_size, num_samples):
    print(note_str, audio, batch_num, batch_size, num_samples)