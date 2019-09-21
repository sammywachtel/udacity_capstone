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

def graph_wav(wave_data, wav_name):
    fig, axs = plt.subplots(1, 1, figsize=(10, 2))
    print('plotting...')
    axs.plot(wave_data);
    axs.set_title('Audio Signal for: {}'.format(wav_name))
    plt.show()

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
    return batch_num * batch_size + batch_i_num+1, batch_num, num_samples

def create_spectrogram_parallelized_callback(ret):
    if ret[0]%50 == 0:
        print('\rprocessed {} files out of {} in {} batches'.format(ret[0], ret[2], ret[1]), end="")


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
                [pool.apply_async(create_spectrogram_parallelized, args=(batch['audio'][i], 64000,
                                                                   batch['note_str'][i].decode('utf-8'), directory,
                                                                   batch_num, i, batch_size, num_samples),
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
                    