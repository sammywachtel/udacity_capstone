import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.preprocessing import image as kimage                  
from tqdm import tqdm
from utils import general_utils
from math import ceil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import seaborn as sn
import keras


# The feature dictionary for the NSynth libarary dataset
feats = {
    "note_str": tf.FixedLenFeature([], dtype=tf.string),
    "sample_rate": tf.FixedLenFeature([], dtype=tf.int64),
    "instrument_source": tf.FixedLenFeature([1], dtype=tf.int64),
    "qualities": tf.FixedLenFeature([10], dtype=tf.int64),
    "instrument_source_str": tf.FixedLenFeature([], dtype=tf.string),
    "audio": tf.FixedLenFeature([64000], dtype=tf.float32),
    "instrument": tf.FixedLenFeature([], dtype=tf.int64),
    #"qualities_str": tf.FixedLenFeature([], dtype=tf.string),
    "note":tf.FixedLenFeature([1], dtype=tf.int64),
    "instrument_str": tf.FixedLenFeature([], dtype=tf.string),
    "instrument_family_str": tf.FixedLenFeature([], dtype=tf.string),
    "velocity": tf.FixedLenFeature([1], dtype=tf.int64),
    "pitch": tf.FixedLenFeature([1], dtype=tf.int64),
    "instrument_family": tf.FixedLenFeature([1], dtype=tf.int64)
}

def get_tfrecord_count(tfrecord_dataset):
    num_records = 0
    for r in tf.python_io.tf_record_iterator(tfrecord_dataset):
        num_records += 1
        
    return num_records

def get_tfdataset_count(dataset):
    d = dataset.batch(batch_size=50)
    itr = d.make_one_shot_iterator()
    batch_itr = itr.get_next()
    with tf.Session() as sess:
        num_records = 0
        #loop through all batches
        while True:
            try:
                # sess.run returns a dict
                batch = sess.run(batch_itr)
            except tf.errors.OutOfRangeError:
                break

            for i in batch['note_str']:
                num_records += 1
                
    return num_records

def load_dataset(path):
    data = load_files(path, shuffle=True, load_content=False)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 10)
    target_names = np.array(data['target_names'])[np.array(data['target'])]

    return files, targets, target_names

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = kimage.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = kimage.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def get_label_dict_from_generator(generator):   
    # for use later, let's get the entire dictionary of data labels 
    #  looks like: 
    #     {'bass': 0, 'brass': 1, 'flute': 2, 'guitar': 3, 'keyboard': 4, 'mallet': 5, 
    #      'organ': 6, 'reed': 7, 'string': 8, 'vocal': 9}
    data_class_indices = generator.class_indices
    
    # reverse the dictionary so the keys are the values and the values are the kees so 
    #   we can look up label names by label number
    data_labels_dict = dict((v,k) for k, v in data_class_indices.items())
    
    return data_labels_dict

def run_prediction(model, image_generator):
    #reset the image generator first
    image_generator.reset()

    data_labels_dict = get_label_dict_from_generator(image_generator)

    # Run the test prediction
    ## each record returned from predict_generator contains the predictions for each lable for one sound
    ## looks like: [0.06944881 0.01173394 0.11198635 0.05482391 0.04346911 0.11438505 
    ##              0.06325968 0.02916289 0.00357502 0.49815515]
    y_pred_res = model.predict_generator(image_generator)

    # to get results, we'll get the original y labels and the predicted y labels and compare them

    #### first get the predicted labels
    ## get the label index (translating from lists of probabilities using argmax)
    y_predicted = np.argmax(y_pred_res, axis=1)
    
    #### now get the original y labels from the ImageGenerator
    y_actual = np.array([l for l in image_generator.labels])
    
    #print('y_actual',len(y_actual),'y_predicted',len(y_predicted))

    # calculate the results! and also return the actual and predicted labels
    return np.equal(y_actual, y_predicted), y_predicted, y_actual
    
def list_data_in_tfrecord(tfrecord_file, num_records_to_list=1):
    tf.enable_eager_execution()

    #testData = tf.data.TFRecordDataset("data/nsynth-test.tfrecord")
    res = tf.python_io.tf_record_iterator(tfrecord_file)

    i = 0
    for e in res:
        print("\nRECORD ", i+1, "\n")
        e = tf.train.Example.FromString(e)
        d = dict(e.features.feature)
        keys = d.keys()
        for key in keys:
            if key == 'audio':
                print(key)
                print('omitted\n')
            else:
                print(key)
                print (d.get(key))

        i += 1
        if i > num_records_to_list-1:
            break

def get_data_from_tfrecord_by_note_str(note_str, tfrecord_file_name='data/nsynth-test.tfrecord'):
    num_records = general_utils.get_tfrecord_count(tfrecord_file_name)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Read data
            dataset = tf.data.TFRecordDataset(tfrecord_file_name)

            parse_func = lambda example_proto: tf.parse_single_example(example_proto, feats)
            dataset = dataset.map(parse_func)

            # need to do this in batches as tfrecords are streamed
            batch_size = 1
            dataset = dataset.batch(batch_size=batch_size)
            itr = dataset.make_one_shot_iterator()

            batch = itr.get_next()
            
            #loop through all batches
            for i in range(num_records):
                ret = sess.run(batch)
                note_string_name = ret['note_str'][0].decode('utf-8')
                if note_string_name == note_str:
                    print('found the file: ', note_string_name)
                    return ret

def display_image(image_path):
    plt.axis('off')
    image = mpimg.imread(image_path)
    imgplot = plt.imshow(image)
    plt.show()
    
def display_confusion(y_predicted, y_actual, labels_dict):

    p = np.vectorize(labels_dict.get)(y_predicted)
    a = np.vectorize(labels_dict.get)(y_actual)
    
    data = {'y_Predicted': p,
            'y_Actual':    a
            }
    
    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    sn.heatmap(confusion_matrix, annot=True)
    
def visualize_model_filters(model, layer_number=1, n_filters=6, size=1):
    
    #print(model.layers[0].get_weights())
    model_layer = model.layers[layer_number]
    
    # Return message if this is not a convolutional later.
    if 'con' not in model_layer.name:
        print('Layer {} is not a convolutional layer.'.format(layer_number))
        return
    
    filters, biases = model_layer.get_weights()
    
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    plt.figure(num=None, figsize=(8, 10), dpi=80)
    
    n_rows, index = n_filters, 1
    for i in range(n_filters):
        f = filters[:, :, :, i]
        for j in range(3):
            ax = plt.subplot(n_rows, 3, index)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(f[:, :, j])
            index += 1
    # show the figure
    plt.show()
    
def print_model_conv_layers(model, conv_only=False):
    i = 0
    for layer in model.layers:
        shape = 0,0,0
        is_con = False
        if 'con' in layer.name:
            #continue
            filters, biases = layer.get_weights()
            shape = filters.shape
            is_con = True

        i += 1

        if is_con or not conv_only:
            print('layer', i-1, layer.name, shape)    
        
def clear_keras_sessions():
    keras.backend.clear_session()