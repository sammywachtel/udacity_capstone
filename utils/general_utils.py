import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_files
from keras.utils import np_utils
import tensorflow as tf
import numpy as np
from keras.preprocessing import image as kimage                  
from tqdm import tqdm


def get_tfrecord_count(tfrecord_dataset):
    num_records = 0
    for r in tf.python_io.tf_record_iterator(tfrecord_dataset):
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

