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

def run_prediction(model, image_generator, steps):
    # for use later, let's get the entire dictionary of data labels 
    #  looks like: 
    #     {'bass': 0, 'brass': 1, 'flute': 2, 'guitar': 3, 'keyboard': 4, 'mallet': 5, 
    #      'organ': 6, 'reed': 7, 'string': 8, 'vocal': 9}
    data_class_indices = image_generator.class_indices
    
    # reverse the dictionary so the keys are the values and the values are the kees so 
    #   we can look up label names by label number
    data_labels_dict = dict((f,i) for i, f in data_class_indices.items())

    # Run the test prediction
    ## each record returned from predict_generator contains the predictions for each lable for one sound
    ## looks like: [0.06944881 0.01173394 0.11198635 0.05482391 0.04346911 0.11438505 
    ##              0.06325968 0.02916289 0.00357502 0.49815515]
    test_scores_trained = model.predict_generator(image_generator, steps=steps)

    # to get results, we'll get the original y labels and the predicted y labels and compare them

    #### first get the predicted labels
    ## get the label index (translating from lists of probabilities using argmax)
    test_y_pred = np.argmax(test_scores_trained, axis=1)

    #### now get the original y labels from the ImageGenerator
    test_y_labels = np.array([l for l in image_generator.labels])

    # calculate the results!
    return test_y_labels == test_y_pred
    
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