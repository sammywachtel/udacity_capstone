import tensorflow as tf

def get_tfrecord_count(tfrecord_dataset):
    num_records = 0
    for r in tf.python_io.tf_record_iterator(tfrecord_dataset):
        num_records += 1
        
    return num_records
    

    