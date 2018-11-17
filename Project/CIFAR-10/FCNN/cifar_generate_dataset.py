import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Generate standard tfrecord of training or testing set
image_train_path = '../cifar-10/train/'
label_train_path = '../cifar-10/cifar_train.txt'
image_test_path = '../cifar-10/test/'
label_test_path = '../cifar-10/cifar_test.txt'
data_path = './data/'
tfRecord_train = './data/cifar_train.tfrecords'
tfRecord_test = './data/cifar_test.tfrecords'
resize_height = 32; resize_width = 32

# Generate tfRecord file
def write_tfRecordd(tfRecordName, image_path, label_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName) # Create a writer

    with open(label_path, 'r') as label_file:
        picfile_label_pair = label_file.readlines()
    
    for num, content in enumerate(picfile_label_pair):
        # Construct picture path
        picfile, label = content.split()
        pic_path = image_path + picfile

        img = Image.open(pic_path)
        img_raw = img.tobytes() # Transfer image into bytes
        # One-hot encode: transfer label e.g. 3 -> 0001000000
        labels = [0] * 10
        labels[int(label)] = 1

        # Create an example
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        })) # warp image and label data

        writer.write(example.SerializeToString()) # serialize the example

        #print("finish processing number of picture: ", num + 1)

    writer.close()

    #print("write tfRecord successfully")
        
def generate_tfRecord():
    if not os.path.exists(data_path):
        # if the folder doesn't exist then mkdir
        os.makedirs(data_path)
    else:
        print("Directory has already existed")
    
    # Generate training set
    print("Generating training set...")
    write_tfRecordd(tfRecord_train, image_train_path, label_train_path)
    # Generate test set
    print("Generating test set...")
    write_tfRecordd(tfRecord_test, image_test_path, label_test_path)

def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path])

    reader = tf.TFRecordReader() # Create a reader
    serialized_example = reader.read(filename_queue)[1] # store samples
    
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([10], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)
    })

    img = tf.decode_raw(features['img_raw'], tf.uint8) # Decode img_raw into unsigned int
    img.set_shape([resize_height* resize_width * 3]) # Reshape image into a row of 32*32 *3 pixel (because of RGB image)
    img = tf.cast(img, tf.float32) * (1/255) # Normalize image into float

    label = tf.cast(features['label'], tf.float32) # Transfer label into float

    return img, label

# Construct a batcher (generator)
def get_tfRecord(num, getTrain=True):
    if getTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    
    img, label = read_tfRecord(tfRecord_path)

    # Shuffle the image order
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=num, num_threads=2, capacity=1000, min_after_dequeue=700)

    return img_batch, label_batch

def main():
    generate_tfRecord()

if __name__ == '__main__':
    main()
