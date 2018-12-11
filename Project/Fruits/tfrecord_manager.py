import tensorflow as tf
import numpy as np

import os
import json

from functools import reduce

# These imports are relevant for displaying and encoding image strings.
# import base64
from PIL import Image

IMAGE_SHAPE = [100, 100, 3]
PREDICT_CLASSES = 83

DATA_FOLDER = './data/'

TRAIN_FILE = 'training.tfrecords'
TEST_FILE = 'test.tfrecords'

# Path to the json file
IMG_LABEL_JSON = DATA_FOLDER + 'ImgLabel.json'

# For Images
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# For Labels
def _int64_feature(onehot_list):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=onehot_list))

def create_example(image, label):
    """
    Creates a tf.Example message ready to be written to a file.
    
    Inputs:
        - image: image in binary format in the observation
        - label: label of the image
    """
    # Create a dictionary mapping the feature name
    # to the tf.Example-compatible data type.
    feature = {
        'image_raw': _bytes_feature(image),
        'label': _int64_feature(label),
    }
    # Create a Features message using tf.train.Example.
    return tf.train.Example(features=tf.train.Features(feature=feature))

def writeTFRecord(tfRecordName, imgFolder):
    # Write the tf.Example observations to test.tfrecords.
    writer = tf.python_io.TFRecordWriter(tfRecordName)

    classname_to_encode_dict = readImageLabelToDict()[0]
    for classname, label in classname_to_encode_dict.items():

        oneHot_labels = [0] * PREDICT_CLASSES
        oneHot_labels[label] = 1

        print('Processing', classname, '...', end=' ')
        for image_filename in os.listdir(os.path.join(imgFolder, classname)):
            image_path = os.path.join(imgFolder, classname, image_filename)

            # img_file = open(image_path, 'rb').read()
            # image_string = base64.b64encode(img_file)

            image = Image.open(image_path)
            image_string = image.tobytes()

            example = create_example(image_string, oneHot_labels)
            writer.write(example.SerializeToString())
        print('finished!')
    writer.close()

def testPrintTFRecord(tfrecord):
    """
    Read one image from TFRecord and display it.
    """
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord)
    encode_to_classname_dict = readImageLabelToDict()[1]
    image_bytes = {}
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        # 'google.protobuf.pyext._message.RepeatedScalarConta' object
        oneHot_label = list(example.features.feature['label'].int64_list.value)
        label = oneHot_label.index(1)

        print('One-hot Label:', oneHot_label, '\nOriginal Label:', label)
        image_bytes[label] = example.features.feature['image_raw'].bytes_list.value[0]
        
        # testImgName = 'test.jpg'
        # with open(testImgName, 'wb') as f:
        #     f.write(base64.b64decode(image_bytes[label]))

        img = Image.frombytes(mode='RGB', size=[100, 100], data=image_bytes[label])
        print('This is', encode_to_classname_dict[str(label)])

        # img = Image.open(testImgName)
        img.show()

        # Exit after 1 iteration as this is purely demonstrative.
        break # if you try to comment this, it will be spectacular :P

def dataPreprocessing(trainPath, testPath):
    """
    Transform image dataset into tfrecord
    """
    # Training
    print('Generating training set...')
    writeTFRecord(os.path.join(DATA_FOLDER, TRAIN_FILE), trainPath)
    # Test
    print('Generating test set...')
    writeTFRecord(os.path.join(DATA_FOLDER, TEST_FILE), testPath)

def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([PREDICT_CLASSES], tf.int64),
        })

    # Convert from a scalar string tensor
    # image_string = base64.b64decode(features['image_raw'])
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape(reduce(lambda x, y: x*y, IMAGE_SHAPE)) # [100 * 100 * 3]
    # image_decoded = tf.image.decode_jpeg(image_string)
    # image_decoded = tf.decode_raw(image_string, tf.uint8)
    # image_resized = tf.image.resize_images(image_decoded, IMAGE_SHAPE)
    image = tf.cast(image, tf.float32)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label

def readTFRecord(filename, batch_size, num_epochs):
    with tf.name_scope('input'):
        # TFRecordDataset opens a binary file and reads one record at a time.
        # `filename` could also be a list of filenames, which will be read in order.
        dataset = tf.data.TFRecordDataset(filename)

        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(decode)

        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
    return iterator

def loadDataset(batch_size, num_epochs, isTrain=True):
    """Reads input data num_epochs times.
    Args:
        isTrain: Selects between the training (True) and test (False) data.
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
        train forever.
    Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, IMAGE_SHAPE]
        * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, NUM_CLASSES).
        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
        required.
    """
    filename = os.path.join(DATA_FOLDER, TRAIN_FILE if isTrain else TEST_FILE)
    return readTFRecord(filename, batch_size, num_epochs)

def generateImageLabel(path, saveTo=IMG_LABEL_JSON):
    """
    Generate image label based on the alphabet order of the file folder structure
    """
    classes = sorted(os.listdir(path))
    classname_to_encode_dict = {}
    encode_to_classname_dict = {}
    for class_encode, classname in enumerate(classes):
        classname_to_encode_dict[classname] = class_encode
        encode_to_classname_dict[class_encode] = classname
    with open(saveTo, "w") as write_file:
        json.dump([classname_to_encode_dict, encode_to_classname_dict], write_file)

def readImageLabelToDict(loadFrom=IMG_LABEL_JSON):
    """
    Load Image label json to return to the original dict

    Return:
        - classname_to_encode_dict
        - encode_to_classname_dict
    """
    with open(loadFrom, "r") as read_file:
        data = json.load(read_file)
    classname_to_encode_dict = data[0]
    encode_to_classname_dict = data[1]
    return classname_to_encode_dict, encode_to_classname_dict

def main():
    dataset_path = './fruits-360/'

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    # Generate a json file of classname label pair
    generateImageLabel('fruits-360/Training')
    dataPreprocessing(trainPath=dataset_path+'Training', testPath=dataset_path+'Test')

if __name__ == "__main__":
    main()
    # testPrintTFRecord('./data/test.tfrecords')
