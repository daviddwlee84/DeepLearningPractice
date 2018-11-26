# Usage:
#   Input path to picture (can input multiple pictures) and return prediction

import tensorflow as tf
import numpy as np
from PIL import Image # Python Image Library
import cifar_backward
import cifar_forward
import sys
import os

# Restore model and predict
def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [1, cifar_forward.IMAGE_SIZE, cifar_forward.IMAGE_SIZE, cifar_forward.IMAGE_CHANNELS])
        y = cifar_forward.forward(x, None, train=False)
        preValue = tf.argmax(y, 1) # The maximum label is the prediction

        variable_averages = tf.train.ExponentialMovingAverage(cifar_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            # Restore model
            ckpt = tf.train.get_checkpoint_state(cifar_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                # Predict
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1

# Picture preprocessing
def pre_pic(picName):
    img = Image.open(picName)
    # Transform picture size to fit our training model
    reIm = img.resize((cifar_forward.IMAGE_SIZE, cifar_forward.IMAGE_SIZE), Image.ANTIALIAS)
    im_arr = np.array(reIm)

    # Reshape picture
    reshape_img_arr = im_arr.reshape((1, cifar_forward.IMAGE_SIZE, cifar_forward.IMAGE_SIZE, cifar_forward.IMAGE_CHANNELS))

    return reshape_img_arr

def main():
    with open('../cifar-10/class_encode.txt', 'r') as encodefile:
        label_encode_pair = encodefile.readlines()
    
    encode_dict = {}
    for pair in label_encode_pair:
        sep_pair = pair.split()
        encode_dict[int(sep_pair[1])] = sep_pair[0]

    for path in sys.argv[1:]:
        testPic = path
        # 1. Preprocess the pictures
        testPicArr = pre_pic(testPic)
        # 2. Predict class
        preClass = restore_model(testPicArr)
        print('Image:', testPic, end='\n\t')
        print('This is ...', encode_dict[int(preClass)], '\t(I think :P')

if __name__ == '__main__':
    main()
