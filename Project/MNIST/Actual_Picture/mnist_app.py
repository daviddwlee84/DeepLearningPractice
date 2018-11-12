# Usage:
#   Input path to picture (can input multiple pictures) and return prediction

import tensorflow as tf
import numpy as np
from PIL import Image # Python Image Library
import mnist_backward
import mnist_forward
import sys

# Restore model and predict
def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y, 1) # The maximum label is the prediction

        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            # Restore model
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                # Predict
                preValue = sess.run(preValue, feed_dict={x:testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1

# Picture preprocessing
def pre_pic(picName):
    img = Image.open(picName)
    # Transform picture size to fit our training model
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    # Convert image to inverse color (back background and white word)
    im_arr = np.array(reIm.convert('L'))

    # Binarize => get rid of noise
    thershold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < thershold:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    
    # Reshape picture to one row
    nm_arr = im_arr.reshape((1, 784))
    # Normalize value to [0, 1]
    img_ready = np.multiply(nm_arr, 1/255)

    return img_ready

def main():
    for path in sys.argv[1:]:
        testPic = path
        # 1. Preprocess the pictures
        testPicArr = pre_pic(testPic)
        # 2. Predict value
        preValue = restore_model(testPicArr)
        print('Image:', testPic, end='\t')
        print('The prediction number is:', preValue)

if __name__ == '__main__':
    main()
