import tensorflow as tf
import numpy as np
from PIL import Image # Python Image Library
import backward
import forward
import sys

# Restore model and predict
def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
        y = forward.forward(x, None)
        preValue = tf.argmax(y, 1) # The maximum label is the prediction

        variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            # Restore model
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
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
    reIm = img.resize((16, 16), Image.ANTIALIAS)
    # Convert image to inverse color (back background and white word)
    im_arr = np.array(reIm.convert('L'))

    # Binarize => get rid of noise
    thershold = 100
    for i in range(16):
        for j in range(16):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < thershold:
                im_arr[i][j] = 255
            else:
                im_arr[i][j] = 0

    # Reshape picture to one row
    nm_arr = im_arr.reshape((1, 16*16))
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
