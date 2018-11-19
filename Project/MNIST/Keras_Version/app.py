# Usage:
#   Input path to picture (can input multiple pictures) and return prediction

import tensorflow as tf
import numpy as np
from PIL import Image # Python Image Library
import sys
import train

# Restore weight and create model to fit it
def restore_weight(model, verbose=True):
    if verbose:
        _, _, x_test, y_test = train.load_dataset()
        _, acc = model.evaluate(x_test, y_test)
        print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

    latest = tf.train.latest_checkpoint(train.checkpoint_dir)
    if latest:
        print("loading weight...")
        model.load_weights(latest)
    else:
        print("can't find weight = =")
        return

    if verbose:
        _, acc = model.evaluate(x_test, y_test)
        print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# Restore model directly
def restore_model():
    return tf.keras.models.load_model(train.model_path)

# Picture preprocessing
def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))

    thershold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < thershold:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    nm_arr = im_arr.reshape((1, 784))
    img_ready = np.multiply(nm_arr, 1/255)

    return img_ready

def main():
    # Restore model
    model = train.create_model()
    restore_weight(model, verbose=False)
    #model = restore_model()
    for path in sys.argv[1:]:
        testPic = path
        # 1. Preprocess the pictures
        testPicArr = pre_pic(testPic)

        # 2. Predict value
        prediction = model.predict(testPicArr) # This will return probability distribution vector (due to softmax)
        preValue = np.argmax(prediction) # Get the maximum index as prediction
        print('Image:', testPic, end='\t')
        print('The prediction number is:', preValue)

if __name__ == '__main__':
    main()
