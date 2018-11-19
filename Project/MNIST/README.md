# MNIST Handwriting Digit

## Overview

Version

* [Use Online Database](#Use-Online-Database)
* [Use Actual Images](#Use-Actual-Images)
* [Use tf.keras (Tensorflow tutorial)](#Use-Keras)

### Network Structure

![structure](structure.png)

## Use Online Database

### Files

```txt
.
├── data (auto generate) - MNIST dataset will download here
├── mnist_app.py - input image and output number
├── mnist_backward.py - back propagation
├── mnist_forward.py - forward propagation
├── mnist_test.py - testing
├── model (auto generate) - saved model training status
├── test_picture - custom number images
└── test_picture.sh - test custom images
```

### Instructions

1. Go to project directory

    `cd Online_Database`

2. Training phase

    `python3 mnist_backward.py`

3. Testing phase (can parallel running while training)

    `python3 mnist_forward.py`

4. Custom test

    `bash test_picture.sh`

    or

    `python3 mnist_app.py path/to/image1 path/to/image2 ...`

### Result

## Use Actual Images

### Files

```txt
.
├── custom_num - custom number images
├── custom_test.sh - test custom images
├── custom_test.txt - custom image test list
├── data (auto generate) - orgnized data generate by mnist_generate_dataset.py
├── execute-in-parallel-tmux.sh - run training and testing at the same time
├── mnist_app.py - input image and output number
├── mnist_backward.py - back propagation
├── mnist_data_jpg.tar.gz - MNIST image dataset
├── mnist_forward.py - forward propagation
├── mnist_generate_dataset.py - generate TFRecord file from dataset
├── mnist_test.py - testing
└── model (auto generate) - saved model training status
```

### Instructions

1. Goto project directory

    `cd Actual_Picture`

2. Decompress the MNIST images (there are 6000 + 1000 images so it might take a while)

    `tar xzf mnist_data_jpg.tar.gz`

> [How to obtain maximum compression with .tar.gz](https://superuser.com/questions/514260/how-to-obtain-maximum-compression-with-tar-gz)

3. Generate the TFRecord file

    `python3 mnist_generate_dataset.py`

4. Train the model
    * You can use tmux to see the training and testing progress at the same time (i.e. step 5)

        `bash execute-in-parallel-tmux.sh`

    * Or just train the model

        `python3 mnist_backward.py`

5. Test with testing data (optional)

    `python3 mnist_test.py` (terminate with `Ctrl + c`)

6. Test with custom images (by default, image should be black number and white background)
    * Use my images (you can save your image paths in `custom_test.txt`)

        `bash custom_test.sh`

    * Or directly input image path

        `python3 mnist_app.py path/to/image1 path/to/image2 ...`

### Result

TBD

* Model
* Batch size
* Steps
* Learing rate
* Accuracy after xx round

TODO: try to get rid of warnings

## Use Keras

* [Tensorflow Tutorial Example](https://www.tensorflow.org/tutorials/)
* [Original File](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/_index.ipynb)

This practice implement storing training checkpoints and restoring model weights. ([TensorFlow Tutorial: Save and restore models](https://www.tensorflow.org/tutorials/keras/save_and_restore_models))

(And I found error when using model.save())

> NotImplementedError: Currently `save` requires model to be a graph network. Consider using `save_weights`, in order to save the weights of the model.

### Instruction

```sh
# Train Model
python3 train.py
# Use App
find ../Actual_Picture/custom_num -type f | xargs python3 app.py
```

### Result

```txt
10000/10000 [==============================] - 1s 96us/step
Untrained model, accuracy: 15.55%
10000/10000 [==============================] - 1s 78us/step
Restored model, accuracy: 97.81%
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
flatten (Flatten)            multiple                  0
_________________________________________________________________
dense (Dense)                multiple                  401920
_________________________________________________________________
dropout (Dropout)            multiple                  0
_________________________________________________________________
dense_1 (Dense)              multiple                  5130
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
```

## Links

### Article

* [Putting it all together and Classifying MNIST dataset](https://deepnotes.io/classify-mnist)
