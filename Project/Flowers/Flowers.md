# Flowers Classification

## Data Preprocessing

> Modify image shape to fit the pretrained model.

* Resnet_v2: [244, 244]
    ```py
    # ResNet-101 for image classification into 1000 classes:
    # inputs has shape [batch, 224, 224, 3]
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, end_points = resnet_v2.resnet_v2_101(inputs, 1000, is_training=False)
    ```
* Inception_v3: [299, 299]

> Change global variable `IMG_RESHAPE` and `OUTPUT_FILE`

```sh
# Extract dataset
tar xzf flower_photos.tar.gz
# Generate numpy array format and reshape image size
python3 dataProcessing.py
```

## Transfer Learning

Steps

1. Process the data to fit the pre-trained model's input
2. Load pre-processed data
3. Load the model
    1. Model itself
    2. Specify the parameters we want to keep
        * basically exclude the inference weights
    3. Specify the parameters we want to train
4. Define loss function and training process
5. Load parameters into the model
6. Now it should be able to train

### Tensorflow Slim

* [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim)
* [TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)
* [Tensorflow finetune_inception_v3_on_flowers.sh](https://github.com/tensorflow/models/blob/master/research/slim/scripts/finetune_inception_v3_on_flowers.sh)

```py
import tensorflow.contrib.slim as slim

# import tensorflow as tf
# slim = tf.contrib.slim
```

## [ResNet V2](TL_resnet_v2.py)

* [ResNet V2 paper](https://arxiv.org/abs/1603.05027)
* [Network](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v2.py)
* [checkpoint](http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz)

```py
import tensorflow.contrib.slim.python.slim.nets.resnet_v2 as resnet_v2
```

```sh
# Download pretrained checkpoint
wget http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz
mkdir pretrained
# Extract checkpoint
tar xzf resnet_v2_152_2017_04_14.tar.gz -C pretrained
# Train model
python3 TL_resnet_v2.py
```

Result:

```txt
########################################################################
#      Date:           Tue Dec  4 06:50:20 PST 2018
#    Job ID:           203984.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,vmem=92gb,walltime=06:00:00
########################################################################

Starting resnet_v2
2894 training examples, 391 validation examples and 385 testing examples.
Loading tuned variables from pretrained/resnet_v2_152.ckpt
Step 0: Training loss is 2.5 Validation accuracy = 9.2%
Step 30: Training loss is 1.8 Validation accuracy = 36.8%
Step 60: Training loss is 1.2 Validation accuracy = 56.3%
Step 90: Training loss is 1.0 Validation accuracy = 62.4%
Step 120: Training loss is 1.3 Validation accuracy = 65.5%
Step 150: Training loss is 1.1 Validation accuracy = 68.8%
Step 180: Training loss is 1.0 Validation accuracy = 70.1%
Step 210: Training loss is 0.9 Validation accuracy = 70.8%
Step 240: Training loss is 0.8 Validation accuracy = 72.6%
Step 270: Training loss is 0.7 Validation accuracy = 74.7%
Step 299: Training loss is 1.0 Validation accuracy = 74.9%
Final test accuracy = 82.6%
End of resnet_v2

########################################################################
# End of output for job 203984.c009
# Date: Tue Dec  4 06:59:53 PST 2018
########################################################################
```

## [Inception V3](TL_inception_v3.py)

* [Inception V3 paper](https://arxiv.org/abs/1512.00567)
* [Network](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py)
* [checkpoint](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)

```py
from tensorflow.contrib.slim.python.slim.nets import inception_v3
```

There are different mechanism between pretrained models.

In Inception_v3 you need to find `tf.variable_scope` you want to train and then exclude it.

For example:

* Final pooling and prediction: Logits
* Auxiliary Head logits: AuxLogits

```sh
# Download pretrained checkpoint
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
mkdir pretrained
# Extract checkpoint
tar xzf inception_v3_2016_08_28.tar.gz -C pretrained
# Train model
python3 TL_inception_v3.py
```

Result:

```txt
########################################################################
#      Date:           Tue Dec  4 04:37:31 PST 2018
#    Job ID:           203923.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,vmem=92gb,walltime=06:00:00
########################################################################

2985 training examples, 337 validation examples and 348 testing examples.
Loading tuned variables from pretrained/inception_v3.ckpt
Step 0: Training loss is 1.9 Validation accuracy = 19.3%
Step 30: Training loss is 1.9 Validation accuracy = 25.5%
Step 60: Training loss is 1.6 Validation accuracy = 44.2%
Step 90: Training loss is 1.0 Validation accuracy = 75.7%
Step 120: Training loss is 0.6 Validation accuracy = 88.7%
Step 150: Training loss is 0.5 Validation accuracy = 92.3%
Step 180: Training loss is 0.7 Validation accuracy = 92.3%
Step 210: Training loss is 0.4 Validation accuracy = 92.3%
Step 240: Training loss is 0.3 Validation accuracy = 93.8%
Step 270: Training loss is 0.2 Validation accuracy = 93.8%
Step 299: Training loss is 0.2 Validation accuracy = 92.9%
Final test accuracy = 94.5%

########################################################################
# End of output for job 203923.c009
# Date: Tue Dec  4 04:52:28 PST 2018
########################################################################
```

## Links

Transfer Learning

* [**Transfer Learning with TensorFlow Tutorial: Image Classification Example**](https://lambdalabs.com/blog/transfer-learning-with-tensorflow-tutorial-image-classification-example/)
* [Transfer Learning - Everything about Transfer Learning and Domain Adaptation](http://transferlearning.xyz/)
    * [github](https://github.com/jindongwang/transferlearning)
* [Transfer Learning Tutorial](https://github.com/kwotsin/transfer_learning_tutorial)
* [Github Topic #transfer-learning](https://github.com/topics/transfer-learning)
