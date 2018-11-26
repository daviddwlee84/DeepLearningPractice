# Usage:
#   Define forward propagation process

import tensorflow as tf

IMAGE_SIZE = 32
IMAGE_CHANNELS = 3 # RGB

CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
DENSE_SIZE = 512 # Fully-connected layer

OUTPUT_NODE = 10

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):  
    b = tf.Variable(tf.zeros(shape))  
    return b

def conv2d(x, w):
    # strides
    # padding
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
# Forward propagation
def forward(x, regularizer, train=False):
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, IMAGE_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_2x2(relu1)

    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    dense1_w = get_weight([nodes, DENSE_SIZE], regularizer)
    dense1_b = get_bias([DENSE_SIZE])
    dense1 = tf.nn.relu(tf.matmul(reshaped, dense1_w) + dense1_b)
    if train:
        dense1 = tf.nn.dropout(dense1, 0.5)
    
    dense2_w = get_weight([DENSE_SIZE, OUTPUT_NODE], regularizer)
    dense2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(dense1, dense2_w) + dense2_b
    return y
