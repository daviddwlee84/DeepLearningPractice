import numpy as np
import tensorflow as tf

import sys
import os

from tfrecord_manager import loadDataset, readImageLabelToDict

#tf.logging.set_verbosity(tf.logging.INFO)

IMAGE_SHAPE = [100, 100, 3]
PREDICT_CLASSES = 83
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'flower'

BATCH_SIZE = 100
STEPS = 2000
NUM_EPOCHS = 5

def cnn_model_fn(input_data):
    """Model function for CNN."""
    # Input Layer
    # [batch_size, image_height, image_width, channels]
    input_layer = tf.reshape(input_data, [-1, 100, 100, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 25 * 25 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=PREDICT_CLASSES)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    return logits, predictions


def training(datasetIterator, multiThread=True):

    # Create a feedable iterator that use a placeholder to switch between dataset
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, output_types=datasetIterator.output_types, output_shapes=datasetIterator.output_shapes)
    next_element = iterator.get_next() # return string

    data = tf.placeholder(tf.float32, [BATCH_SIZE, 100, 100, 3])
    labels = tf.placeholder(tf.float32, [BATCH_SIZE, PREDICT_CLASSES])
    global_step = tf.train.create_global_step()

    logits, _ = cnn_model_fn(data)

    # Calculate Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(labels, 1), logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=global_step)
    
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Create saveable object from iterator.
        saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
        # Save the iterator state by adding it to the saveable objects collection.
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
        # Return handles that can be feed as the iterator in sess.run
        data_handler = sess.run(datasetIterator.string_handle())

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Restore saved session
        ckpt = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
        if ckpt:
            saver.restore(sess, ckpt)
        else:
            tf.global_variables_initializer()

        if multiThread:
            # Thread
            coordinator = tf.train.Coordinator() # coordinator for threads
            # start_queue_runners (from tensorflow.python.training.queue_runner_impl)
            # is deprecated and will beremoved in a future version.
            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
        
        for i in range(STEPS):
            #xs, ys = sess.run([train_data, train_labels])
            #string_record = sess.run(next_element, feed_dict={handle: data_handler})
            (features, label) = sess.run(next_element, feed_dict={handle: data_handler})
            reshaped_feature = np.reshape(features, [BATCH_SIZE, 100, 100, 3])
            _, loss_value, step = sess.run([train_op, loss, tf.train.get_global_step()], feed_dict={data: reshaped_feature, labels: label})

            if i % 10 == 0:
                print("After %d training batch(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
            i += 1
        
        if multiThread:    
            coordinator.request_stop()
            coordinator.join(threads)

def test(datasetIterator, multiThread=False):

    # This can write in tfrecord_manager.py next time
    # seems like it doesn't need to be feedable to work
    test_data, test_labels = datasetIterator.get_next()

    data = tf.placeholder(tf.float32, [BATCH_SIZE, 100, 100, 3])
    labels = tf.placeholder(tf.float32, [BATCH_SIZE, PREDICT_CLASSES])
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
    _, predictions = cnn_model_fn(data)

    saver = tf.train.Saver()

    # Add evaluation metrics
    # eval_metric_ops = tf.metrics.accuracy(
    #         labels=tf.argmax(labels, 1), predictions=predictions["classes"])
    eval_metric_ops = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), predictions["classes"]), tf.float32))
        
    with tf.Session() as sess:

        ckpt = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
        if ckpt:
            saver.restore(sess, ckpt)
        else:
            print('No checkpoint file found in', MODEL_SAVE_PATH)
            return
        
        if multiThread:
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
        
        xs, y_ = sess.run([test_data, test_labels])
        reshaped_xs = np.reshape(xs, [BATCH_SIZE, 100, 100, 3])

        accuracy_score = sess.run(eval_metric_ops, feed_dict={data: reshaped_xs, labels: y_})
        print("After %s training step(s), test accuracy = %g" % (sess.run(global_step), accuracy_score))
        
        if multiThread:    
            coordinator.request_stop()
            coordinator.join(threads)
        
def main(argv_mode, argv_thread):
    if argv_mode == 'train':
        # Load training and
        datasetIterator = loadDataset(BATCH_SIZE, NUM_EPOCHS, isTrain=True)
        # Train the model
        training(datasetIterator, multiThread=argv_thread)
    else:
        # Load test data
        datasetIterator = loadDataset(BATCH_SIZE, NUM_EPOCHS, isTrain=False)
        test(datasetIterator, multiThread=argv_thread)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python model.py (train/test) 0/1    (0: single thread, 1: multi-thread)')
    mode = sys.argv[1]
    if sys.argv[2] == '1':
        multiThread = True
        print(mode, "mode with multi-thread")
    else:
        multiThread = False
        print(mode, "mode with single thread")
    
    main(mode, multiThread)
