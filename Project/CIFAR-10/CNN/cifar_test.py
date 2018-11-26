#coding:utf-8
# Usage:
# Test trained  model accuracy by loading checkpoint status
# (Can be execute parallel with training program)

import time
import tensorflow as tf
import cifar_forward
import cifar_backward
import cifar_generate_dataset
import numpy as np

TEST_INTERVAL_SECS = 5 # Test each round for 5 seconds delay
TEST_BATCH_NUM = 10000

def test():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [TEST_BATCH_NUM, 
                                        cifar_forward.IMAGE_SIZE,
                                        cifar_forward.IMAGE_SIZE,
                                        cifar_forward.IMAGE_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, cifar_forward.OUTPUT_NODE])
        y = cifar_forward.forward(x, None, train=False)

        ema = tf.train.ExponentialMovingAverage(cifar_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        image_batch, label_batch = cifar_generate_dataset.get_tfRecord(TEST_BATCH_NUM, getTrain=False)

        # Keep testing
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(cifar_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    coordinator = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

                    xs, ys = sess.run([image_batch, label_batch])
                    reshaped_xs = np.reshape(xs, [TEST_BATCH_NUM,
                                                  cifar_forward.IMAGE_SIZE,
                                                  cifar_forward.IMAGE_SIZE,
                                                  cifar_forward.IMAGE_CHANNELS])

                    accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_xs, y_: ys})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))

                    coordinator.request_stop()
                    coordinator.join(threads)

                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    test()

if __name__ == '__main__':
    main()
