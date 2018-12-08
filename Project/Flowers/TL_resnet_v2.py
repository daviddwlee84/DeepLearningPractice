import glob
import os.path
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# === Transfer Learning ===
# Load predefined resnet_v2 model by TensorFlow-Slim
from tensorflow.contrib.slim.nets import resnet_v2
# Pretrained parameters
CKPT_FILE = 'pretrained/resnet_v2_152.ckpt'
# =========================

# Load preprocessed data
INPUT_DATA = './data/flower_processed_data_224x224.npy'
# Path to store model
TRAIN_FILE = 'model/resnet_v2_model/resnet_v2_flowers'

# Define training parameter
LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 32
N_CLASSES = 5 # flowers

def get_tuned_variables():
    """
    Get all the parameter from pretrained model except excluded scopes.
    """
    # Get all variables from the model.
    variables_to_restore = {v.name.split(":")[0]: v
                            for v in tf.get_collection(
                                tf.GraphKeys.GLOBAL_VARIABLES)}

    # Skip some variables during restore.
    skip_pretrained_var = ["resnet_v2_152/logits", "global_step"]
    variables_to_restore = {
        v: variables_to_restore[v] for
        v in variables_to_restore if not
        any(x in v for x in skip_pretrained_var)}
    
    return variables_to_restore

def get_trainable_variables():
    # Collect all trainale variables
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # Discard variables that are not in the last layer
    trainable_vars = ["resnet_v2_152/logits"]

    train_vars = [v for v in train_vars
                if any(x in v.name
                        for x in trainable_vars)]
    
    return train_vars

def main():
    # Load preprocessed data
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]
    
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    
    testing_images = processed_data[4]
    testing_labels = processed_data[5]
    print("%d training examples, %d validation examples and %d testing examples." % (
        n_training_example, len(validation_labels), len(testing_labels)))

    images = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, _ = resnet_v2.resnet_v2_152(images, N_CLASSES, is_training=False)

    logits = tf.reshape(net, [-1, 5]) # From (?, 1, 1, 5) to (?, 5)

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Define loss function and training process
    tf.losses.softmax_cross_entropy(
        tf.one_hot(labels, N_CLASSES), logits, weights=1.0)
    total_loss = tf.losses.get_total_loss()


    variables_to_restore = get_tuned_variables()

    # Restore the remaining variables
    saver_pre_trained = tf.train.Saver(
        var_list=variables_to_restore)

    train_vars = get_trainable_variables()

    # Performs gradient decent on the trainable variables 
    optimizer = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=0.9)
    grads = optimizer.compute_gradients(total_loss, var_list=train_vars)
    minimize_op = optimizer.apply_gradients(grads)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        print('Loading tuned variables from %s' % CKPT_FILE)
        saver_pre_trained.restore(sess, CKPT_FILE)
            
        start = 0
        end = BATCH
        for i in range(STEPS):            
            _, loss = sess.run([minimize_op, total_loss], feed_dict={
                images: training_images[start:end], 
                labels: training_labels[start:end]})

            if i % 30 == 0 or i + 1 == STEPS:
                saver.save(sess, TRAIN_FILE, global_step=i)
                
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    images: validation_images, labels: validation_labels})
                print('Step %d: Training loss is %.1f Validation accuracy = %.1f%%' % (
                    i, loss, validation_accuracy * 100.0))
                            
            start = end
            if start == n_training_example:
                start = 0
            
            end = start + BATCH
            if end > n_training_example: 
                end = n_training_example
            
        test_accuracy = sess.run(evaluation_step, feed_dict={
            images: testing_images, labels: testing_labels})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

if __name__ == '__main__':
    toSaveDir = os.path.dirname(TRAIN_FILE)
    if not os.path.exists(toSaveDir):
        # if the folder doesn't exist then mkdir
        os.makedirs(toSaveDir)
    main()