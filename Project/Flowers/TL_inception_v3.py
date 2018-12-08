import glob
import os.path
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# === Transfer Learning ===
# Load predefined inception_v3 model by TensorFlow-Slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3
# Pretrained parameters
CKPT_FILE = 'pretrained/inception_v3.ckpt'
# =========================

# Load preprocessed data
INPUT_DATA = './data/flower_processed_data_299x299.npy'
# Path to store model
TRAIN_FILE = 'model/inception_v3_model/inception_v3_flowers'

# Define training parameter
LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 32
N_CLASSES = 5 # flowers

# The parameters we want to exclude
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
# The parameters we need to train. i.e. the last fully connected layer
TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogit'

def get_tuned_variables():
    """
    Get all the parameter from pretrained model except excluded scopes.
    """
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables(): # Go through every parameters
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

def get_trainable_variables():    
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    
    # List all the training parameters' prefix then find the weights need to be trained
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

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

    # Define input of inception_v3
    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')
    
    # Define model of inception_v3
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(
            images, num_classes=N_CLASSES, is_training=True)
    
    trainable_variables = get_trainable_variables()

    # Define loss function and training process
    tf.losses.softmax_cross_entropy(
        tf.one_hot(labels, N_CLASSES), logits, weights=1.0)
    total_loss = tf.losses.get_total_loss()
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(total_loss)
    
    # Calculate accuracy
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                
    # Load pretrained model
    load_fn = slim.assign_from_checkpoint_fn(
      CKPT_FILE,
      get_tuned_variables(),
      ignore_missing_vars=True)
    
    # A saver to save new model
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        # Load model
        print('Loading tuned variables from %s' % CKPT_FILE)
        load_fn(sess)
            
        start = 0
        end = BATCH
        for i in range(STEPS):            
            _, loss = sess.run([train_step, total_loss], feed_dict={
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
            
        # Test accuracy
        test_accuracy = sess.run(evaluation_step, feed_dict={
            images: testing_images, labels: testing_labels})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

if __name__ == '__main__':
    toSaveDir = os.path.dirname(TRAIN_FILE)
    if not os.path.exists(toSaveDir):
        # if the folder doesn't exist then mkdir
        os.makedirs(toSaveDir)
    main()