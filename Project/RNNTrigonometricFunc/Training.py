import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from TrainDataGen import TIMESTEPS, TESTING_EXAMPLES, generate_data

from enum import Enum

HIDDEN_SIZE = 30       # Hidden nodes in LSTM
NUM_LAYERS = 2         # Layer of LSTM
TRAINING_STEPS = 10000
BATCH_SIZE = 32

class TRAINING(Enum):
    SINE = 0
    COSINE = 1

def lstm_model(X, y=[0.0], is_training=False):
    # Use multi-layer LSTM cell
    cell = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.LSTMCell(HIDDEN_SIZE, name='basic_lstm_cell') 
        for _ in range(NUM_LAYERS)])

    # Construct LSTM into an RNN network, output feed forward result
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = outputs[:, -1, :]

    # Add a dense layer as the last layer
    predictions = tf.contrib.layers.fully_connected(
        output, 1, activation_fn=None)
    
    # Only calculate loss and optimization result when training
    if not is_training:
        # Return prediction when testing and predicting
        return predictions, None, None
        
    # Use mean squared error as loss function
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    # Construct an optimizer to get train operation
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1)
    return predictions, loss, train_op

def run_eval(sess, test_X, test_y, mode):
    # Transform data into tf.Dataset
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()
    
    # Get the prediction of the model
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X)
    
    # Store prediction result
    predictions = []
    labels = []
    for _ in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    # Use root mean square error as evaluation metrics
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Root Mean Square Error is: %f" % rmse)
    
    plt.figure()
    plt.plot(predictions, label='predictions')
    if mode == TRAINING.SINE:
        plt.plot(labels, label='real sin')
    elif mode == TRAINING.COSINE:
        plt.plot(labels, label='real cos')
    plt.legend()
    plt.show()

def training(mode=TRAINING.SINE):
    # Generate data
    if mode == TRAINING.SINE:
        train_X, train_y, test_X, test_y = generate_data(np.sin)
    elif mode == TRAINING.COSINE:
        train_X, train_y, test_X, test_y = generate_data(np.cos)

    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model"):
        _, loss, train_op = lstm_model(X, y, True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Evaluate model before training.")
        run_eval(sess, test_X, test_y, mode)

        for i in range(TRAINING_STEPS):
            _, l = sess.run([train_op, loss])
            if i % 1000 == 0:
                print("train step: " + str(i) + ", loss: " + str(l))
        
        print("Evaluate model after training.")
        run_eval(sess, test_X, test_y, mode)

if __name__ == "__main__":
    training(TRAINING.COSINE)
