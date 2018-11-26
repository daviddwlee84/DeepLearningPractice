import tensorflow as tf
import forward
import dataloader
import os

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/" # Save the model in this directory
MODEL_NAME = "model" # Model file prefix

# Back propagation
def backward(data):
    x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
    y = forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    # Computes sparse softmax cross entropy between logits and labels
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce) # Computes the mean of elements across dimensions of a tensor
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # Applies exponential decay to the learning rate
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        data.train.num_examples / BATCH_SIZE, 
        LEARNING_RATE_DECAY,
        staircase=True)

    # Optimizer that implements the gradient descent algorithm
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Maintains moving averages of variables by employing an exponential decay
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Restore pretrained session (if exist)
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = data.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main():
    data = dataloader.getSemeionData()
    backward(data)

if __name__ == '__main__':
    main()
