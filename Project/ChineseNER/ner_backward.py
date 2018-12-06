import tensorflow as tf
tf.set_random_seed(1)
import numpy as np
np.random.seed(1)
import os
import data_helper

# ===== Use different model ===== #
#import ner_forward_BasicRNNCell as ner_forward
import ner_forward_fromScratch as ner_forward
#import ner_forward_LSTM as ner_forward
# =============================== #

BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 100
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "ner_model"

train_file = "data/ner.train"

emb_file = "data/ner.emb"

datautil = data_helper.DataUtil()


def backward():
    # Two dimensional data
    # Batch size and Sequence length [batch_size, seqlen]
    x = tf.placeholder(tf.int32, [None, None])

    word_emb = tf.Variable(datautil._word_emb, dtype=tf.float32, name='word_emb')
    # For each input word transfer to 100-dimension vector
    # [batch_size, seqlen, emb_size]
    x_emb = tf.nn.embedding_lookup(word_emb, x)

    # Two dimensional Label
    # [batch_size, seqlen]
    y_ = tf.placeholder(tf.int32, [None, None])
    y = ner_forward.forward(x_emb, is_train=True, regularizer=REGULARIZER)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cem = tf.reduce_mean(ce)

    loss = cem + tf.add_n(tf.get_collection('losses'))

    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(1, STEPS+1):
            batches = datautil.gen_mini_batch(BATCH_SIZE)

            total_num, total_loss = 0, 0
            for batch_id, batch_data in enumerate(batches): # Iterate through every batch
                x_batch, label_batch = batch_data

                _, loss_value = sess.run([train_step, loss], feed_dict={x: x_batch, y_: label_batch})
                total_num = total_num + len(x_batch)
                total_loss = total_loss + loss_value * len(x_batch)

            avg_loss = total_loss/total_num # Calculate average loss of entire training set
            print("After %d training step(s), loss on training batch is %g." % (i, avg_loss))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=i)

def main():
    datautil.load_emb(emb_file) # Load pretrained embedding
    datautil.load_data(train_file) # Load training set
    backward()

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()
