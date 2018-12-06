#coding:utf-8
import tensorflow as tf
import os
import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import data_helper
import ner_backward
import time
import conlleval

# ===== Use different model ===== #
#import ner_forward_BasicRNNCell as ner_forward
import ner_forward_fromScratch as ner_forward
#import ner_forward_LSTM as ner_forward
# =============================== #

MAX_SEQ_LEN = 600
BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 100
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "ner_model"
RESULT_PATH = "./result/"

TEST_INTERVAL_SECS = 10

test_file = "data/ner.dev"

emb_file = "data/ner.emb"

datautil = data_helper.DataUtil()


def test(data):
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.int32, [None, None])
        y_ = tf.placeholder(tf.int64, [None, None])

        word_emb = tf.Variable(datautil._word_emb, dtype=tf.float32, name='word_emb')
        x_emb = tf.nn.embedding_lookup(word_emb, x)

        y = ner_forward.forward(x_emb, is_train=False, regularizer=None)
        predict = tf.argmax(y, -1)

        saver = tf.train.Saver()

        x_batch = []
        for i in range(len(data)):
            pad_lst = [0] * (MAX_SEQ_LEN - len(data[i][0]))
            x_pad = data[i][0] + pad_lst
            x_batch.append(x_pad)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(ner_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    predict_id = sess.run(predict, feed_dict={x: x_batch})

                    filename = os.path.join(RESULT_PATH, 'ner.result')
                    fw = open(filename, 'w')
                    for i in range(len(data)):
                        fw.write('{} {} {}\n'.format("<S>", "O", "O"))
                        for j in range(len(data[i][0])):
                            word = data[i][2][j]
                            predict_str = datautil.id2label(predict_id[i][j])
                            label_str = datautil.id2label(data[i][1][j])
                            fw.write('{} {} {}\n'.format(word, label_str, predict_str))
                        fw.write('{} {} {}\n\n'.format("<E>", "O", "O"))
                    fw.close()
                    print("After %s training step(s), test result is:" % (global_step))
                    conlleval.evaluate(filename)
            time.sleep(TEST_INTERVAL_SECS)

def main():
    datautil.load_emb(emb_file)
    data = datautil.load_data(test_file)

    test(data)

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()
