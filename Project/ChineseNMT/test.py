import tensorflow as tf
import codecs
import sys
from enum import Enum

from Seq2SeqNMT import Seq2SeqNMTModel
from AttentionNMT import AttentionNMTModel

# Vocabulary
SRC_VOCAB = "output_vocab/en.vocab"
TRG_VOCAB = "output_vocab/zh.vocab"

class MODEL(Enum):
    Seq2Seq = 0
    Attention = 1

def EncodeSourceEmbedding(english_text):
    # Transfer english words into embedding by vocabulary table
    with codecs.open(SRC_VOCAB, "r", "utf-8") as f_vocab:
        src_vocab = [w.strip() for w in f_vocab.readlines()]
        src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))
    english_ids = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>'])
                   for token in english_text.split()]
    return english_ids

def DecodeTargetEmbedding(chinese_ids):
    # Transfer output embedding to chinese words by vocabulary table
    with codecs.open(TRG_VOCAB, "r", "utf-8") as f_vocab:
        trg_vocab = [w.strip() for w in f_vocab.readlines()]
    output_text = ''.join([trg_vocab[x] for x in chinese_ids])

    return output_text.encode('utf8').decode(sys.stdout.encoding)

def translateSentence(english_text, model):
    # Define the model of RNN
    if model == MODEL.Seq2Seq:
        CHECKPOINT_PATH = tf.train.latest_checkpoint("./seq2seq_model")
        with tf.variable_scope("seq2seq_nmt_model", reuse=None):
            model = Seq2SeqNMTModel()
    elif model == MODEL.Attention:
        CHECKPOINT_PATH = tf.train.latest_checkpoint("./attention_model")
        with tf.variable_scope("attention_nmt_model", reuse=None):
            model = AttentionNMTModel()

    english_ids = EncodeSourceEmbedding(english_text)
    # print(english_ids)

    # Construct compute graph for decode
    output_op = model.inference(english_ids)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, CHECKPOINT_PATH)

        # Get translation result
        output_ids = sess.run(output_op)
        # print(output_ids)
        result = DecodeTargetEmbedding(output_ids)

    return result

def main():
    test_en_text = "This is a test . <eos>"
    #test_en_text = "Please give me a hundred on my final score . For the sake of how hard I paid on this course . <eos>"
    print('English:', test_en_text)

    seq2seq_output = translateSentence(test_en_text, MODEL.Seq2Seq)
    #attention_output = translateSentence(test_en_text, MODEL.Attention)
    print('Chinese (seq2seq):', seq2seq_output)
    #print('Chinese (attention):', attention_output)

if __name__ == "__main__":
    main()
