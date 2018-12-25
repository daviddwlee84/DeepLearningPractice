import tensorflow as tf
from enum import Enum

from Seq2SeqNMT import Seq2SeqNMTModel
from AttentionNMT import AttentionNMTModel

SRC_TRAIN_DATA = "train_data/train.en"
TRG_TRAIN_DATA = "train_data/train.zh"

BATCH_SIZE = 100 # Training batch size
NUM_EPOCH = 5

MAX_LEN = 50   # Maximum words in a sentence
SOS_ID  = 1    # ID of <sos> in target vocabulary

class MODEL(Enum):
    Seq2Seq = 0
    Attention = 1

# Construct TensorFlow dataset
# (every words are embedding)
def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    # Seperate embedding by white space and put it into an one-dimension array
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # Transfer string into integer
    dataset = dataset.map(
        lambda string: tf.string_to_number(string, tf.int32))
    # Calculate words number and put the this information into dataset
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset

# Load data from source and target document
def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    # Load source and target data
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)
    # Combine two dataset into one dataset
    # single transaction is constructed by 4 tensors:
    #   ds[0][0] is original sentence
    #   ds[0][1] is original sentence length
    #   ds[1][0] is target sentence
    #   ds[1][1] is target sentence length
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    # Delete empty sentence (only contain <eos>) and sentence which length exceed the maximum
    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(
            tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(
            tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok)
    dataset = dataset.filter(FilterLength)
    
    # Decoder need two target sentence with different format:
    #   1. Input of decoder (trg_input) e.g. "<sos> X Y Z"
    #   2. Output of decoder (trg_label) e.g. "X Y Z <eos>"
    # "X Y Z <eos>" is what we've loaded from document
    # now we need to generate "<sos> X Y Z" and add to dataset
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))
    dataset = dataset.map(MakeTrgInput)

    # Random shuffle
    dataset = dataset.shuffle(10000)

    # Define the output data dimension
    padded_shapes = (
        (tf.TensorShape([None]),      # Source sentence is an unknown length vector
         tf.TensorShape([])),         # Source sentence length is a number
        (tf.TensorShape([None]),      # Target sentence (input of decoder) is an unknown length vector
         tf.TensorShape([None]),      # Target sentence (output of decoder) is an unknown length vector
         tf.TensorShape([])))         # Target sentence length is a number
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset

# Train an epoch and return global step
def run_epoch(session, cost_op, train_op, saver, step):
    # Repeat training steps until iterate through all the data in the dataset
    while True:
        try:
            # Calculate loss
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print("After %d steps, per token cost is %.3f" % (step, cost))
            # Save checkpoint for every 200 steps
            if step % 200 == 0:
                saver.save(session, CHECKPOINT_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step

def train(model):
    # Define initializer
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # Define the model of RNN
    global CHECKPOINT_PATH
    if model == MODEL.Seq2Seq:
        CHECKPOINT_PATH = "seq2seq_model/seq2seq_ckpt"
        with tf.variable_scope("seq2seq_nmt_model", reuse=None, 
                            initializer=initializer):
            train_model = Seq2SeqNMTModel()
    elif model == MODEL.Attention:
        CHECKPOINT_PATH = "attention_model/attantion_ckpt"
        with tf.variable_scope("attention_nmt_model", reuse=None, 
                    initializer=initializer):
            train_model = AttentionNMTModel()

    # Define input data
    data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()
 
    # Define compute graph
    # Input input data as tensor to forward
    cost_op, train_op = train_model.forward(src, src_size, trg_input,
                                            trg_label, trg_size)

    # Train model
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)

def main():
    train(MODEL.Seq2Seq)
    train(MODEL.Attention)
    
if __name__ == "__main__":
    main()
