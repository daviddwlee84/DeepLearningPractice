import tensorflow as tf

INPUT_NODE = 100
OUTPUT_NODE = 7
MAX_SEQ_LEN = 100
HIDDEN_SIZE = 50
BATCH_SIZE = 128

def get_weight(shape, regularizer=None):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, is_train=True, regularizer=None): #[batch, seqlen, emb_size]
    seq_len = 100
    if not is_train:
        # In case of out of memory, the maximum of sentence is 600
        seq_len = 600

    #### Key Part
    # Input Layer --> Hidden Layer
    weight_xh = get_weight(shape=[INPUT_NODE, HIDDEN_SIZE])
    bias_xh = get_bias(shape=[HIDDEN_SIZE])

    # Hidden Layer --> Hidden Layer Weight Matrix (only in RNN!)
    weight_hh = get_weight(shape=[HIDDEN_SIZE, HIDDEN_SIZE])
    bias_hh = get_bias(shape=[HIDDEN_SIZE])

    # Hidden Layer --> Output Layer
    weight_ho = get_weight(shape=[HIDDEN_SIZE, OUTPUT_NODE], regularizer=regularizer)
    bias_ho = get_bias(shape=[OUTPUT_NODE])

    h = tf.zeros(shape=[tf.shape(x)[0], HIDDEN_SIZE], dtype=tf.float32) #[batch, hidden_size]
    
    output = [] # [batch_size, output_node] * seqlen
    seq_lst = tf.transpose(x, [1, 0, 2]) # [batch_size, seqlen, emb_size] to [seqlen, batch, emb_size]
    seq_lst = tf.unstack(seq_lst, num=seq_len) # [seqlen, batch, emb_size] to [seqlen, emb_size] * batch
    with tf.variable_scope("SimpleRNN"):
        for i, inp in enumerate(seq_lst):
            if i > 0:
                # For each step, RNN share the weights.
                tf.get_variable_scope().reuse_variables()
            
            # h = tanh(x_t * W_xh + b_xh + h_t-1 * W_hh + b_hh)
            h = tf.matmul(inp, weight_xh) + bias_xh + tf.matmul(h, weight_hh) + bias_hh
            h = tf.tanh(h)

            output.append(h)
        output = tf.transpose(output, [1, 0, 2])
    ###

    y = tf.matmul(tf.reshape(output, [-1, HIDDEN_SIZE]), weight_ho) + bias_ho
    return tf.reshape(y, [-1, seq_len, OUTPUT_NODE])

