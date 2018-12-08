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


def LSTMCell(x_t, h_t_1, C_t_1):
    """
    Input: input x_t, output of last layer h_t-1, last cell state C_t-1
    Output: updated h_t and cell state C_t
    """
    # Weights
    # Forget Gate
    W_f = get_weight([INPUT_NODE + HIDDEN_SIZE, HIDDEN_SIZE])
    b_f = get_bias([HIDDEN_SIZE])
    # Input Gate
    W_i = get_weight([INPUT_NODE + HIDDEN_SIZE, HIDDEN_SIZE])
    b_i = get_bias([HIDDEN_SIZE])
    # Input Gate Candidate weight
    W_c = get_weight([INPUT_NODE + HIDDEN_SIZE, HIDDEN_SIZE])
    b_c = get_bias([HIDDEN_SIZE])
    # Output Gate
    W_o = get_weight([INPUT_NODE + HIDDEN_SIZE, HIDDEN_SIZE])
    b_o = get_bias([HIDDEN_SIZE])

    # Calculation
    # Concatenate x_t and h_t-1
    conInput = tf.concat([h_t_1, x_t], -1) # Concatenates tensors along one dimension
    # Forget Gate
    forgetGate = tf.sigmoid(tf.matmul(conInput, W_f) + b_f)
    C_forgotten = forgetGate * C_t_1
    # Input Gate
    inputGate = tf.sigmoid(tf.matmul(conInput, W_i) + b_i)
    candidate = tf.tanh(tf.matmul(conInput, W_c) + b_c)
    C_t = C_forgotten + inputGate * candidate
    # Output Gate
    outputGate = tf.sigmoid(tf.matmul(conInput, W_o) + b_o)
    h_t = outputGate * tf.tanh(C_t)

    return h_t, C_t

def forward(x, is_train=True, regularizer=None): #[batch, seqlen, emb_size]
    seq_len = 100
    if not is_train:
        # In case of out of memory, the maximum of sentence is 600
        seq_len = 600

    #### Key Part
    # Weight from LSTM Hidden Layer to Output Layer
    weight_ho = get_weight(shape=[HIDDEN_SIZE, OUTPUT_NODE], regularizer=regularizer)
    bias_ho = get_bias(shape=[OUTPUT_NODE])

    h = tf.zeros(shape=[tf.shape(x)[0], HIDDEN_SIZE], dtype=tf.float32) #[batch, hidden_size]
    C = tf.zeros(shape=[tf.shape(x)[0], HIDDEN_SIZE], dtype=tf.float32) #[batch, hidden_size]
    
    output = [] # [batch_size, output_node] * seqlen
    seq_lst = tf.transpose(x, [1, 0, 2]) # [batch_size, seqlen, emb_size] to [seqlen, batch, emb_size]
    seq_lst = tf.unstack(seq_lst, num=seq_len) # [seqlen, batch, emb_size] to [seqlen, emb_size] * batch

    with tf.variable_scope("LSTMCell"):
        for i, inp in enumerate(seq_lst):
            if i > 0:
                # For each step, LSTM share the weights.
                tf.get_variable_scope().reuse_variables()
            
            h, C = LSTMCell(inp, h, C)

            y = tf.matmul(h, weight_ho) + bias_ho

            output.append(y)
        output = tf.transpose(output, [1, 0, 2])
    ####
    return output

