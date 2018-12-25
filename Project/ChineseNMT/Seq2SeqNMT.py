import tensorflow as tf

HIDDEN_SIZE = 1024
NUM_LAYERS = 2                       # Layers of LSTM
SRC_VOCAB_SIZE = 10000               # Source Vocabulary size
TRG_VOCAB_SIZE = 4000                # Target Vocabulary size
BATCH_SIZE = 100                     # Training batch size
KEEP_PROB = 0.8                      # Probability of node not be dropout
MAX_GRAD_NORM = 5                    # Maxumum of gradient limit
SHARE_EMB_AND_SOFTMAX = True         # Share weights with softmax and embedding layer

class Seq2SeqNMTModel(object):
    def __init__(self):
        # Define Encoder and decoder
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
          [tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE, name='basic_lstm_cell')
           for _ in range(NUM_LAYERS)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
          [tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE, name='basic_lstm_cell') 
           for _ in range(NUM_LAYERS)])

        # Embedding of source and target language
        self.src_embedding = tf.get_variable(
            "src_emb", [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable(
            "trg_emb", [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        # Weights of softmax layer
        if SHARE_EMB_AND_SOFTMAX:
           self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
           self.softmax_weight = tf.get_variable(
               "weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable(
            "softmax_bias", [TRG_VOCAB_SIZE])

    # Define compute graph in forward propgation
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]
    
        # Transfer input and output words to embedding
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)
        
        # Dropout embedding
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        # Construct encoder
        # Encoder read embeddings in every position and output the enc_state of last state
        # Encoder is a double layer LSTM
        # thus enc_state contain two LSTMStateTuple class, each for each layer
        with tf.variable_scope("encoder"): 
            _, enc_state = tf.nn.dynamic_rnn(
                self.enc_cell, src_emb, src_size, dtype=tf.float32)

        # Construct decoder
        # Decoder read embeddings in every position and output the dec_state
        # for every output of last layer LSTM
        # Output dimension of dec_outputs is [batch_size, max_time, HIDDEN_SIZE]
        with tf.variable_scope("decoder"):
            dec_outputs, _ = tf.nn.dynamic_rnn(
                self.dec_cell, trg_emb, trg_size, initial_state=enc_state)
                # initial_state is used to initialize the hidden state of first step

        # Calculate log perplexity of decoder
        output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(trg_label, [-1]), logits=logits)

        # When calculate average loss, we need to set weights to 0
        # to prevent interfere of prediction caused by illegal position
        label_weights = tf.sequence_mask(
            trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)
        
        # Define backprop
        trainable_variables = tf.trainable_variables()

        grads = tf.gradients(cost / tf.to_float(batch_size),
                             trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables))

        return cost_per_token, train_op
