import tensorflow as tf

HIDDEN_SIZE = 1024
NUM_LAYERS = 2                       # Layers of LSTM
SRC_VOCAB_SIZE = 10000               # Source Vocabulary size
TRG_VOCAB_SIZE = 4000                # Target Vocabulary size
BATCH_SIZE = 100                     # Training batch size
KEEP_PROB = 0.8                      # Probability of node not be dropout
MAX_GRAD_NORM = 5                    # Maxumum of gradient limit
SHARE_EMB_AND_SOFTMAX = True         # Share weights with softmax and embedding layer

# ID of <sos> and <eos> in vocabulary table
# In the decode process we'll need <sos> as the first input
# and check whether the sentence reach <eos>
SOS_ID = 1
EOS_ID = 2

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

    def inference(self, src_input):
        # Although we'll only inference one sentence, but dynamic_rnn require
        # input to be a batch, so we reshape it to [1, sentence length]
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        # Encoder (this is the same one as forward)
        with tf.variable_scope("encoder"):
            _, enc_state = tf.nn.dynamic_rnn(
                self.enc_cell, src_emb, src_size, dtype=tf.float32)

        # Set maximum decode steps
        # to prevent from infinity loop in extreme situation
        MAX_DEC_LEN=100

        with tf.variable_scope("decoder/rnn/multi_rnn_cell"):
            # Use a dynamic size TensorArray to store generated sentence
            init_array = tf.TensorArray(dtype=tf.int32, size=0,
                dynamic_size=True, clear_after_read=False)
            # Insert first word <sos> as the input of decoder
            init_array = init_array.write(0, SOS_ID)
            # Construct the initial status of recurrent state
            # Recurrent state include hidden state of RNN
            # used to store the TensorArray and an integer to record decode step
            init_loop_var = (enc_state, init_array, 0)

            # Loop condition of tf.while_loop:
            # Recurrent until decode <eos> or reach the maximum steps
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), EOS_ID),
                    tf.less(step, MAX_DEC_LEN-1)))

            def loop_body(state, trg_ids, step):
                # Read the last output and get its embedding
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding,
                                                 trg_input)
                # We don't use dynamic_rnn here
                # but use dec_cell to calculate one forward step
                dec_outputs, next_state = self.dec_cell.call(
                    state=state, inputs=trg_emb)
                # Calculate every possible words' logit
                # and pick word with the maximum logist as the output os this step
                output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
                logits = (tf.matmul(output, self.softmax_weight)
                          + self.softmax_bias)
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                # Write this word into trg_ids of recurrent state
                trg_ids = trg_ids.write(step+1, next_id[0])
                return next_state, trg_ids, step+1

            # Execute tf.while_loop until return final state
            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()
