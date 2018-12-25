import numpy as np
import tensorflow as tf

TRAIN_DATA = "train_data/ptb.train"
EVAL_DATA = "train_data/ptb.valid"
TEST_DATA = "train_data/ptb.test"
HIDDEN_SIZE = 300                 # Hidden layer size
NUM_LAYERS = 2                    # Layers of LSTM
VOCAB_SIZE = 10000
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35               # A cut length of training data

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1                 # A cut length of test data
NUM_EPOCH = 5
LSTM_KEEP_PROB = 0.9              # Probability of a LSTM node not be dropout
EMBEDDING_KEEP_PROB = 0.9         # Probability of word vector not be dropout
MAX_GRAD_NORM = 5                 # Used to control the limit of gradient expanding speed
SHARE_EMB_AND_SOFTMAX = True      # Share the parameter between softmax and embedding layer

class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        
        # Input and output placeholder with shape [batch_size, num_steps]
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        
        # Define LSTM Neural Network with dropout
        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0
        lstm_cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
                output_keep_prob=dropout_keep_prob)
            for _ in range(NUM_LAYERS)]     
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)            
        
        # Initial state => zero vector
        # (used in the first batch of every epoch)
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # Define the embedding matrix
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
        
        # Transfer input words into embedding
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        
        # Only use dropout while training
        if is_training:
            inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)
 
        # Define output (collect output of LSTM in different time step
        # then output to softmax layer at once
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output) 
        # Unfold output list to shape of [batch, hidden_size*num_steps]
        # then reshape it to [batch*numsteps, hidden_size]
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        # Softmax layer: Transfer every position of RNN to each words' logits
        if SHARE_EMB_AND_SOFTMAX:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias
        
        # Define loss function and Average loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]),
            logits=logits)
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state
        
        # Only do backprop while training
        if not is_training: return

        trainable_variables = tf.trainable_variables()
        # Control the volume of gradient
        # Define the optimizer and training operation
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        self.train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables)) 

# Execute train_op and return perplexity of the data
def run_epoch(session, model, batches, train_op, output_log, step):
    # variables used to calculate average perplexity
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state) 
    # Train an epoch。
    for x, y in batches:
        # Execute train_op and calculate loss
        # The cross entropy loss is the probability of whether the next word is the given word
        cost, state, _ = session.run(
             [model.cost, model.final_state, train_op],
             {model.input_data: x, model.targets: y,
              model.initial_state: state})
        total_costs += cost
        iters += model.num_steps

        # Output log
        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (
                  step, np.exp(total_costs / iters)))
        step += 1

    # Return perplexity of the data
    return step, np.exp(total_costs / iters)

# Read embedding from file
def load_embedding(filename):
    with open(filename, "r") as fin:
        # Read entire file as a string
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]  # Transfer string to int
    return id_list

def generate_batches(id_list, batch_size, num_step):
    # Calculate batch number
    # (words for each batch is batch_size * num_step)
    num_batches = (len(id_list) - 1) // (batch_size * num_step)

    # Reshape data dimension into [batch_size, num_batches * num_step]
    data = np.array(id_list[: num_batches * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batches * num_step])
    # Split data into numbers of batch
    data_batches = np.split(data, num_batches, axis=1)

    # Repeat the operations but shift one position as the prediction label
    label = np.array(id_list[1 : num_batches * batch_size * num_step + 1]) 
    label = np.reshape(label, [batch_size, num_batches * num_step])
    label_batches = np.split(label, num_batches, axis=1)

    # Return training data and label with length of num_batches
    return list(zip(data_batches, label_batches)) 

def main():
    # Define initializer
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    
    # Define training model
    with tf.variable_scope("language_model", 
                           reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    # Define evaluation model
    # (it share the parameter with train_model but without dropout)
    with tf.variable_scope("language_model",
                           reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    # Training phase
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        train_batches = generate_batches(
            load_embedding(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
        eval_batches = generate_batches(
            load_embedding(EVAL_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)
        test_batches = generate_batches(
            load_embedding(TEST_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)

        step = 0
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            step, train_pplx = run_epoch(session, train_model, train_batches,
                                         train_model.train_op, True, step)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_pplx))

            _, eval_pplx = run_epoch(session, eval_model, eval_batches,
                                     tf.no_op(), False, 0)
            print("Epoch: %d Eval Perplexity: %.3f" % (i + 1, eval_pplx))

        _, test_pplx = run_epoch(session, eval_model, test_batches,
                                 tf.no_op(), False, 0)
        print("Test Perplexity: %.3f" % test_pplx)

if __name__ == "__main__":
    main()
