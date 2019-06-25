import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os


class CRF:
    def __init__(self, num_words, num_features, num_tags, model_dir: str = "model", model_name: str = "crf"):
        self.num_words = num_words
        self.num_features = num_features
        self.num_tags = num_tags
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model_name = model_name
        self.model_path = os.path.join(model_dir, model_name)

        self.graph = tf.Graph()

    def build_model(self):
        with self.graph.as_default():
            # Add the data to the TensorFlow graph.
            self.x_t = tf.placeholder(
                tf.float32, [None, self.num_words, self.num_features])
            self.y_t = tf.placeholder(
                tf.int32, [None, self.num_words])
            self.sequence_lengths_t = tf.placeholder(
                tf.int32, [None])
            # Create a variable to hold the global_step.
            self.global_step_tensor = tf.Variable(
                0, trainable=False, name='global_step')

            # Compute unary scores from a linear layer.
            weights = tf.get_variable(
                "weights", [self.num_features, self.num_tags])
            matricized_x_t = tf.reshape(self.x_t, [-1, self.num_features])
            matricized_unary_scores = tf.matmul(matricized_x_t, weights)
            unary_scores = tf.reshape(matricized_unary_scores,
                                      [-1, self.num_words, self.num_tags])

            # Compute the log-likelihood of the gold sequences and keep the transition
            # params for inference at test time.
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                unary_scores, self.y_t, self.sequence_lengths_t)

            # Compute the viterbi sequence and score.
            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
                unary_scores, transition_params, self.sequence_lengths_t)

            # Add a training op to tune the parameters.
            loss = tf.reduce_mean(-log_likelihood)
            train_op = tf.train.GradientDescentOptimizer(
                0.01).minimize(loss, global_step=self.global_step_tensor)

            # Add ops to save and restore all the variables.
            self.saver = tf.train.Saver()

            # Sessions created in this scope will run operations from this graph
            self.session = tf.Session()

            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt and ckpt.model_checkpoint_path:  # Restore variables from disk.
                print("Found pre-trained model, restoring")
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
            else:  # No pre-trained model, initial variables
                print("Creating new model")
                self.session.run(tf.global_variables_initializer())

        self.loss = loss
        self.viterbi_sequence = viterbi_sequence
        self.train_op = train_op

    def _eval_during_train(self, predict, gold, mask, total_labels):
        correct_labels = np.sum((gold == predict) * mask)
        accuracy = 100.0 * correct_labels / float(total_labels)
        print("Accuracy: %.2f%%" % accuracy)

    # Train and evaluate the model.
    def train(self, x, y, sequence_len, epoch: int = 1000, echo_per_epoch: int = 100, save_per_epoch: int = 100):
        """ train using x, y, sequence_len, if echo_per_epoch or save_per_epoch <= 0 that means don't do it """
        # make sure the session was created in the graph
        assert self.session.graph is self.graph

        mask = (np.expand_dims(np.arange(self.num_words), axis=0) <
                np.expand_dims(sequence_len, axis=1))
        total_labels = np.sum(sequence_len)

        # Train for a fixed number of iterations.
        # for i in tqdm(range(epoch)):
        for i in range(epoch):
            tf_viterbi_sequence, _ = self.session.run(
                [self.viterbi_sequence, self.train_op], feed_dict={self.x_t: x, self.y_t: y, self.sequence_lengths_t: sequence_len})
            if echo_per_epoch > 0 and (i + 1) % echo_per_epoch == 0:
                # evaluate the model
                global_step = tf.train.global_step(
                    self.session, self.global_step_tensor)
                print("Step: %d" % global_step)
                self._eval_during_train(
                    tf_viterbi_sequence, y, mask, total_labels)
                loss = self.session.run(self.loss, feed_dict={
                                        self.x_t: x, self.y_t: y, self.sequence_lengths_t: sequence_len})
                print("Loss: %.2f%%" % loss)
            if save_per_epoch > 0 and (i + 1) % save_per_epoch == 0:
                # save the model
                self.saver.save(self.session, self.model_path,
                                global_step=self.global_step_tensor)

    def inference(self, x, sequence_len, y_to_eval=None):
        """ predict the sequence, if input y_to_eval the eval it (word granularity) """
        # make sure the session was created in the graph
        assert self.session.graph is self.graph

        tf_viterbi_sequence = self.session.run(
            self.viterbi_sequence, feed_dict={self.x_t: x, self.sequence_lengths_t: sequence_len})

        if y_to_eval is not None:
            mask = (np.expand_dims(np.arange(self.num_words), axis=0) <
                    np.expand_dims(sequence_len, axis=1))
            total_labels = np.sum(sequence_len)
            self._eval_during_train(
                tf_viterbi_sequence, y_to_eval, mask, total_labels)

        return tf_viterbi_sequence


class BiRNN_CRF:
    def __init__(self, num_words, num_features, num_tags, max_seq_len,  # Input Data
                 is_training: bool, learning_rate: float = 0.01, num_layers: int = 1, dropout_rate: float = 0.9, hidden_unit: int = 512, cell_type: str = 'lstm',  # BiRNN
                 model_dir: str = "model", model_name: str = "birnn_crf"):

        # Input Data
        self.num_words = num_words
        self.num_features = num_features  # embedding dimension
        self.num_tags = num_tags
        self.max_seq_len = max_seq_len

        # BiRNN
        self.num_layers = num_layers
        assert 1 > dropout_rate >= 0
        self.dropout_rate = dropout_rate
        self.is_training = is_training  # whether to use dropout
        self.hidden_unit = hidden_unit
        assert cell_type in ['lstm']
        self.cell_type = cell_type  # current only support lstm
        self.learning_rate = learning_rate

        # General
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model_name = model_name
        self.model_path = os.path.join(model_dir, model_name)

        self.graph = tf.Graph()

    def _get_rnn_cell(self):
        """ Get a RNN layer """
        if self.cell_type == 'lstm':
            cell = tf.contrib.rnn.LSTMCell(self.hidden_unit)
        return cell

    def _birnn(self):
        """ Get a bidirectional RNN cells (forward & backward) """
        rnn_forward = self._get_rnn_cell()
        rnn_backward = self._get_rnn_cell()
        if self.dropout_rate > 0 and self.is_training:
            rnn_forward = tf.contrib.rnn.DropoutWrapper(
                rnn_forward, output_keep_prob=(1 - self.dropout_rate))
            rnn_backward = tf.contrib.rnn.DropoutWrapper(
                rnn_backward, output_keep_prob=(1 - self.dropout_rate))
        return rnn_forward, rnn_backward

    def _birnn_layer(self, embedded_input):
        """ Get a bidirectional RNN layer(s) """
        with tf.variable_scope('birnn_layer'):
            rnn_forward, rnn_backward = self._birnn()

            if self.num_layers > 1:  # multi-layer birnn
                rnn_forward = tf.contrib.rnn.MultiRNNCell(
                    [rnn_forward] * self.num_layers, state_is_tuple=True)
                rnn_backward = tf.contrib.rnn.MultiRNNCell(
                    [rnn_backward] * self.num_layers, state_is_tuple=True)

            output, _ = tf.nn.bidirectional_dynamic_rnn(
                rnn_forward, rnn_backward, embedded_input, dtype=tf.float32)

            # concatenate two direction output
            output = tf.concat(output, axis=2)

        # [batch_size, num_steps, emb_size]
        # [num_examples, num_words, num_features] (not sure yet)
        return output

    def _project_birnn_layer(self, birnn_output):
        """ hidden layer between bidirectional RNN layer and logits """
        with tf.variable_scope('birnn_project'):
            with tf.variable_scope('hidden'):
                W = tf.get_variable("W", shape=[self.hidden_unit * 2, self.hidden_unit],
                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.hidden_unit], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(
                    birnn_output, shape=[-1, self.hidden_unit * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

        # project to score of tags
        with tf.variable_scope('logits'):
            W = tf.get_variable("W", shape=[self.hidden_unit, self.num_tags],
                                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            pred = tf.nn.xw_plus_b(hidden, W, b)

        return tf.reshape(pred, [-1, self.max_seq_len, self.num_tags])

    def _crf_layer(self, logits):
        """ CRF layer & loss """
        with tf.variable_scope('crf_loss'):
            transition_params = tf.get_variable(
                "transitions",
                shape=[self.num_tags, self.num_tags],
                initializer=tf.contrib.layers.xavier_initializer())

            # Compute the log-likelihood of the gold sequences and keep the transition
            # params for inference at test time.
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=self.y_t,
                sequence_lengths=self.sequence_lengths_t,
                transition_params=transition_params
            )

            loss = tf.reduce_mean(-log_likelihood)

        with tf.variable_scope('crf_inference'):
            # Compute the viterbi sequence and score.
            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
                potentials=logits,
                transition_params=transition_params,
                sequence_length=self.sequence_lengths_t)

        return loss, viterbi_sequence

    # I was try to set is_training here, but there are too many "tf variable reuse" problem
    def build_model(self):
        with self.graph.as_default():
            # Add the data to the TensorFlow graph.
            self.x_t = tf.placeholder(
                tf.float32, [None, self.num_words, self.num_features])
            self.y_t = tf.placeholder(
                tf.int32, [None, self.num_words])
            self.sequence_lengths_t = tf.placeholder(
                tf.int32, [None])
            # Create a variable to hold the global_step.
            self.global_step_tensor = tf.Variable(
                0, trainable=False, name='global_step')

            # not sure if this layer is good or bad for the model
            if self.is_training:
                self.x_t = tf.nn.dropout(self.x_t, self.dropout_rate)

            # BiRNN
            birnn_output = self._birnn_layer(embedded_input=self.x_t)
            # project
            logits = self._project_birnn_layer(birnn_output)

            # CRF
            loss, viterbi_sequence = self._crf_layer(logits)

            # Add a training op to tune the parameters.
            # train_op = tf.train.GradientDescentOptimizer(
            #     learning_rate=self.learning_rate).minimize(loss, global_step=self.global_step_tensor)
            train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss, global_step=self.global_step_tensor)

            # Add ops to save and restore all the variables.
            self.saver = tf.train.Saver()

            # Sessions created in this scope will run operations from this graph
            self.session = tf.Session()

            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt and ckpt.model_checkpoint_path:  # Restore variables from disk.
                print("Found pre-trained model, restoring")
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
            else:  # No pre-trained model, initial variables
                print("Creating new model")
                self.session.run(tf.global_variables_initializer())

        self.loss = loss
        self.viterbi_sequence = viterbi_sequence
        self.train_op = train_op

    def _eval_during_train(self, predict, gold, mask, total_labels):
        correct_labels = np.sum((gold == predict) * mask)
        accuracy = 100.0 * correct_labels / float(total_labels)
        print("Accuracy: %.2f%%" % accuracy)

    # Train and evaluate the model.
    def train(self, x, y, sequence_len, epoch: int = 100, echo_per_epoch: int = 10, save_per_epoch: int = 10):
        """ train using x, y, sequence_len, if echo_per_epoch or save_per_epoch <= 0 that means don't do it """
        # make sure the session was created in the graph
        assert self.session.graph is self.graph
        # make sure is in training mode (use dropout)
        assert self.is_training == True

        mask = (np.expand_dims(np.arange(self.num_words), axis=0) <
                np.expand_dims(sequence_len, axis=1))
        total_labels = np.sum(sequence_len)

        # Train for a fixed number of iterations.
        # for i in tqdm(range(epoch)):
        for i in range(epoch):
            tf_viterbi_sequence, _ = self.session.run(
                [self.viterbi_sequence, self.train_op], feed_dict={self.x_t: x, self.y_t: y, self.sequence_lengths_t: sequence_len})
            if echo_per_epoch > 0 and (i + 1) % echo_per_epoch == 0:
                # evaluate the model
                global_step = tf.train.global_step(
                    self.session, self.global_step_tensor)
                print("Step: %d" % global_step)
                self._eval_during_train(
                    tf_viterbi_sequence, y, mask, total_labels)
                loss = self.session.run(self.loss, feed_dict={
                                        self.x_t: x, self.y_t: y, self.sequence_lengths_t: sequence_len})
                print("Loss: %.2f%%" % loss)
            if save_per_epoch > 0 and (i + 1) % save_per_epoch == 0:
                # save the model
                self.saver.save(self.session, self.model_path,
                                global_step=self.global_step_tensor)

    def inference(self, x, sequence_len, y_to_eval=None):
        """ predict the sequence, if input y_to_eval the eval it (word granularity) """
        # make sure the session was created in the graph
        assert self.session.graph is self.graph
        # make sure is not in training mode (disable dropout)
        assert self.is_training == False

        tf_viterbi_sequence = self.session.run(
            self.viterbi_sequence, feed_dict={self.x_t: x, self.sequence_lengths_t: sequence_len})

        if y_to_eval is not None:
            mask = (np.expand_dims(np.arange(self.num_words), axis=0) <
                    np.expand_dims(sequence_len, axis=1))
            total_labels = np.sum(sequence_len)
            self._eval_during_train(
                tf_viterbi_sequence, y_to_eval, mask, total_labels)

        return tf_viterbi_sequence


def get_test_sample(num_examples: int, num_words: int, num_features: int, num_tags: int):
    np.random.seed(87)  # make sure generate same data between executions

    # Random features.
    x = np.random.rand(num_examples, num_words,
                       num_features).astype(np.float32)

    # Random tag indices representing the gold sequence.
    y = np.random.randint(
        num_tags, size=[num_examples, num_words]).astype(np.int32)

    # All sequences in this example have the same length, but they can be variable in a real model.
    # sequence_lengths = np.full(num_examples, num_words - 1, dtype=np.int32)
    sequence_lengths = np.full(num_examples, num_words, dtype=np.int32)

    return x, y, sequence_lengths


def CRF_model_test(x, y, sequence_lengths, num_words, num_features, num_tags):
    print("Creating init model")
    CRFModel = CRF(num_words, num_features, num_tags,
                   model_dir="model/crf_test", model_name="test_crf")
    print("Building model")
    CRFModel.build_model()
    print("Training model")
    CRFModel.train(x, y, sequence_lengths)
    print("Training again")
    CRFModel.train(x, y, sequence_lengths)
    print("Testing model")
    answer = CRFModel.inference(x, sequence_lengths)
    print(answer)
    print("Testing with evaluation")
    CRFModel.inference(x, sequence_lengths, y)


def BiRNN_CRF_model_test(x, y, sequence_lengths, num_words, num_features, num_tags):

    # calculate max sequence length
    max_seq_len = max(sequence_lengths)

    print("Creating model for training")
    BiRNNCRFModel = BiRNN_CRF(num_words, num_features, num_tags, max_seq_len, is_training=True,
                          model_dir="model/birnn_crf_test", model_name="test_birnn_crf")
    print("Building model (use dropout)")
    BiRNNCRFModel.build_model()
    print("Training model")
    BiRNNCRFModel.train(x, y, sequence_lengths)
    print("Training again")
    BiRNNCRFModel.train(x, y, sequence_lengths)

    print("Creating model for testing")
    BiRNNCRFModel = BiRNN_CRF(num_words, num_features, num_tags, max_seq_len, is_training=False,
                          model_dir="model/birnn_crf_test", model_name="test_birnn_crf")
    print("Building model (disable dropout)")
    BiRNNCRFModel.build_model()
    print("Testing model")
    answer = BiRNNCRFModel.inference(x, sequence_lengths)
    print(answer)
    print("Testing with evaluation")
    BiRNNCRFModel.inference(x, sequence_lengths, y)

if __name__ == "__main__":
    # Data settings.
    num_examples = 10
    num_words = 20
    num_features = 100
    num_tags = 5

    # Generate test sample
    x, y, sequence_lengths = get_test_sample(
        num_examples, num_words, num_features, num_tags)

    print("Test CRF Model")
    CRF_model_test(x, y, sequence_lengths, num_words, num_features, num_tags)

    print("Test BiRNN CRF Model")
    BiRNN_CRF_model_test(x, y, sequence_lengths, num_words, num_features, num_tags)
