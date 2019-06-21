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


def get_test_sample(num_examples: int, num_words: int, num_features: int, num_tags: int):
    np.random.seed(87) # make sure generate same data between executions

    # Random features.
    x = np.random.rand(num_examples, num_words,
                       num_features).astype(np.float32)

    # Random tag indices representing the gold sequence.
    y = np.random.randint(
        num_tags, size=[num_examples, num_words]).astype(np.int32)

    # All sequences in this example have the same length, but they can be variable in a real model.
    sequence_lengths = np.full(num_examples, num_words - 1, dtype=np.int32)

    return x, y, sequence_lengths


if __name__ == "__main__":
    # Data settings.
    num_examples = 10
    num_words = 20
    num_features = 100
    num_tags = 5

    # Generate test sample
    x, y, sequence_lengths = get_test_sample(
        num_examples, num_words, num_features, num_tags)

    # Test CRF model
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
