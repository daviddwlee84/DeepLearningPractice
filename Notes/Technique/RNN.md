# Recurrent Neural Network

## Brief Description

* RNNs can use their internal state (memory) to process sequences of inputs.
* NN with loops in it, allowing information to persist.
    * A loop allows information to be passed form one stop of the network to the next.

### Quick View

## Basic Structure

![An unrolled recurrent neural network](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

![The repeating module in a standard RNN contains a single layer](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)

## The Problem of Long-Term Dependencies

* It's entirely possible for the gap between the relevant information and the point where it is needed to become vary large.
* As the gap grows, RNNs become unable to learn to connect the information.

> Solution --> [LSTM](LSTM.md)

## Resources

* [Recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) (not Recursive NN)

### Arcitle

* [試談RNN中Gate的變遷](https://wyydsb.xin/other/rnn.html)

### TensorFlow

* [BasicRNNCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/BasicRNNCell)
* [RNNCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/RNNCell)
