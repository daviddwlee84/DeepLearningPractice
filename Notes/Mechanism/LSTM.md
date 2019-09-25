# Long Short Term Memory (LSTM)

## Brief Description

* A special kind of RNN, capable of learning long-term dependencies.
* LSTMs are explicitly designed to avoid the long-term dependency problem.

## Basic Structure

![The repeating module in an LSTM contains four interacting layers](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

![notations](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM2-notation.png)

![Wiki1](https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Long_Short-Term_Memory.svg/800px-Long_Short-Term_Memory.svg.png)

![Wiki2](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/The_LSTM_cell.png/800px-The_LSTM_cell.png)

## The Key Component Behind LSTMs

### Cell State

* The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. => For information to just flow along it unchanged!

### Gate

* The ability for LSTM to remove or add information to cell state.
* A way to optionally let information through.

They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.

#### Sigmoid Neural Layer

Sigmoid output between 0 and 1: Describing how much of each component should be let through

* 0: Let nothing through (i.e. Completely get rid of this)
* 1: Let everything through (i.e. Completely Keep this)

## Gates (LSTM steps)

### First. Forget Gate

How much we want to forget.

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

> Multiply the old state by $f_t$ to forget the things we decided to forget.

### Second. Input Gate

#### Input gate layer

Decides which values we'll update.

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

#### Tanh layer

Creates a vector of new candidate values, $\tilde{C_t}$, that could be added to the state.

$$
\tilde{C_t} = \operatorname{tanh}(W_C \cdot[h_{t-1}, x_t] + b_C)
$$

#### update old state

Update the old cell state, $C_{t-1}$, into the new cell state $C_t$.

> Add $i_t \times \tilde{C_t}$. This is the new candidate value scaled by how much we decided to update each state value.

### Final. Ouput Gate

Decide what we're going to output.

This output will be based on the **cell state**, but will be a filtered version.

1. Run a sigmoid layer which decides what parts of the cell state we're going to output.
    $$
    o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
    $$
2. Put the cell state through **tanh** (to push the values to be between -1 and 1)
    $$
    \operatorname{tanh}(C_t)
    $$
3. Multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.
    $$
    h_t = o_t \times \operatorname{tanh}(C_t)
    $$

(Output to both "output" and "the next state"(i.e. Itself))

## Variants on LSTM (Different Gate Strategy)

TBD

### Peephole LSTM

![Wiki Peephole](https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Peephole_Long_Short-Term_Memory.svg/300px-Peephole_Long_Short-Term_Memory.svg.png)

### Gated Recurrent Unit (GRU)

### Others

* Depth Gated RNNs
* Clockwork RNNs
* Grid LSTMs
* using RNNs in generative models.

## Next Steps of LSTM

**Attention**! To let every step of an RNN pick information to look at from some larger collection of information.

## Reference

* [Wiki - Long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory)
* [**An exploration of dropout with LSTMs**](https://www.danielpovey.com/files/2017_interspeech_dropout.pdf)

### Article

* [**Udnderstanding LSTM Networks**](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [試談RNN中Gate的變遷](https://wyydsb.xin/other/rnn.html)

### TensorFlow

* [BasicLSTMCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/BasicLSTMCell)
