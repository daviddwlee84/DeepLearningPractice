# Vanishing/Exploding Gradient

> this is not just a RNN problem: it can be a problem for all neural architectures especially *deep* ones

Conclusion

> (paper "OnOn the difficulty of training recurrent neural networks")

* Vanishing gradient: if the **largest eigenvalue of $W_h$ < 1**, then the gradient will **shrink** exponentially
* Exploding gradient: if the **largest eigenvalue of $W_h$ > 1**, ...

Problem might cause

* RNN-LM
  * learning *sequential recency* >  *sysyntactic recency*
  * the model weight are only updated only with respect to *near effects* => can't learn *long-distance dependencies* (the information (gradient) vanished)
* SGD
  * if gradient becomes too big
    * bad update => take too large a step and reach a bad parameter configuration
    * maybe result in *inf* or *nan*

Solution

* Gradient clipping: if the norm of the gradient is greater than some threshold, scale it down before applying SGD update => solution for gradient exploding
* LSTM
* Skip connections (direct connections, shortcut connections, residual connections)



* Residual connection (ResNet)
* Dense connection (DenseNet)
* Highway connection (HighwayNet)