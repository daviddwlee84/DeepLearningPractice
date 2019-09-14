# PyTorch

## Installation

```sh
pip3 install torch torchvision
```

## Important Feature

[Broadcast Semantics](https://pytorch.org/docs/stable/notes/broadcasting.html): the 1D operation can also apply on higher dimension (In short, if a PyTorch operation supports broadcast, then its Tensor arguments can be automatically expanded to be of equal sizes (without making copies of the data).)

* can't broadcast:
  * `torch.dot` (unlike `numpy.dot`)
    * [issue #138 torch dot function consistent with numpy](https://github.com/pytorch/pytorch/issues/138)

### Compare with TensorFlow

* PyTorch use dynamic computational graph while TensorFlow use static one.

## Dataset and DataLoader

* [Kaggle PyTorch Dataset and DataLoader](https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader)
* [PyTorch Data Loading and Processing Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
* [How to get mini-batches in pytorch in a clean and efficient way?](https://stackoverflow.com/questions/45113245/how-to-get-mini-batches-in-pytorch-in-a-clean-and-efficient-way)
* [A detailed example of how to generate your data in parallel with PyTorch](https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel)
* [莫煩Python - Batch Training](https://morvanzhou.github.io/tutorials/machine-learning/torch/3-05-train-on-batch/)

Customized

* [torch.utils.data](https://pytorch.org/docs/stable/data.html)
* `from torch.utils.data import DataLoader, Dataset`

Ready-made

* `from torchtext.datasets import text_classification`

## Resources

* [Get Started](https://pytorch.org/get-started/locally/)
* [PyTorch Tutorial](https://pytorch.org/tutorials/)
  * [github](https://github.com/pytorch/tutorials)
  * [**Learning PyTorch with Exmaples**](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)
    * Tensors
    * Autograd
    * nn Module
  * [Deep Learning for NLP with Pytorch](https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html)
    * [**Word Embeddings: Encoding Lexical Semantics**](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)
* [PyTorch Examples](https://github.com/pytorch/examples)
  * [Image classification (MNIST) using Convnets](https://github.com/pytorch/examples/tree/master/mnist)
  * [Word level Language Modeling using LSTM RNNs](https://github.com/pytorch/examples/tree/master/word_language_model)
  * [Natural Language Inference (SNLI) with GloVe vectors, LSTMs, and torchtext](https://github.com/pytorch/examples/tree/master/snli)
* [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)

### Tutorial

* [**yunjey/pytorch-tutorial**](https://github.com/yunjey/pytorch-tutorial) - PyTorch Tutorial for Deep Learning Researchers
* [INTERMT/Awesome-PyTorch-Chinese](https://github.com/INTERMT/Awesome-PyTorch-Chinese)
  * [PyTorch Chinese Tutorial](http://pytorchchina.com/)
* [PyTorchZeroToAll](https://github.com/hunkim/PyTorchZeroToAll) - Quick 3~4 day lecture materials for HKUST students
* [L1aoXingyu/pytorch-beginner](https://github.com/L1aoXingyu/pytorch-beginner) - pytorch tutorial for beginners
* [xiaobaoonline/pytorch-in-action](https://github.com/xiaobaoonline/pytorch-in-action) - Source code of book PyTorch機器學習從入門到實戰
* [pytorch handbook (Chinese)](https://github.com/zergtant/pytorch-handbook)

### Article

* [Understanding PyTorch with an example: a step-by-step tutorial](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e)
* [深度學習新手村：PyTorch入門](https://medium.com/pyladies-taiwan/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E6%96%B0%E6%89%8B%E6%9D%91-pytorch%E5%85%A5%E9%96%80-511df3c1c025)
  * [Basic MNIST Example](https://github.com/pytorch/examples/tree/master/mnist)
  * [github](https://github.com/pyliaorachel/MNIST-pytorch-tensorflow-eager-interactive)

### Example/Approach

BiRNN

* [**Kaggle baseline: pytorch BiLSTM**](https://www.kaggle.com/ziliwang/baseline-pytorch-bilstm)
* [**BiRNN**](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py)

Embedding

* [**PyTorch / Gensim - How to load pre-trained word embeddings**](https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings)

Attention

* [**Attention - Pytorch and Keras**](https://www.kaggle.com/mlwhiz/attention-pytorch-and-keras)

### Related Project

* [AllenNLP](https://allennlp.org/) - An open-source NLP research library, built on PyTorch
* [**torchtext**](https://github.com/pytorch/text)
  * [github](https://github.com/allenai/allennlp)
* [visionml/pytracking](https://github.com/visionml/pytracking) - Visual tracking library based on PyTorch
* [**huggingface/pytorch-transformers**](https://github.com/huggingface/pytorch-transformers)
  * [documentation](https://huggingface.co/pytorch-transformers/quickstart.html#documentation)

### Others

* [VSCode PyTorch: error message “torch has no […] member”](https://stackoverflow.com/questions/50319943/pytorch-error-message-torch-has-no-member)

## Appendix

* `torch.bmm` vs. `torch.mm` vs. `torch.matmul`
* `torch.permute` vs. `torch.transpose` vs. `torch.view`
