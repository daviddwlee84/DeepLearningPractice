# Deep Learning Practice

Basically neural network based implementation and corresponding notes.

More "general" machine learning notes will be noted in my [Machine Learning](https://github.com/daviddwlee84/MachineLearningPractice) repository.

## Environment

* Using Python 3

### Dependencies

* [`tensorflow`](https://tensorflow.org)
    * [github](https://github.com/tensorflow/tensorflow)
    * [Brief Notes](Notes/Framework/Tensorflow.md)
    * `tf.keras`: Implementation of the Keras API meant to be a high-level API for TensorFlow.
    * [`h5py`](https://www.h5py.org/) - HDF5 for Python: To store model in HDF5 binary data format
    * [`pyyaml`](https://pyyaml.org/) - PyYAML: YAML framework
* [`keras`](http://keras.io/)
    * [github](https://github.com/keras-team/keras/)
    * [Brief Notes](Notes/Framework/Keras.md)

## Project

Subject|Technique|Framework|Remark
-------|---------|---------|------
[Perceptron Practice](Project/PerceptronPractice)|MLP|Numpy|XOR
[Softmax Deduction](Project/SoftmaxDeduction)|FCNN|Numpy|with Cross Entropy
[MNIST Handwriting Digit](Project/MNIST)|FCNN|Tensorflow, Keras|Implement by different ways
[CIFAR-10](Project/CIFAR-10)|FCNN|Tensorflow|

## Deep Learning Categories

### Technique / Network Structure

* Feedforward Neural Network
    * Multilayer Perceptron (MLP)
* [`Fully Connected Neural Network (FCNN)`](Notes/Technique/Fully_Connected_Neural_Network.md) - And an overview of neural network training process including [forward](Notes/Technique/Fully_Connected_Neural_Network.md#Forward-Propagation) and [back](Notes/Technique/Fully_Connected_Neural_Network.md#Back-Propagation) propagation
    * Dense Neural Network (DNN)

#### Computer Vision (CV)

* `Convolusion Neural Network (CNN)`

#### Natural Language Processing (NLP)

* `Recurrent Neural Network (RNN)`

#### Uncategorized

* [`Reinforcement Learning (RL)`](Notes/Technique/Reinforcement_Learning.md)
    * `Q Learning`
    * `Policy Gradient Methods (PG)`
* `Generative Adversarial Network (GAN)`
* `Variational Autoencoder (VAE)`
* `Self-Organizing Map (SOM)`

### Learning Framework / Model

#### Object Detection

* [You Only Look Once (YOLO)](Notes/LearningFramework/YOLO.md)

#### Text and Sequence

* Long Short Term Memory (LSTM)
* [Sequence-to-Sequence (seq-to-seq)](Notes/LearningFramework/seq-to-seq.md)
    * RNN-Based seq-to-seq
    * Convolution-based seq-to-seq
    * Transformer-based seq-to-seq
    * Attention Is All You Need
    * BERT
* Gated Recurrent Unit (GRU) Neural Network
* Word Piece Model (WPM) aka. SentencePiece

#### Others

* Neural Architecture Search

## Ingredient of magic

### [Layer](Notes/Element/Layers.md)

* BatchNorm
* Convolution
* Pooling
* Dropout
* Linear
* LSTM
* RNN

General speaking

* Input
* Hidden
* Output

### [Activation Function](Notes/Element/Activation_Function.md)

* Sigmoid
* Hyperbolic Tangent
* Rectified Linear Unit (ReLU)
* Leaky ReLU
* Softmax

### [Loss Function](Notes/Element/Loss_Function.md)

* Cross-Entropy
* Hinge
* Huber
* Kullback-Leibler
* MAE (L1)
* MSE (L2)

### [Optimizer / Optimization Algorithm](Notes/Element/Optimizer.md)

- Exponential Moving Average (Exponentially Weighted Moving Average)

* Adadelta
* Adagrad
* Adam
* Conjugate Gradients
* BFGS
* Momentum
* Nesterov Momentum
* Newton’s Method
* RMSProp
* Stochastic Gradient Descent (SGD)

### Regularization

* Data Augmentation
* Dropout
* Early Stopping
* Ensembling
* Injecting Noise
* L1 Regularization
* L2 Regularization

## Common Concept

### Terminology / Tricks

* one-hot encoding
* ground truth
* Data Parallelism
* Word Embedding
* Word2Vec
* Vanilla - means standard, usual, or unmodified version of something.
    * Vanilla gradient descent (aka. Batch gradient descent) - means the basic gradient descent algorithm without any bells or whistles.

### Network Framework

* LeNet
* AlexNet
* ZFNet
* VGG-Net
* GoogLeNet
* ResNet
* DenseNet
* ResNeXt
* DPN（Dual Path Network）

### Programming Framework

Framework |Organization|Support Language|Remark
----------|------------|----------------|------
TensorFlow|Google|Python, C++, Go, JavaScript, ...|
Keras|fchollet|Python|on top of TensorFlow, CNTK, or Theano
PyTorch|Facebook|Python|
CNTK|Microsoft|C++|
OpenNN||C++|
Caffe|BVLC|C++, Python|
MXNet|DMLC|Python, C++, R, ...|
Torch7|Facebook|Lua|
Theano|U. Montreal|Python|
Deeplearning4J|DeepLearning4J|Java, Scala|
Leaf|AutumnAI|Rust|
Lasagne|Lasagne|Python|
Neon|NervanaSystems|Python|

### Problem - Solution

* [Vanishing / Exploding gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
    * Solutions:
        * Multi-level hierarchy
        * LSTM
        * Faster hardware
        * Residual networks
        * Other activation functions

## Applications

### CV

### NLP

* Speech Recognition
    * End-to-End Models:
        * (Traditional --> HMM)
        * CTC
        * RNN Transducer
        * Attention-based Model
    * Improved attention
        * Single head attention
        * Multi-headed attention
    * Word Pieces
    * Sequence-Training
        * Beam-Search Decoding Based EMBR
* Neural Machine Translation (NMT)
    * Encoder LSTM + Decoder LSTM
    * Google NMT (GNMT)
* Speech Synthesis
    * WaveNet: A Generative Model for Raw Audio
    * [Tacotron](https://google.github.io/tacotron/): An end-to-end speech synthesis system
* Personalized Recommendation

## Books Recommendation

* [MIT Deep Learning](https://www.deeplearningbook.org/)
    * [github pdf](https://github.com/janishar/mit-deep-learning-book-pdf)
    * [chinese translation pdf](https://github.com/exacity/deeplearningbook-chinese)
        * [online reading](https://exacity.github.io/deeplearningbook-chinese/)
* [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)

## Tools

### Visualize Drawing Tools

* [NN-SVG](http://alexlenail.me/NN-SVG/) - FCNN, LeNet, AlexNet style
    * [github](https://github.com/zfrenchee/NN-SVG)
* [draw.io](https://www.draw.io/)
* [Netscope](https://ethereon.github.io/netscope/quickstart.html)

Latex

* [HarisIqbal88/PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)

Toy

* [martisak/dotnets](https://github.com/martisak/dotnets)

## Resources

### Tutorial

#### Course

* [Tensorflow and deep learning without a PhD series by @martin_gorner](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd)
* [YSDA Natural Language Processing course](https://github.com/yandexdataschool/nlp_course)
* [**Tensorflow and deep learning without a PhD**](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd) - Martin Görner

#### Interactive Learning

* [Kaggle Learn Deep Learning](https://www.kaggle.com/learn/deep-learning)

#### MOOC

* [Stanford - CS231n](http://cs231n.stanford.edu/)
* [PKU - 人工智慧實踐：Tensorflow筆記](https://www.icourse163.org/course/PKU-1002536002)

#### Document

* [Learning TensorFlow](https://learningtensorflow.com/)
* [**DeepNotes**](https://deepnotes.io/)
    * [deepnet](https://github.com/parasdahal/deepnet) - Implementations of CNNs, RNNs and cool new techniques in deep learning from scratch
* [Machine Learning Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/index.html)

#### Slides

* [Supervised Deep Learning](https://sites.google.com/site/deeplearningcvpr2014/ranzato_cvpr2014_DLtutorial.pdf)

### Summaries

NLP

* [Tracking Progress in Natural Language Processing](https://github.com/sebastianruder/NLP-progress)

CV

* [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision) - A curated list of awesome computer vision resources
