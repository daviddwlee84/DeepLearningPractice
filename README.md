# Deep Learning Practice

Basically neural network based implementation and corresponding notes.

More "general" machine learning notes will be noted in my [Machine Learning](https://github.com/daviddwlee84/MachineLearningPractice) repository.

## Environment

* Using Python 3

### Dependencies

* [`tensorflow`](https://tensorflow.org)
    * [github](https://github.com/tensorflow/tensorflow)
    * [Brief Notes](Notes/Framework/Tensorflow.md)

## Project

Subject|Technique|Framework|Remark
-------|---------|---------|------
[Perceptron Practice](Project/PerceptronPractice)|MLP|Numpy|XOR
[Softmax Deduction](Project/SoftmaxDeduction)|FCNN|Numpy|with Cross Entropy
[MNIST Handwriting Digit](Project/MNIST)|FCNN|Tensorflow|actual image vs. online dataset
[CIFAR-10](Project/CIFAR-10)|FCNN|Tensorflow|

## Deep Learning Categories

### Technique

* Feedforward Neural Network
    * Multilayer Perceptron (MLP)
* [`Fully Connected Neural Network (FCNN)`](Notes/Technique/Fully_Connected_Neural_Network.md) - Overview of neural network training process
    * Dense Neural Network (DNN)

#### Computer Vision (CV)

* `Convolusion Neural Network (CNN)`

#### Natural Language Processing (NLP)

* `Recurrent Neural Network (RNN)`

#### Uncategorized

* [`Reinforcement Learning (RL)`](Notes/Technique/Reinforcement_Learning.md)
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

### Ingredient of magic

#### Layer

#### [Activation Function](Notes/Element/Activation_Function.md)

* Sigmoid
* Hyperbolic Tangent
* Rectified Linear Unit (ReLU)
* Leaky ReLU
* Softmax

#### [Loss Function](Notes/Element/Loss_Function.md)

* Cross-Entropy
* Hinge
* Huber
* Kullback-Leibler
* MAE (L1)
* MSE (L2)

#### [Forward Propagation](Notes/Element/Forward_Propagation.md)

#### [Back Propagation](Notes/Element/Back_Propagation.md)

#### Optimizer

* Adadelta
* Adagrad
* Adam
* Conjugate Gradients
* BFGS
* Momentum
* Nesterov Momentum
* Newton’s Method
* RMSProp
* SGD

#### Regularization

* Data Augmentation
* Dropout
* Early Stopping
* Ensembling
* Injecting Noise
* L1 Regularization
* L2 Regularization

### Common Sense / Terminology / Tricks

* one-hot encoding
* ground truth
* Data Parallelism
* Word Embedding
* Word2Vec

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

Framework |Organization|Support Language
----------|------------|-----------------
TensorFlow|Google|Python, C++, Go, JavaScript, ...
Keras|fchollet|Python
PyTorch|Facebook|Python
CNTK|Microsoft|C++
OpenNN||C++
Caffe|BVLC|C++, Python
MXNet|DMLC|Python, C++, R, ...
Torch7|Facebook|Lua
Theano|U. Montreal|Python
Deeplearning4J|DeepLearning4J|Java, Scala
Leaf|AutumnAI|Rust
Lasagne|Lasagne|Python
Neon|NervanaSystems|Python

## Problem - Solution

* Vanishing gradient problem
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

#### Interactive Learning

* [Kaggle Learn Deep Learning](https://www.kaggle.com/learn/deep-learning)

#### MOOC

* [Stanford - CS231n](http://cs231n.stanford.edu/)
* [PKU - 人工智慧實踐：Tensorflow筆記](https://www.icourse163.org/course/PKU-1002536002)

#### Document

* [Learning TensorFlow](https://learningtensorflow.com/)
* [**DeepNotes**](https://deepnotes.io/)
    * [deepnet](https://github.com/parasdahal/deepnet) - Implementations of CNNs, RNNs and cool new techniques in deep learning from scratch

NLP

* [YSDA Natural Language Processing course](https://github.com/yandexdataschool/nlp_course)
* [Tracking Progress in Natural Language Processing](https://github.com/sebastianruder/NLP-progress)

CV

* [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision) - A curated list of awesome computer vision resources

#### Slides

* [Supervised Deep Learning](https://sites.google.com/site/deeplearningcvpr2014/ranzato_cvpr2014_DLtutorial.pdf)