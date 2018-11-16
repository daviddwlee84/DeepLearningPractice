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
[MNIST Handwriting Digit](Project/MNIST)|FCNN|Tensorflow|

## Deep Learning Categories

### Technique

* Feedforward Neural Network
    * Multilayer Perceptron (MLP)
* [`Fully Connected Neural Network (FCNN)`](Notes/Technique/Fully_Connected_Neural_Network.md) - Overview of neural network training process

#### Computer Vision (CV)

* `Convolusion Neural Network (CNN)`

#### Natural Language Processing (NLP)

* `Recurrent Neural Network (RNN)`

#### Uncategorized

* `Generative Adversarial Network (GAN)`
* `Variational Autoencoder (VAE)`
* `Self-Organizing Map (SOM)`

### Learning Framework

#### Object Detection

* [You Only Look Once (YOLO)](Notes/LearningFramework/YOLO.md)

#### Text and Sequence

* Long Short Term Memory (LSTM)
* Gated Recurrent Unit (GRU) Neural Network

### Element

#### Layer

#### [Activation Function](Notes/Element/Activation_Function.md)

* Linear
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

CV

[Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision) - A curated list of awesome computer vision resources

#### Slides

* [Supervised Deep Learning](https://sites.google.com/site/deeplearningcvpr2014/ranzato_cvpr2014_DLtutorial.pdf)