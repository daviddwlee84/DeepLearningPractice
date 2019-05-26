# Deep Learning Practice

Basically neural network based implementation and corresponding notes.

More "general" machine learning notes will be noted in my [Machine Learning](https://github.com/daviddwlee84/MachineLearningPractice) repository.

## Environment

* Using Python 3

### Dependencies

* [`tensorflow`](https://tensorflow.org)
    * [github](https://github.com/tensorflow/tensorflow)
    * [Brief Notes](Notes/Framework/Tensorflow.md) - Placeholder, Graph, Session
    * Model Save and Restore Notes - ckpt, transfer learning
    * Data Manipulating Notes - TFRecord, Iterator
    * Multi-thread Notes
    * High-level API Notes - tf.keras, tf.layer
    * simple demos with maybe jupyter notebook?!
    * TensorFlow 2.0
* [`keras`](https://keras.io)
    * [github](https://github.com/keras-team/keras/)
    * [Brief Notes](Notes/Framework/Keras.md)

Not in this project

* [`pytorch`](https://pytorch.org/)
  * [Brief Notes](Notes/Framework/PyTorch.md)

## Project

Subject|Technique|Framework|Complexity|Remark
-------|---------|---------|----------|------
[Perceptron Practice](Project/PerceptronPractice)|SLP, MLP|Numpy|○○●●●|Truth Table (AND, OR, XOR) and Iris Dataset (simulate Keras API)
[Softmax Derivation](Project/SoftmaxDerivation)|FCNN|Numpy|○○○○●|Backpropagation of Softmax with Cross Entropy Loss
[MNIST Handwriting Digit](Project/MNIST)|FCNN|Tensorflow (and tf.keras)|○○●●●|Implement by different ways
[Semeion Handwritten Digit](Project/SemeionHandwrittenDigit)|FCNN|Tensorflow|○○○●●|Made a Tensorflow like Dataset Class
[CIFAR-10](Project/CIFAR-10)|FCNN, CNN|Tensorflow|○○●●●|Comparison of FCNN and CNN
[Chinese Named Entity Recognizer](Project/ChineseNER)|RNN, LSTM|Tensorflow|○●●●●|TODO: LSTM testing
[Flowers](Project/Flowers)|CNN|Tensorflow|○○●●●|Transfer Learning
[Fruits](Project/Fruits)|CNN|Tensorflow (and tf.layer)|○○●●●|Multi-thread training and TFRecord TODO: Try more complex model
[Trigonometric Function Prediction](Project/RNNTrigonometricFunc)|RNN|Tensorflow|○○○○●|Predict sine, cosine using LSTM
[Penn TreeBank](Project/PTB)|RNN, LSTM|Tensorflow|○○●●●|Language corpus preprocessing and training
[Chinese Neural Machine Translation](Project/ChineseNMT)|RNN, Attention|Tensorflow|○●●●●|A practice of Seq2Seq and Attention TODO: Multi-graph, Try transformer
[Dogs!](Project/Dogs)|CNN|Keras|○○●●●|Using images from ImageNet, Keras Transfer learning and Data augmentation
[2048](https://github.com/daviddwlee84/ReinforcementLearning2048)|FCNN with Policy Gradient|Tensorflow|●●●●●|Reinforcement Learning
[Online ImageNet Classifier](Project/ImgClassifierAPI)|CNN|Keras|○○●●●|(TODO) Using Keras Applications combine with RESTful API
[First TF.js](Project/TFjs)||||(TODO) Using TensorFlow.js to load pre-trained model and make prediction on the browser
[YOLO](Project/YOLO)|CNN|Tensorflow||(TODO) Real-time Object Detection
[Word Similarity](Project/WordSimilarity)||||(TODO) Word Similarity Based on Dictionary and Based on Corpus
[Text Relation Classification](https://github.com/pku-nlp-forfun/SemEval-2018-RelationClassification)|Multiple Models|Multiple Libraries|●●●●●|SemEval2018 Task 7 Semantic Relation Extraction and Classification in Scientific Papers
[Medical Corpus](Project/MedicalCorpus)|Human Labor|Naked Eyes|●●●●●|From *Chinese word segmentation* to *POS tagging* to *NER*
[Word Sense Induction](https://github.com/pku-nlp-forfun/SemEval2013-WordSenseInduction)|Multiple Models|Multiple Libraries|●●●●●|SemEval2013 Task 13 Word Sense Induction for Graded and Non-Graded Senses
[Chinese WS/POS/(NER)](https://github.com/pku-nlp-forfun/CWS_POS_NER)|||●●●●●|The "from scratch" version of the previous project ("Medical Corpus")

## Deep Learning Categories

### Technique / Network Structure

* Feedforward Neural Network
    * Multilayer Perceptron (MLP)
* [`Fully Connected Neural Network (FCNN)`](Notes/Technique/Fully_Connected_Neural_Network.md) - And an overview of neural network training process including [forward](Notes/Technique/Fully_Connected_Neural_Network.md#Forward-Propagation) and [back](Notes/Technique/Fully_Connected_Neural_Network.md#Back-Propagation) propagation
    * Dense Neural Network (DNN)

#### Image Learning

* [`Convolusion Neural Network (CNN)`](Notes/Technique/CNN.md)

#### Sequence Learning

Basic Block for Sequence Model!

* [`Recurrent Neural Network (RNN)`](Notes/Technique/RNN.md) - Basis of Sequence model
* [`Long Short Term Memory (LSTM)`](Notes/Technique/LSTM.md) - Imporvement of "memory" (brief introduce other regular RNN block)
* [`Gated Recurrent Units (GRUs)`](Notes/Technique/GRU.md)

#### [`Reinforcement Learning (RL)`](Notes/Technique/Reinforcement_Learning.md)

* `Q Learning`
* `Policy Gradient Methods (PG)`

#### Uncategorized

* `Generative Adversarial Network (GAN)`
* `Variational Autoencoder (VAE)`
* `Self-Organizing Map (SOM)`

### Learning Framework / Model

#### Object Detection

* [You Only Look Once (YOLO)](Notes/LearningFramework/YOLO.md)

#### Text and Sequence

* [Sequence-to-Sequence (seq-to-seq) (Encoder-Decoder) Architecture](Notes/LearningFramework/seq-to-seq.md) - Overview of sequence models
    * [`Bidirectional RNN (BRNN)`](Notes/LearningFramework/BRNN.md) - RNN-Based seq-to-seq
    * Convolution-based seq-to-seq
    * [`Attention Model`](Notes/LearningFramework/Attention.md) - Transformer-based seq-to-seq
    * [`Transformer`](Notes/LearningFramework/Transformer.md) - Attention Is All You Need - Transformer-based multi-headed self-attention
* Word Piece Model (WPM) aka. SentencePiece

#### [Transfer Learning in NLP](Notes/Technique/NLPTransferLearning.md)

> "Pre-training in NLP" ≈ "Embedding"

* `ELMo`
* [`BERT`](Notes/LearningFramework/BERT.md)
* GPT

#### Others

* Neural Architecture Search

## Ingredient of magic

### [Layer](Notes/Element/Layers.md)

* BatchNorm
* Convolution
* Pooling
* Fully Connected (Dense)
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

Parameter

* Learning Rate: Used to limit the amount each weight is corrected each time it is updated.
* Epochs: The number of times to run through the training data while updating the weight.

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
* Vanilla - means standard, usual, or unmodified version of something.
    * Vanilla gradient descent (aka. Batch gradient descent) - means the basic gradient descent algorithm without any bells or whistles.

[Tricks for language model](Notes/Concept/LanguageModel.md) - a sort of overview

* [Word Representation](Notes/Concept/WordRepresentation.md)
  * [Embedding](Notes/Concept/Embedding.md)
    * [Train Embedding](Notes/Concept/TrainEmbedding.md)
* CNN for NLP
* RNN for NLP

* Capsule net with GRU
    * [Kaggle kernel - Capsule net with GRU](https://www.kaggle.com/chongjiujjin/capsule-net-with-gru)
    * [Kaggle kernel - Capsule net with GRU on Preprocessed data](https://www.kaggle.com/fizzbuzz/capsule-net-with-gru-on-preprocessed-data)

### Network Framework

* LeNet - CNN
* AlexNet - CNN
* ZFNet
* VGG-Net - CNN
* GoogleNet - CNN
* ResNet - CNN
* DenseNet
* ResNeXt
* DPN (Dual Path Network)
* [CliqueNet](https://github.com/iboing/CliqueNet)

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

## Applications

### CV

* [Image Classification](Notes/Application/CV/ImageClassification.md)

### NLP

* [Basis](Notes/Application/NLP/NLPBasis.md)
    * Text segmentation
    * Part-of-speech tagging (POS tagging)
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
* [Named Entity Recognition (NER)](Notes/Application/NLP/NER.md)
* Neural Machine Translation (NMT)
    * Encoder LSTM + Decoder LSTM
    * Google NMT (GNMT)
* Speech Synthesis
    * WaveNet: A Generative Model for Raw Audio
    * [Tacotron](https://google.github.io/tacotron/): An end-to-end speech synthesis system
* Personalized Recommendation
* Machine Translation
* Sentiment classification
* [Chatbot](Notes/Application/NLP/Chatbot.md)
    * [Sequential Matching Network (SMN)](Notes/LearningFramework/SMN.md)

### Other Sequence Learning Problem

* Music generation
* DNA sequence analysis
* Video activity recognition

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
* [Graphviz](https://www.graphviz.org/) - Graph Visualization Software
  * [Keras model visualization](https://keras.io/visualization/)
  * pydot

Latex

* [HarisIqbal88/PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)

Toy

* [martisak/dotnets](https://github.com/martisak/dotnets)

## Resources

### Corpus/NLP Dataset

* SemCor
* [SENSEVAL](https://web.eecs.umich.edu/~mihalcea/senseval/)
* SemEval
  * [wiki](https://en.wikipedia.org/wiki/SemEval)

### Github Repository

* [**brightmart/text_classification**](https://github.com/brightmart/text_classification) - all kinds of text classification models and more with deep learning

### Application

* [Leon](https://getleon.ai/)
  * [leon-ai/leon](https://github.com/leon-ai/leon)

### Tutorial

#### Course

* [Tensorflow and deep learning without a PhD series by @martin_gorner](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd)
* [YSDA Natural Language Processing course](https://github.com/yandexdataschool/nlp_course)
* [**Tensorflow and deep learning without a PhD**](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd) - Martin Görner

#### Interactive Learning

* [Kaggle Learn Deep Learning](https://www.kaggle.com/learn/deep-learning)

#### MOOC

* [Stanford - CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
* [Stanford - CS244n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
    * [Hank's blog](http://www.hankcs.com/tag/cs224n/) ([github](https://github.com/hankcs/CS224n))
    * [CS224n Chinese camp](https://github.com/learning511/cs224n-learning-camp)
* [MIT Deep Learning](https://deeplearning.mit.edu/)
    * [Github](https://github.com/lexfridman/mit-deep-learning) - Tutorials, assignments, and competitions for MIT Deep Learning related courses
* [PKU - 人工智慧實踐：Tensorflow筆記](https://www.icourse163.org/course/PKU-1002536002)

#### Document

* [**DeepNotes**](https://deepnotes.io/)
    * [deepnet](https://github.com/parasdahal/deepnet) - Implementations of CNNs, RNNs and cool new techniques in deep learning from scratch
* [UFLDL Tutorial](http://ufldl.stanford.edu/tutorial/)
  * [Starter Code](http://ufldl.stanford.edu/tutorial/StarterCode/) - [github](https://github.com/amaas/stanford_dl_ex)
* [Machine Learning Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/index.html)
* [深度學習500問](https://github.com/scutan90/DeepLearning-500-questions)
* [Machine Learning Notebook](https://mlnotebook.github.io/)

#### Github

* [**Dive into Deep Learning (D2L Book)**](http://en.diveintodeeplearning.org/) ([d2l.ai](https://www.d2l.ai/index.html)) / [**動手學深度學習**](https://zh.diveintodeeplearning.org/)
    * [English github](https://github.com/diveintodeeplearning/d2l-en) / [Chinese github](https://github.com/diveintodeeplearning/d2l-zh)
* [**Machine Learning cheatsheets for Stanford's CS 229**](https://github.com/afshinea/stanford-cs-229-machine-learning)
    * [Webpage - CS 229 ― Machine Learning](https://stanford.edu/~shervine/teaching/cs-229.html)
* [**Deep Learning cheatsheets for Stanford's CS 230**](https://github.com/afshinea/stanford-cs-230-deep-learning)
    * [Webpage - CS 230 ― Deep Learning](https://stanford.edu/~shervine/teaching/cs-230.html)
* [Lambda Deep Learning Demos](https://lambda-deep-learning-demo.readthedocs.io/en/latest/)
    * [github](https://github.com/lambdal/lambda-deep-learning-demo)
* [Azure/MachineLearningNotebooks](https://github.com/Azure/MachineLearningNotebooks)

#### Slides

* [Supervised Deep Learning](https://sites.google.com/site/deeplearningcvpr2014/ranzato_cvpr2014_DLtutorial.pdf)

### Paper

#### Conference

* NAACL North American Chapter of the Association for Computational Linguistics
  * [NAACL-HLT 2019](https://naacl2019.org/)
* NPIS Neural Information Processing Systems
  * [NIPS Proceedingsβ](https://papers.nips.cc/)

### Summaries

* [StateOfTheArt.ai](https://www.stateoftheart.ai/)
* [**Papers With Code: Browse state-of-the-art**](https://paperswithcode.com/state-of-the-art)

NLP

* [Tracking Progress in Natural Language Processing](https://github.com/sebastianruder/NLP-progress)

CV

* [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision) - A curated list of awesome computer vision resources

### Article

NLP

* [初入NLP領域的一些小建議](https://zhuanlan.zhihu.com/p/59184256)

### Lexical Database

* [WordNet](https://wordnet.princeton.edu/)
  * [Open Multilingual Wordnet](http://compling.hss.ntu.edu.sg/omw/)
* [BabelNet](https://babelnet.org/)

### Other

Manipulate Github Large File (>100MB)

`.gitattributes`

* [Git large file storage](https://git-lfs.github.com/)
* [Bitbucket tutorial - Git LFS](https://www.atlassian.com/git/tutorials/git-lfs#clone-respository)
* [Configuring Git Large File Storage](https://help.github.com/articles/configuring-git-large-file-storage/)
* [Moving a file in your repository to Git Large File Storage](https://help.github.com/articles/moving-a-file-in-your-repository-to-git-large-file-storage/)
    * [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) - `brew install bfg`
        * [github](https://github.com/rtyley/bfg-repo-cleaner)
    * [Removing sensitive data from a repository](https://help.github.com/articles/removing-sensitive-data-from-a-repository/) - git filter-branch

Time measure

* [Python decorator to measure the execution time of methods](https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d)

Export Markdown

* [Pandoc](https://pandoc.org/)
  * [User Manual](https://pandoc.org/MANUAL.html)
* toDOCX
  * `pandoc -o output.docx -f markdown -t docx filename.md`
  * [PDFtoDOCX](https://pdf2docx.com/)
* toPPT
  * [Smallpdf PDF to PPT Converter](https://smallpdf.com/pdf-to-ppt)

Machine Learning/Deep Learning Platform

* [Anaconda](https://www.anaconda.com/)
* [PaddlePaddle](http://www.paddlepaddle.org/)
  * [github](https://github.com/PaddlePaddle/Paddle)

## Deprecated notes

* [`h5py`](https://www.h5py.org/) - HDF5 for Python: To store model in HDF5 binary data format
* [`pyyaml`](https://pyyaml.org/) - PyYAML: YAML framework
