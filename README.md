# Deep Learning Practice

Basically neural network based implementation and corresponding notes.

More "general" machine learning notes will be noted in my [Machine Learning](https://github.com/daviddwlee84/MachineLearningPractice) repository.

## Environment

* Using Python 3

### Dependencies

* [`tensorflow`](https://tensorflow.org)
    * [github](https://github.com/tensorflow/tensorflow)
    * [Brief Notes](Notes/Framework/Tensorflow.md) - Placeholder, Graph, Session
    * [TensorFlow 2.0 Notes](Notes/Framework/Tensorflow2.0.md)
    * Model Save and Restore Notes - ckpt, transfer learning
    * Data Manipulating Notes - TFRecord, Iterator
    * Multi-thread Notes
    * High-level API Notes - tf.keras, tf.layer
    * simple demos with maybe jupyter notebook?!
* [`keras`](https://keras.io)
    * [github](https://github.com/keras-team/keras/)
    * [Brief Notes](Notes/Framework/Keras.md)
* [`pytorch`](https://pytorch.org/)
  * [github](https://github.com/pytorch/pytorch)
  * [Brief Notes](Notes/Framework/PyTorch.md)
  * torch friends
    * [`tensorboardX`](https://github.com/lanpa/tensorboardX) - tensorboard for pytorch (and chainer, mxnet, numpy, ...)
    * [`pytorch-lightning`](https://github.com/williamFalcon/pytorch-lightning) - The lightweight PyTorch wrapper for ML researchers. Scale your models. Write less boilerplate
    * [`tnt`](https://github.com/pytorch/tnt) - is torchnet for pytorch, supplying you with different metrics (such as accuracy) and abstraction of the train loop
    * [`inferno`](https://github.com/inferno-pytorch/inferno/) and [`torchsample`](https://github.com/ncullen93/torchsample/) - attempt to model things very similar to Keras and provide some tools for validation
    * [`skorch`](https://github.com/skorch-dev/skorch) - is a scikit-learn wrapper for pytorch that lets you use all the tools and metrics from sklearn

## Project

### PKU Courses and Some side projects

* Basically based on TensorFlow 1.x and Keras
* Begin with the most basic model > CV > NLP

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
[Text Relation Classification](https://github.com/pku-nlp-forfun/SemEval-2018-RelationClassification)|Multiple Models|Multiple Libraries|●●●●●|SemEval2018 Task 7 Semantic Relation Extraction and Classification in Scientific Papers
[Medical Corpus](Project/MedicalCorpus)|Human Labor|Naked Eyes|●●●●●|From *Chinese word segmentation* to *POS tagging* to *NER*
[Word Sense Induction](https://github.com/pku-nlp-forfun/SemEval2013-WordSenseInduction)|Multiple Models|Multiple Libraries|●●●●●|SemEval2013 Task 13 Word Sense Induction for Graded and Non-Graded Senses
[Chinese WS/POS/(NER)](https://github.com/pku-nlp-forfun/CWS_POS_NER)|RNN, CRF|TansorFlow|●●●●●|The "from scratch" version of the previous project ("Medical Corpus") ([paper](https://github.com/pku-nlp-forfun/SemEval2013-Task13-Paper))
[Toxicity Classification](Project/ToxicityClassification)|BiLSTM|Keras|●●●●●|Jigsaw Unintended Bias in Toxicity Classification - Detect toxicity across a diverse range of conversations
[CWS/NER](Project/CWSNER)|RNN, CRF|TensorFlow|●●●●●|The sequence labeling model on the classic Chinese NLP task

### NLP PyTorch

* Basically based on PyTorch and most of the contents are NLP

Subject|Technique|Framework|Complexity|Remark
-------|---------|---------|----------|------
[Machine Translation](Project/MachineTranslation)|RNN, Transformer|PyTorch|●●●●●|Machine translation model from Chinese to English based on WMT17 corpus

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
* [`BERT`](Notes/SOTA_Models/BERT.md)
* GPT
  * [openai/gpt-2](https://github.com/openai/gpt-2)
  * [OpenAI GPT-2: An Almost Too Good Text Generator - YouTube](https://www.youtube.com/watch?v=8ypnLjwpzK8)
* XLNet

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

Big Pucture: Machine Learning vs. Deep Learning

[![ML vs DL](https://content-static.upwork.com/blog/uploads/sites/3/2017/06/27095812/image-16.png)](https://www.upwork.com/hiring/for-clients/log-analytics-deep-learning-machine-learning/)

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

### Dataset/Corpus

#### Corpus/NLP Dataset

* SemCor
* [SENSEVAL](https://web.eecs.umich.edu/~mihalcea/senseval/)
* SemEval
  * [wiki](https://en.wikipedia.org/wiki/SemEval)
* [WMT17](http://www.statmt.org/wmt17/)
  * [News](http://www.statmt.org/wmt17/translation-task.html)
    * Chinese-English

#### Animate Dataset

* [nico-opendata](https://nico-opendata.jp/en/index.html)
* [Danbooru2018](https://www.gwern.net/Danbooru2018) - A Large-Scale Crowdsourced and Tagged Anime Illustration Dataset
* [MyAnimeList Dataset](https://www.kaggle.com/azathoth42/myanimelist)
  * [MyAnimeList](https://myanimelist.net/)

### Github Repository

Example

* [hadikazemi/Machine-Learning](https://github.com/hadikazemi/Machine-Learning)

Summary

* [**brightmart/text_classification**](https://github.com/brightmart/text_classification) - all kinds of text classification models and more with deep learning

### Application

* [Leon](https://getleon.ai/)
  * [leon-ai/leon](https://github.com/leon-ai/leon)

### Mature Tools

NLP

* Chinese
  * jieba
* English
  * [spaCy](https://spacy.io/) - Industrial-Strength Natural Language Processing in Python
    * [explosion/spaCy](https://github.com/explosion/spaCy)
  * gensim
  * nltk
  * [fairseq](https://github.com/facebookresearch/fairseq) - Facebook AI Research Sequence-to-Sequence Toolkit

### Tutorial

#### Course

* [Tensorflow and deep learning without a PhD series by @martin_gorner](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd)
* [YSDA Natural Language Processing course](https://github.com/yandexdataschool/nlp_course)
* [**Tensorflow and deep learning without a PhD**](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd) - Martin Görner
* [**dataflowr | deep learning courses**](https://mlelarge.github.io/dataflowr-web/) - [github](https://github.com/mlelarge/dataflowr)
  * [Hands-on tour to deep learning with PyTorch](https://mlelarge.github.io/dataflowr-web/cea_edf_inria.html)

#### Interactive Learning

* [**Kaggle Learn Deep Learning**](https://www.kaggle.com/learn/deep-learning)
* [**Intel AI Developer Program - AI Courses**](https://software.intel.com/en-us/ai/courses)
  * [Natural Language Processing](https://software.intel.com/en-us/ai/courses/natural-language-processing)
  * [Computer Vision](https://software.intel.com/en-us/ai/courses/computer-vision)
  * [Deep Learning](https://software.intel.com/en-us/ai/courses/deep-learning)

#### MOOC

* [Stanford - CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
* [Stanford - CS244n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
  * Winter 2019 - first time using PyTorch
    * [Videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)
  * [Winter 2017](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/) - using TensorFlow
    * [Videos](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
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
  * [English github](https://github.com/d2l-ai/d2l-en) / [Chinese github](https://github.com/diveintodeeplearning/d2l-zh)
  * [PyTorch](https://github.com/dsgiitr/d2l-pytorch) / [PyTorch (Chinese)](https://github.com/ShusenTang/Dive-into-DL-PyTorch)
  * [Numpy](http://numpy.d2l.ai/)
  * [STAT 157 UC Berkeley](https://github.com/d2l-ai/berkeley-stat-157)
* [**Machine Learning cheatsheets for Stanford's CS 229**](https://github.com/afshinea/stanford-cs-229-machine-learning)
    * [Webpage - CS 229 ― Machine Learning](https://stanford.edu/~shervine/teaching/cs-229.html)
* [**Deep Learning cheatsheets for Stanford's CS 230**](https://github.com/afshinea/stanford-cs-230-deep-learning)
    * [Webpage - CS 230 ― Deep Learning](https://stanford.edu/~shervine/teaching/cs-230.html)
* [**aymericdamien/TensorFlow-Examples**](https://github.com/aymericdamien/TensorFlow-Examples) - TensorFlow Tutorial and Examples for Beginners (support TF v1 & v2)
* [**graykode/nlp-tutorial**](https://github.com/graykode/nlp-tutorial) - Natural Language Processing Tutorial for Deep Learning Researchers
* [**Microsoft Natural Language Processing Best Practices & Examples**](https://github.com/microsoft/nlp)
* [Microsoft AI education materials for Chinese students, teachers and IT professionals](https://github.com/microsoft/ai-edu)
* [Lambda Deep Learning Demos](https://lambda-deep-learning-demo.readthedocs.io/en/latest/)
    * [github](https://github.com/lambdal/lambda-deep-learning-demo)
* [Azure/MachineLearningNotebooks](https://github.com/Azure/MachineLearningNotebooks)
* [smilelight/lightNLP](https://github.com/smilelight/lightNLP)
* [RasaHQ/rasa](https://github.com/RasaHQ/rasa) - Open source machine learning framework to automate text- and voice-based conversations

#### Slides

* [Supervised Deep Learning](https://sites.google.com/site/deeplearningcvpr2014/ranzato_cvpr2014_DLtutorial.pdf)

### Paper

* [**2019-2020 International Conferences in Artificial Intelligence, Machine Learning, Computer Vision, Data Mining, Natural Language Processing and Robotics**](https://jackietseng.github.io/conference_call_for_paper/)
* [自然語言處理領域國內外著名會議和期刊](http://deepon.me/2018/10/02/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E9%A2%86%E5%9F%9F%E5%9B%BD%E5%86%85%E5%A4%96%E8%91%97%E5%90%8D%E4%BC%9A%E8%AE%AE%E5%92%8C%E6%9C%9F%E5%88%8A/)
* [NLP領域國際頂會（ACL/EMNLP/NAACLl等）的難度如何？](https://www.zhihu.com/question/266242639/answer/312713059)

* Level Criteria
  * CCF level: only used in China
  * H-index
    * [h-index - Wikipedia](https://en.wikipedia.org/wiki/H-index)
    * [h-index：簡單易懂的評估指標，呈現出多數文章的被引用表現 | 臺大圖書館參考服務部落格](http://tul.blog.ntu.edu.tw/archives/2485)

#### Conference

NLP

* ACL Association for Computational Linguistics
  * [ACL 2020](https://acl2020.org/)
* EMNLP
  * [roomylee/EMNLP-2019-Papers](https://github.com/roomylee/EMNLP-2019-Papers) - Statistics and Accepted paper list with arXiv link of EMNLP-IJCNLP 2019
* NAACL North American Chapter of the Association for Computational Linguistics
  * [NAACL-HLT 2019](https://naacl2019.org/)
* COLING

Application Scenario

* WWW The Web Conference
  * [WWW 2020](https://www2020.thewebconf.org/)

General Models

* NPIS Neural Information Processing Systems
  * [NIPS Proceedingsβ](https://papers.nips.cc/)

Not Sure

* CCL
* AAAI
* ICLR
  * [AminJun/ICLR2020](https://github.com/AminJun/ICLR2020) - ICLR2020 Downloader & Search Tool

### Summaries

* [StateOfTheArt.ai](https://www.stateoftheart.ai/)
* [**Didi Chinese NLP**](https://chinesenlp.xyz/#/)
  * [github](https://github.com/didi/ChineseNLP)
* [**Papers With Code: Browse state-of-the-art**](https://paperswithcode.com/state-of-the-art)

NLP

* [graykode/nlp-roadmap: ROADMAP(Mind Map) and KEYWORD for students those who have interest in learning NLP](https://github.com/graykode/nlp-roadmap)
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

## Pending Project

Subject|Technique|Framework|Complexity|Remark
-------|---------|---------|----------|------
[Online ImageNet Classifier](Project/ImgClassifierAPI)|CNN|Keras|○○●●●|(TODO) Using Keras Applications combine with RESTful API
[First TF.js](Project/TFjs)||||(TODO) Using TensorFlow.js to load pre-trained model and make prediction on the browser
[YOLO](Project/YOLO)|CNN|Tensorflow||(TODO) Real-time Object Detection
[Word Similarity](Project/WordSimilarity)||||(TODO) Word Similarity Based on Dictionary and Based on Corpus
