# Chinese Word Segmentation and Named Entity Recognization

## Requirement

* Split the training data into ratio 7:3 (train:test) and Evaluate them with P/R/F1 metrics
* Use entire training data to train the model and Test with the test data and then Submit the prediciton as the same format as the training data

> data format is utf-16 little endian

### Corpus

This is a Traditional Chinese corpus.

### CWS Data

* Max sentence (sequence) length: 165 (training data max: 164)
* Total unique word (include PAD): 4744
* Training data size
  * examples (sentences): 66713 (70% training data)
  * examples (sentences): 95304 (100% training data)
  * words (max sentence length): 165
  * features (one-hot encode, i.e. total unique word): 4744
  * tags (cws tags): 4

### NER Data

* Max sentence (sequence) length: 374 (training data max > test data max)
* Total unique word (include PAD): 4379
* Training data size
  * examples (sentences): 25434 (70% training data)
  * examples (sentences): 36334 (100% training data)
  * words (max sentence length): 374
  * features (one-hot encode, i.e. total unique word): 4379
  * tags (ner tags): 7 (PER x 2 + LOC x 2 + ORG x 2 + N)

| Category | Tag          |
| -------- | ------------ |
| PER      | Person       |
| LOC      | Location     |
| ORG      | Organization |

The NER starts with `B-[Tag]`, if it is multiple words than will follow by `I-[Tag]`.

If the word is not NER than use the `N` tag.

### Report

* Method/Approach
* Experiment Settings and Steps
* The 30% test data evaluation result
* Question analysis and discussion

> Submission should be named as `Name-ID.seg` and `Name-ID.ner`

## Usage

Train and Predict

* `python3 cws_crf.py`
* `python3 ner_crf.py`

## Chinese Word Segmentation

### CWS Evaluation

> [previous notes](../MedicalCorpus/README.md#Evaluation)
>
> * [[原創]中文分詞器分詞效果的評測方法](https://www.codelast.com/%E5%8E%9F%E5%88%9B%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D%E5%99%A8%E5%88%86%E8%AF%8D%E6%95%88%E6%9E%9C%E7%9A%84%E8%AF%84%E6%B5%8B%E6%96%B9%E6%B3%95/)

## Named Entity Recognization

### NER Evaluation

> * [sklearn_crfsuite Evaluation](https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#evaluation)
> * [Named-Entity evaluation metrics based on entity-level](http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/)
>   * [example-full-named-entity-evaluation.ipynb](https://github.com/davidsbatista/NER-Evaluation/blob/master/example-full-named-entity-evaluation.ipynb)

* Performance per label type per token
  * [sklearn_crfsuite.metrics](https://sklearn-crfsuite.readthedocs.io/en/latest/_modules/sklearn_crfsuite/metrics.html)
* Performance over full named-entity
  * [davidsbatista/NER-Evaluation](https://github.com/davidsbatista/NER-Evaluation)

## Model

### CRF

### BiLSTM + CRF

![cws](https://cdn.nlark.com/yuque/0/2019/png/104214/1559369780794-69fd076c-8f3b-4895-ac7f-5533aff25df2.png)

![ner](https://github.com/scofield7419/sequence-labeling-BiLSTM-CRF/blob/master/snapshot/BiLSTM_CRF.png)

* [tf.contrib.layers.xavier_initializer](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer)
* [tf.nn.xw_plus_b](https://www.tensorflow.org/api_docs/python/tf/nn/xw_plus_b): Computes matmul(x, weights) + biases.

## Resources

> * [sklearn_crfsuite API](https://sklearn-crfsuite.readthedocs.io/en/latest/api.html)

### TensorFlow CRF

* [tensorflow/contrib/crf](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf)
* [TesofrFlow issue - CRF functions in TensorFlow 2.0 #26167](https://github.com/tensorflow/tensorflow/issues/26167)

### Example

* [macanv/BERT-BiLSTM-CRF-NER](https://github.com/macanv/BERT-BiLSTM-CRF-NER): BERT + BiLSTM + CRF
  * [lstm_crf_layer.py](https://github.com/macanv/BERT-BiLSTM-CRF-NER/blob/master/bert_base/train/lstm_crf_layer.py#L13)
* [scofield7419/sequence-labeling-BiLSTM-CRF](https://github.com/scofield7419/sequence-labeling-BiLSTM-CRF): BiLSTM + CRF
  * [BiLSTM_CRFs.py](https://github.com/scofield7419/sequence-labeling-BiLSTM-CRF/blob/master/engines/BiLSTM_CRFs.py#L18)
    * [blocks.py](https://github.com/nyu-mll/multiNLI/blob/master/python/util/blocks.py)

Not same task but similar model

* [nyu-mll/multiNLI](https://github.com/nyu-mll/multiNLI) - Baseline Models for MultiNLI Corpus
  * [bilstm.py](https://github.com/nyu-mll/multiNLI/blob/master/python/models/bilstm.py)

--

## Appendix

### Links

* [Python - read text file with weird utf-16 format](https://stackoverflow.com/questions/19328874/python-read-text-file-with-weird-utf-16-format)
* [How to split/partition a dataset into training and test datasets for, e.g., cross validation?](https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros)

Python Logging

* [logging — Logging facility for Python](https://docs.python.org/3/library/logging.html)
* [[Python] logging 教學](https://stackoverflow.max-everyday.com/2017/10/python-logging/)
* [python-logging](https://zwindr.blogspot.com/2016/08/python-logging.html)
* [[Python] logging 學習紀錄](https://emineminschang.blogspot.com/2016/08/logging-python-logging-module.html)

### TensorFlow notes

* [Graphs and Sessions](https://www.tensorflow.org/guide/graphs)
* [Save and Restore](https://www.tensorflow.org/guide/saved_model)
* global_step
  * [What does global_step mean in Tensorflow?](https://stackoverflow.com/questions/41166681/what-does-global-step-mean-in-tensorflow)
  * [tf.train.global_step](https://www.tensorflow.org/api_docs/python/tf/train/global_step)
  * [tf.train.get_or_create_global_step](https://www.tensorflow.org/api_docs/python/tf/train/get_or_create_global_step)
* [Variables](https://www.tensorflow.org/guide/variables)
  * Sharing Variable
    * [What happens when setting reuse=True in tensorflow](https://stackoverflow.com/questions/53547066/what-happens-when-setting-reuse-true-in-tensorflow)
      * reuse and variable scopes in general are deprecated and will be removed in tf2
      * instead recommend you use the tf.keras layers to build your model, which you can reuse by just reusing the objects
    * [tf.variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope)
      * [TensorFlow基礎：共享變量](https://www.jianshu.com/p/ab0d38725f88)
      * [共享變量](https://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/variable_scope.html)
      * [tensorflow學習筆記（二十六）：構建TF代碼](https://blog.csdn.net/u012436149/article/details/53843158)
  * Is Training
    * [Question of tensorflow : How could I turn is_training of batchnorm to False](https://forums.fast.ai/t/question-of-tensorflow-how-could-i-turn-is-training-of-batchnorm-to-false/5870)

### One-hot solutions

> * [Convert array of indices to 1-hot encoded numpy array](https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array)
> * [Smarter Ways to Encode Categorical Data for Machine Learning](https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159)
> * [How can I one hot encode in Python?](https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python)
> * [Python: One-hot encoding for huge data](https://stackoverflow.com/questions/41058780/python-one-hot-encoding-for-huge-data)
> * [Tutorial: (Robust) One Hot Encoding in Python](https://blog.cambridgespark.com/robust-one-hot-encoding-in-python-3e29bfcec77e)

#### One-time transform

> When I try to use them, it will swallow up more than 50G

* Numpy
  * [X] np.eye
    * `np.eye(num_features, dtype=np.uint8)[numpy_dataset]`
  * [X] np.eye + np.reshape
    * `np.squeeze(np.eye(num_features, dtype=np.uint8)[numpy_dataset.reshape(-1)]).reshape([num_examples, num_words, num_features])`
* [Scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html): currently don't support 3-dim matrix ([scipy issue - 3D sparse matrices #8868](https://github.com/scipy/scipy/issues/8868))
  * [ ] list + scipy.sparse.eye => np.array
    * `sparse.eye(num_features, dtype=np.uint8).tolil()[numpy_dataset.reshape(-1)].toarray().reshape((num_examples, num_words, num_features))` (don't work)
  * [ ] list + scipy.sparse.eye -> tf.sparse.SparseTensor: this need to modify the network structure (X)
* Keras
  * [X] [keras.utils.to_categorical](https://keras.io/utils/#to_categorical) ([tf.keras.utils.to_categorical](https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical))
    * `to_categorical(numpy_dataset, num_classes=num_features)`
* Pandas: don't seem will support 3-dim either
  * [ ] [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) ([How to do sparse one hot encoding with pandas?](https://www.kaggle.com/general/16675))
* TensorFlow: this will need to modify network structure which will limit the generalization
  * [ ] [tf.one_hot](https://www.tensorflow.org/api_docs/python/tf/one_hot)
* Scikit Learn
  * [ ] [sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html): this will need to input the "original word: encode" pair, which is not what I want
  * [X] sklearn.preprocessing.LabelBinarizer: can't transform 3-dim data
* [lazyarray](https://lazyarray.readthedocs.io/en/latest/tutorial.html)

#### Batch transform

#### Transform back from one-hot

> * [Coverting Back One Hot Encoded Results back to single Column in Python](https://stackoverflow.com/questions/45183213/coverting-back-one-hot-encoded-results-back-to-single-column-in-python)

* Numpy
  * np.argmax
