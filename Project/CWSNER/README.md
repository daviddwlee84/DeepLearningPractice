# Chinese Word Segmentation and Named Entity Recognization

## Requirement

* Split the training data into ratio 7:3 (train:test) and Evaluate them with P/R/F1 metrics
* Use entire training data to train the model and Test with the test data and then Submit the prediciton as the same format as the training data

> data format is utf-16 little endian

### Corpus

This is a Traditional Chinese corpus.

### NER

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

## Chinese Word Segmentation

## Named Entity Recognization

## Resources

### TensorFlow CRF

* [tensorflow/contrib/crf](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf)
* [TesofrFlow issue - CRF functions in TensorFlow 2.0 #26167](https://github.com/tensorflow/tensorflow/issues/26167)

--

## Appendix

### Links

* [Python - read text file with weird utf-16 format](https://stackoverflow.com/questions/19328874/python-read-text-file-with-weird-utf-16-format)
* [How to split/partition a dataset into training and test datasets for, e.g., cross validation?](https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros)

### TensorFlow notes

* [Graphs and Sessions](https://www.tensorflow.org/guide/graphs)
* [Save and Restore](https://www.tensorflow.org/guide/saved_model)
* global_step
  * [What does global_step mean in Tensorflow?](https://stackoverflow.com/questions/41166681/what-does-global-step-mean-in-tensorflow)
  * [tf.train.global_step](https://www.tensorflow.org/api_docs/python/tf/train/global_step)
  * [tf.train.get_or_create_global_step](https://www.tensorflow.org/api_docs/python/tf/train/get_or_create_global_step)
