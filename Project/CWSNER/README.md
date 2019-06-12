# Chinese Word Segmentation and Named Entity Recognization

## Requirement

* Split the training data into ratio 7:3 (train:test) and Evaluate them with P/R/F1 metrics
* Use entire training data to train the model and Test with the test data and then Submit the prediciton as the same format as the training data

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
