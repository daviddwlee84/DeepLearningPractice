# Semeion Handwritten Digit

Dataset

* Link: [Semeion Handwritten Digit Data Set](https://archive.ics.uci.edu/ml/datasets/semeion+handwritten+digit)
* Data
	* Image size: 16x16
	* Number of images: 1593

## Usage

```sh
# Training
python3 backward.py
# Testing
python3 test.py
# Predict actual image (bug)
python3 app.py path/to/images ...
```

### Class ImageDataSet

(with subclass ImageData)

> The purpose is to act like Tensorflow official mnist dataset object. I havn't see the source code of how Tensorflow design the class. I just imitate the functionality.

**Note**: Current only support one-hot label. (transfer to one-hot label before creating the instance/object)

```py
DataSetObject = ImageDataSet(data, label, test_set_ratio=0.3, random_seed=87)

# Get a batch of training data
images, labels = data.train.next_batch(BATCH_SIZE)
# Get total training number
training_number = DataSetObject.train.num_examples

# Get all the testing data and label
testData = DataSetObject.test.images
testLabel = DataSetObject.test.labels
```

## Result

Using 80% training data and 20% testing data in 1593 instances.

```txt
After 50000 training step(s), loss on training batch is 0.075765
After 50000 training step(s), test accuracy = 0.909091
```

There are strange things here. (Haven't found the answer yet)

1. Training loss did change but very small
2. Test accuracy barely change <- very weird
3. My prediction of app.py not even close the accuracy (90% vs less than 10%)

possible answer

* maybe is shuffle problem, but I did shuffle the dataset twice
* maybe is test set problem, but training loss is acting weird too. (far away from I playing around with MNIST)
* or maybe is the parameters problem, but I don't think it will affect that much...
* or maybe the training set is too small that the weights keep stay (overfit) in the final state...

## Links

### Others Dataset Class implementation

* [tentone/SemeionNet](https://github.com/tentone/SemeionNet)
    * [Dataset Class](https://github.com/tentone/SemeionNet/blob/master/source/dataset.py)
    * [Load dataset](https://github.com/tentone/SemeionNet/blob/master/source/semeion.py)

### Other Semeion code

* [SemeionNet-CNN (R)](https://github.com/ChiragSatbhaya/Semeion-Handwritten-Digit-Data-Set---CNN)
