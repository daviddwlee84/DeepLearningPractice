# CIFAR-10

## Dataset

[The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

* The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
* There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Extract Dataset

```sh
tar xzf cifar-10.tar.gz
```

## Generate Tags

10 classes

class         |encode
--------------|------
airplane      |0
automobile    |1
bird          |2
cat           |3
deer          |4
dog           |5
frog          |6
hourse        |7
ship          |8
truck         |9

In this directory execute

```sh
# Generate Tagfiles
python3 generate_tagfile.py
```

## FCNN Version

### Instruction

```sh
# 1. Generate TFRecord
python3 cifar_generate_dataset.py
# 2. Training
python3 cifar_backward.py
# 3. Test (Stop with Ctrl + c)
python3 cifar_test.py
```

## Link

[Kaggle - CIFAR-10 - Object Recognition in Images](https://www.kaggle.com/c/cifar-10)
