# Fruits Classification

Motivation

* TFRecord
  * tf.train.Example
  * tf.python_io.TFRecordWriter
  * tf.data.TFRecordDataset
  * [tf.data.Iterator](https://www.tensorflow.org/api_docs/python/tf/data/Iterator)
* Multi-thread
* Try tf.layers

## Dataset

* [Kaggle Fruits 360 dataset](https://www.kaggle.com/moltean/fruits)

Setup Kaggle API

```sh
# Install Kaggle API
pip3 install kaggle
# Setup API credentials
# (Goto Kaggle your account page, Create New API Token, and download it)
# Move it to ~/.kaggle/ and chmod
mkdir -p ~/.kaggle; mv ~/Downloads/kaggle.json ~/.kaggle/; chmod 600 ~/.kaggle/kaggle.json
```

```sh
# Download dataset
kaggle datasets download -d moltean/fruits
# Extract dataset
unzip -qq fruits.zip
```

## Generate TFRecord

```sh
# Clean up .DS_Store file
find fruits-360 -name .DS_Store | xargs rm

# Generate tfrecord
python3 tfrecord_manager.py
```

Here we store label as one-hot vector

> Note:
> Actually, using one-hot or integer as label are both work.
> That `tf.loss.sparse_softmax_cross_entropy` will squeeze int to one-hot
> So if process TFRecord to store label as one-hot label.
> You still need to transfer it back to int by using `tf.argmax`

## Trouble shooting

* TypeError: Tensor objects are only iterable when eager execution is enabled.

```py
# To use TFRecordDataset iterator
# To iterate over this tensor use tf.map_fn.
tf.enable_eager_execution()
```

* RuntimeError: tf.placeholder() is not compatible with eager execution.
    * [issue](https://github.com/tensorflow/tensorflow/issues/18165)

Placeholders don't quite make sense with eager execution since placeholders are meant to be "fed" in a call to Session.run, as part of the feed_dict argument. Since eager execution means immediate (and not deferred execution), there is no notion of a session or a placeholder.

The overall problem is because, my iterator only return one element (but I expected two)

```py
# It's a little bit like this I think
a, b = range(2)
```

**But actually the main problem is when I use TFRecordDataset**,

* [issue](https://github.com/tensorflow/tensorflow/issues/13508)

```py
dataset = tf.data.TFRecordDataset(filename)

# Wrong
dataset.map(decode)
# Correct
dataset = dataset.map(decode)
```

so it didn't decode things before I use it.
