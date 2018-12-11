# Fruits Classification

Motivation

* TFRecord
  * tf.train.Example
  * tf.python_io.TFRecordWriter
  * tf.data.TFRecordDataset
  * [tf.data.Iterator](https://www.tensorflow.org/api_docs/python/tf/data/Iterator)
* Multi-thread
* Try tf.layers

TODO

* [X] test (evaluation)
* figure out image processing part (the reshape part)
* understand more about iterator (is it need to be feedable?!)
* use another approach of multi-thread (start_queue_runners is deprecated)
* how epochs and steps work (and global_step and how it restore)
    * [What does global_step mean in Tensorflow?](https://stackoverflow.com/questions/41166681/what-does-global-step-mean-in-tensorflow)
* Try more complex model (since this is a 83 classes classification problem )
* Maybe there are other evaluation ways...

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

## Training

Compare training time with:

* BATCH_SIZE = 100
* NUM_EPOCHS = 5
* STEPS = 2000

### Single Thread

```sh
python3 model.py train 0
```

```txt
[u22711@c009 FRUIT]$ head SingleThread.o1392; tail SingleThread.o1392

########################################################################
#      Date:           Tue Dec 11 08:28:02 PST 2018
#    Job ID:           1392.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,walltime=06:00:00
########################################################################

train mode with single thread
After 1 training batch(s), loss on training batch is 124.949.
After 1961 training batch(s), loss on training batch is 0.179715.
After 1971 training batch(s), loss on training batch is 0.33047.
After 1981 training batch(s), loss on training batch is 0.498585.
After 1991 training batch(s), loss on training batch is 0.346279.

########################################################################
# End of output for job 1392.c009
# Date: Tue Dec 11 08:43:10 PST 2018
########################################################################
```

Total: 15 mins 8 sec

```sh
# Test
python3 model.py test 0
```

```txt
[u22711@c009 Fruits]$ cat testSingleThread.o1419

########################################################################
#      Date:           Tue Dec 11 08:50:50 PST 2018
#    Job ID:           1419.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,walltime=06:00:00
########################################################################

test mode with single thread
After 1991 training step(s), test accuracy = 0.03

########################################################################
# End of output for job 1419.c009
# Date: Tue Dec 11 08:51:02 PST 2018
########################################################################
```

### Multi-thread

```sh
# Training
python3 model.py train 1
```

```txt
[u22711@c009 FRUIT]$ head MultiThread.o1393; tail MultiThread.o1393

########################################################################
#      Date:           Tue Dec 11 08:28:25 PST 2018
#    Job ID:           1393.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,walltime=06:00:00
########################################################################

train mode with multi-thread
After 1 training batch(s), loss on training batch is 189.721.
After 1961 training batch(s), loss on training batch is 4.41871.
After 1971 training batch(s), loss on training batch is 4.41941.
After 1981 training batch(s), loss on training batch is 4.42031.
After 1991 training batch(s), loss on training batch is 4.42041.

########################################################################
# End of output for job 1393.c009
# Date: Tue Dec 11 08:42:39 PST 2018
########################################################################
```

Total: 14 mins 14 sec

```sh
# Test
python3 model.py test 1
```

```txt
[u22711@c009 Fruits-multi]$ cat testMultiThread.o1424

########################################################################
#      Date:           Tue Dec 11 08:58:32 PST 2018
#    Job ID:           1424.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,walltime=06:00:00
########################################################################

test mode with multi-thread
After 1991 training step(s), test accuracy = 0

########################################################################
# End of output for job 1424.c009
# Date: Tue Dec 11 08:58:47 PST 2018
########################################################################
```

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
