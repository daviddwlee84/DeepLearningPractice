# Dogs

Classification of Dog, Hot Dog and Dog Food.

## ImageNet

### Brief Description

* 14,197,122 images, 21841 synsets indexed (2018/12/17)
* 334MB for only compressed image URLs

> *synset* is a short for *synonym set*.
>
> (information science) A set of one or more synonyms that are interchangeable in some context without changing the truth value of the proposition in which they are embedded.

### Data

Class|Description
-----|-----------
[Dog, domestic dog, Canis familiaris](http://image-net.org/synset?wnid=n02084071)|A member of the genus Canis (probably descended from the common wolf) that has been domesticated by man since prehistoric times; occurs in many breeds; "the dog barked all night"
[Hotdog, hot dog, red hot](http://image-net.org/synset?wnid=n07697537)|A frankfurter served hot on a bun
[Dog food](http://image-net.org/synset?wnid=n07805966)|Food prepared for dogs

The url of images may be invalid

* Size too small
* HTTP Error: 404 not found
* Timeout
* Invalid url character (not ascii)
* Invalid image (can't be opened as array)
* ValueError: Unknown url type

> we'll deal with them in `data_downloader.py` and `invalid_image_deleter.py`

## Dependencies

* [resizeimage](https://pypi.org/project/python-resize-image/) - `pip install python-resize-image`

## Usage

Download images from ImageNet

```sh
python3 data_downloader.py
```

Delete some invalid images (which can't open)

> not sure why it can't be deleted by the same operation logic in data_downloader.py

```sh
python3 invalid_image_deleter.py
```

Train ResNet50

```sh
python3 ResNet50model.py
```

Train VGG16

```sh
python3 VGG16model.py
```

## Result

> Training after 10 epoch

ResNet50

* Accuracy: 96.77%
* Loss: 0.0797

```txt
########################################################################
#      Date:           Thu Dec 27 06:07:09 PST 2018
#    Job ID:           9374.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,walltime=06:00:00
########################################################################

Found 1306 images belonging to 3 classes.
Found 434 images belonging to 3 classes.
Found 1740 images belonging to 3 classes.
Epoch 1/10
55/54 [==============================] - 65s 1s/step - loss: 0.3709 - acc: 0.8823 - val_loss: 0.1635 - val_acc: 0.9470
Epoch 2/10
55/54 [==============================] - 62s 1s/step - loss: 0.1748 - acc: 0.9527 - val_loss: 0.1433 - val_acc: 0.9447
Epoch 3/10
55/54 [==============================] - 61s 1s/step - loss: 0.1391 - acc: 0.9568 - val_loss: 0.1248 - val_acc: 0.9539
Epoch 4/10
55/54 [==============================] - 62s 1s/step - loss: 0.1209 - acc: 0.9644 - val_loss: 0.1030 - val_acc: 0.9677
Epoch 5/10
55/54 [==============================] - 62s 1s/step - loss: 0.1134 - acc: 0.9633 - val_loss: 0.0938 - val_acc: 0.9747
Epoch 6/10
55/54 [==============================] - 61s 1s/step - loss: 0.0966 - acc: 0.9667 - val_loss: 0.0993 - val_acc: 0.9724
Epoch 7/10
55/54 [==============================] - 60s 1s/step - loss: 0.0955 - acc: 0.9691 - val_loss: 0.0887 - val_acc: 0.9677
Epoch 8/10
55/54 [==============================] - 61s 1s/step - loss: 0.1035 - acc: 0.9702 - val_loss: 0.0991 - val_acc: 0.9700
Epoch 9/10
55/54 [==============================] - 61s 1s/step - loss: 0.0864 - acc: 0.9750 - val_loss: 0.0909 - val_acc: 0.9700
Epoch 10/10
55/54 [==============================] - 60s 1s/step - loss: 0.0809 - acc: 0.9750 - val_loss: 0.0948 - val_acc: 0.9770
19/18 [===============================] - 15s 788ms/step
loss : 0.07971909043941355
acc : 0.967741926694246
100/100 [==============================] - 22s 221ms/step

########################################################################
# End of output for job 9374.c009
# Date: Thu Dec 27 06:18:18 PST 2018
########################################################################
```

VGG16

* Accuracy: 93.55%
* Loss: 0.6773

```txt
########################################################################
#      Date:           Thu Dec 27 06:07:49 PST 2018
#    Job ID:           9375.c009
#      User:           u22711
# Resources:           neednodes=1:ppn=2,nodes=1:ppn=2,walltime=06:00:00
########################################################################

Found 1306 images belonging to 3 classes.
Found 434 images belonging to 3 classes.
Found 1740 images belonging to 3 classes.
Epoch 1/10
55/54 [==============================] - 48s 878ms/step - loss: 8.8332 - acc: 0.4459 - val_loss: 8.8761 - val_acc: 0.4493
Epoch 2/10
55/54 [==============================] - 44s 806ms/step - loss: 8.8651 - acc: 0.4500 - val_loss: 8.8761 - val_acc: 0.4493
Epoch 3/10
55/54 [==============================] - 45s 823ms/step - loss: 8.8147 - acc: 0.4531 - val_loss: 8.8761 - val_acc: 0.4493
Epoch 4/10
55/54 [==============================] - 45s 822ms/step - loss: 8.8651 - acc: 0.4500 - val_loss: 8.8761 - val_acc: 0.4493
Epoch 5/10
55/54 [==============================] - 45s 823ms/step - loss: 8.8987 - acc: 0.4479 - val_loss: 8.8761 - val_acc: 0.4493
Epoch 6/10
55/54 [==============================] - 45s 820ms/step - loss: 8.8819 - acc: 0.4489 - val_loss: 8.8761 - val_acc: 0.4493
Epoch 7/10
55/54 [==============================] - 45s 823ms/step - loss: 3.8641 - acc: 0.7441 - val_loss: 1.3152 - val_acc: 0.9124
Epoch 8/10
55/54 [==============================] - 46s 831ms/step - loss: 0.8339 - acc: 0.9285 - val_loss: 0.6211 - val_acc: 0.9470
Epoch 9/10
55/54 [==============================] - 46s 836ms/step - loss: 0.5697 - acc: 0.9509 - val_loss: 2.1983 - val_acc: 0.7765
Epoch 10/10
55/54 [==============================] - 46s 836ms/step - loss: 0.6674 - acc: 0.9447 - val_loss: 0.8443 - val_acc: 0.9263

...
19/18 [===============================] - 12s 624ms/step
loss : 0.6773356663387133
acc : 0.9354838665729294

...
100/100 [==============================] - 3s 30ms/step

########################################################################
# End of output for job 9375.c009
# Date: Thu Dec 27 06:15:50 PST 2018
########################################################################
```

## Notes

Download and show an image in python

```py
import urllib
import io
from PIL import Image
a = urllib.request.urlopen('http://static.flickr.com/2611/3680714896_bb5cbc89cb.jpg')
b = io.BytesIO(a.read()) # Seem like this step is redundant, not sure.
c = Image.open(b)
c.show()
```

Visualize data augmentation

```py
# First make a directory to store the processed data
# mkdir test

# Main
from keras.preprocessing.image import ImageDataGenerator
# Can add more data augmentation trick
gen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2)
iterator = gen.flow_from_directory('data', save_to_dir='./test')
next(iterator) # Test a batch (default 32)
```

## Links

Other ways to download image from ImageNet

* [How to get Images from ImageNet with Python in Google Colaboratory](https://medium.com/coinmonks/how-to-get-images-from-imagenet-with-python-in-google-colaboratory-aeef5c1c45e5)
  * [colaboratory notbook](https://colab.research.google.com/drive/1MALKxRqmNdjBUXJ-6V4PFYU6inPWq7Q)
* [xkumiyu/imagenet-downloader](https://github.com/xkumiyu/imagenet-downloader)
* [My First Attempt at ImageNet](http://theokanning.com/my-first-attempt-at-imagenet/)

Data preprocessing

* [Kaggle notebook - From image files to Numpy Arrays!](https://www.kaggle.com/lgmoneda/from-image-files-to-numpy-arrays)

Keras ImageDataGenerator

* [Keras Image Preprocessing](https://keras.io/preprocessing/image/)
* [Tutorial on using Keras flow_from_directory and generators](https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720)
* [Tutorial on Keras flow_from_dataframe](https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c)

Keras Model

* [VGG-16 pre-trained model for Keras](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)

Keras Transfer Learning

* [Keras Applications](https://keras.io/applications/)
* [Kaggle Transfer Learning](https://www.kaggle.com/dansbecker/transfer-learning)

Keras Data Augmentation

* [Kaggle Data Augmentation](https://www.kaggle.com/dansbecker/data-augmentation)
