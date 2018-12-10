# Fruits Classification

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
