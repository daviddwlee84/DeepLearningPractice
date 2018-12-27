import numpy as np
import os
import requests
import io
import urllib
from PIL import Image
from resizeimage import resizeimage

DATASET_PATH = './data'

IMAGE_MIN_WIDTH  = 224 # VGG16
IMAGE_MIN_HEIGHT = 224

MINIMUM_FILE_SIZE = 3000

def load_image_as_array(file_destination):
    img = Image.open(file_destination)
    data = np.asarray(img, dtype="int32")
    return data

def download_single_image(url, destination):
    file_destination = os.path.join(destination, os.path.basename(url))
    # check if the file exist
    if not os.path.isfile(file_destination):
        print('Downloading', url)
        try:
            fd = urllib.request.urlopen(url, timeout=3)
            image_file = io.BytesIO(fd.read())
            image = Image.open(image_file)

            size = image.size
            if size[0] < IMAGE_MIN_WIDTH or size[1] < IMAGE_MIN_HEIGHT:  # Image too small
                return False

            resized = resizeimage.resize_cover(image, (IMAGE_MIN_WIDTH, IMAGE_MIN_HEIGHT))
            resized.save(file_destination, 'jpeg', icc_profile=resized.info.get('icc_profile'))
        # except (IOError, HTTPException, CertificateError) as e:
        except (IOError, UnicodeEncodeError, ValueError) as e:
            print(e)
            return False

        # Check if photo meets minimum size requirement
        size = os.path.getsize(file_destination)
        if size < MINIMUM_FILE_SIZE:
            os.remove(file_destination)
            print("Image Too Small:", url)
            return False

        # Try opening as array to see if there are any errors
        try:
            load_image_as_array(file_destination) # don't know why this won't work
            Image.open(file_destination)
        except (OSError, ValueError) as e:
            os.remove(file_destination)
            print("Invalid Image:", url)
            return False

        return True
    else:
        print('Already exist', url)
        return True

def download_images():
    URLSurls = {'Dog': 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02084071',
                'Hotdog': 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07697537',
                'DogFood': 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07805966'}

    for label, url in URLSurls.items():
        destination = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(destination):
            os.makedirs(destination)

        urls = requests.get(url).text
        successNum = 0
        for url in urls.split('\r\n'):
            successNum += download_single_image(url, destination)
        
        print('Downloaded', successNum, 'images of', label)

def main():
    download_images()

if __name__ == "__main__":
    main()
