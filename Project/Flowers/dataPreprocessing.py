import glob
import os
import numpy as np
import tensorflow as tf

# Dataset folder location
INPUT_DATA = './flower_photos'
# Transfer data as numpy's format
OUTPUT_DIR = './data'

# Percentage of testing data and validation data
TEST_PERCENTAGE = 10
VALIDATION_PERCENTAGE = 10

# Make sure the image shape fit the model !!!!!!!!!!!!!!VERY IMPORTANT!!!!!!!!!!!!!
IMG_RESHAPE = [224, 224] # for resnet_v2
#IMG_RESHAPE = [299, 299] # for inception_v3
OUTPUT_FILE = 'flower_processed_data_224x224.npy' # for resnet_v2
#OUTPUT_FILE = 'flower_processed_data_299x299.npy' # for inception_v3

# Load dataset and split into training, testing and validation data
def create_image_lists(sess, testing_percentage, validation_percentage):

    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True

    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # Read sub-directory
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue
        print("processing:", dir_name)

        i = 0
        # Processing image data
        for file_name in file_list:
            i += 1
            # Read and reshape image
            image_raw_data = tf.gfile.GFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, IMG_RESHAPE)
            image_value = sess.run(image)

            # Distribute data randomly
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
            if i % 200 == 0:
                print(i, "images processed.")
        current_label += 1

    # Shuffle data for better training result
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    return np.asarray([training_images, training_labels,
                       validation_images, validation_labels,
                       testing_images, testing_labels])

def main():
    if not os.path.exists(OUTPUT_DIR):
        # if the folder doesn't exist then mkdir
        os.makedirs(OUTPUT_DIR)
    # Processing Data
    with tf.Session() as sess:
        processed_data = create_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        # Save data as numpy's format
        np.save(os.path.join(OUTPUT_DIR, OUTPUT_FILE), processed_data)

if __name__ == "__main__":
    main()
