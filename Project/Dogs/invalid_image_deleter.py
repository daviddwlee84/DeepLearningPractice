from data_downloader import load_image_as_array, DATASET_PATH
import os

classes = [folder for folder in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, folder))]

counter = 0

for folder in classes:
    print("Dealing with", folder)
    files = [f for f in os.listdir(os.path.join(DATASET_PATH, folder)) if os.path.isfile(os.path.join(DATASET_PATH, folder, f))]
    for img in files:
        image_path = os.path.join(DATASET_PATH, folder, img)
        try:
            load_image_as_array(image_path)
        except OSError:
            print("Found an invalid image", image_path)
            os.remove(image_path)
            counter += 1

print(counter, "invalid image deleted.")
