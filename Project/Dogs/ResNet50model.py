from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

import numpy as np
import pandas as pd

from data_downloader import DATASET_PATH, load_image_as_array

# ResNet50
IMAGE_SIZE = 224
IMAGE_CHANNEL = 3

NUM_CLASSES = 3 # Dog, Hotdog, DogFood

SEED = 87

BATCH_SIZE = 24
EPOCHS = 10

def buildModel(num_classes):
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    model.add(Dense(num_classes, activation='softmax'))

    # Say not to train first layer (ResNet) model. It is already trained
    model.layers[0].trainable = False

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def buildDataGenerator(dataset_path):
    # Create a generator with data augmentation
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                        validation_split=0.25,
                                        horizontal_flip=True,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2)

    # flow_from_dicrectory will auto label the image by the folder structure!
    train_gen = data_generator.flow_from_directory(dataset_path,
                                        subset="training",
                                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                        shuffle=True, seed=SEED,
                                        batch_size=BATCH_SIZE,
                                        class_mode='categorical')
    eval_gen = data_generator.flow_from_directory(dataset_path,
                                        subset="validation",
                                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                        shuffle=True, seed=SEED,
                                        batch_size=BATCH_SIZE,
                                        class_mode='categorical')
    global STEP_SIZE_TRAIN
    global STEP_SIZE_VALID
    # Total number of steps equal to the number of samples in your dataset divided by the batch size
    STEP_SIZE_TRAIN = train_gen.n / train_gen.batch_size
    STEP_SIZE_VALID = eval_gen.n / eval_gen.batch_size
    
    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_gen = test_generator.flow_from_directory(dataset_path,
                                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                        shuffle=True, seed=SEED,
                                        batch_size=1,
                                        class_mode=None)

    return train_gen, eval_gen, test_gen

def training(model, train_gen, eval_gen):
    model.fit_generator(train_gen,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=EPOCHS,
        validation_data=eval_gen,
        validation_steps=STEP_SIZE_VALID)

def evaluation(model, eval_gen):
    output = model.evaluate_generator(generator=eval_gen,
                             steps=STEP_SIZE_VALID, verbose=1)
    for name, value in zip(model.metrics_names, output):
        print(name, ':', value)

def predict_gen(model, test_gen, testNum=100, output_csv="resnet_results.csv"):
    # Predict
    pred = model.predict_generator(test_gen, steps=testNum, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)

    labels = (test_gen.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    
    filenames = test_gen.filenames[:testNum]
    results = pd.DataFrame({"Filename": filenames,
                            "Predictions": predictions})
    results.to_csv(output_csv, index=False)

def predict(model, image_path):
    img = load_image_as_array(image_path)
    reshaped_img = np.reshape(img, (1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL))
    pred = model.predict(reshaped_img)
    predicted_class_indices = np.argmax(pred, axis=1)
    return predicted_class_indices

def main():
    model = buildModel(NUM_CLASSES)
    train_gen, eval_gen, test_gen = buildDataGenerator(DATASET_PATH)
    training(model, train_gen, eval_gen)
    evaluation(model, eval_gen)
    predict_gen(model, test_gen, 100)
    labelList = list(train_gen.class_indices.keys())
    print(labelList[predict(model, './data/Dog/100_0921.JPG')])
    print(labelList[predict(model, './data/DogFood/3603_happydog_nk_kernig_1.jpg')])
    print(labelList[predict(model, './data/Hotdog/3NathansFamouse01.jpg')])

if __name__ == "__main__":
    main()
