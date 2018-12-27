from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

from data_downloader import DATASET_PATH
from ResNet50model import buildDataGenerator, training, evaluation, predict_gen, predict

# VGG16
IMAGE_SIZE = 224

NUM_CLASSES = 3 # Dog, Hotdog, DogFood

SEED = 87

BATCH_SIZE = 24

def buildModel(num_classes):
    # create the base pre-trained model
    base_model = VGG16(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 3 classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def main():
    model = buildModel(NUM_CLASSES)
    train_gen, eval_gen, test_gen = buildDataGenerator(DATASET_PATH)
    training(model, train_gen, eval_gen)
    evaluation(model, eval_gen)
    predict_gen(model, test_gen, testNum=100, output_csv="vgg16_results.csv")
    labelList = list(train_gen.class_indices.keys())
    print(labelList[predict(model, './data/Dog/100_0921.JPG')])
    print(labelList[predict(model, './data/DogFood/3603_happydog_nk_kernig_1.jpg')])
    print(labelList[predict(model, './data/Hotdog/3NathansFamouse01.jpg')])

if __name__ == "__main__":
    main()
