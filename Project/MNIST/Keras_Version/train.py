import tensorflow as tf
import os
import app

checkpoint_path = "checkpoint/cp-{epoch:04d}.ckpt"
checkpoint_dir = "checkpoint/"
model_path = "model/mnist.h5"

# Load and prepare the MNIST dataset. 
# Convert the samples from integers to floating-point numbers
def load_dataset():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

# Build the `tf.keras` model by stacking layers.
# Select an optimizer and loss function used for training
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def main():
    # Create model structure
    model = create_model()

    # Load dataset
    x_train, y_train, x_test, y_test = load_dataset()

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, 
        save_weights_only=True,
        verbose=1,
        # Save weights, every 5-epochs.
        period=5)

    # Load checkpoint status
    checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
    print('checkpoint_state:\n', checkpoint_state)

    if checkpoint_state:
        print('model already exist, load weights...')
        app.restore_weight(model)
    else:
        print('can\'t find model, train a new one')
        # Train and evaluate model
        model.fit(x_train, y_train, epochs=10,
            validation_data=(x_test, y_test),
            callbacks=[cp_callback])
    
    model.summary()

    # NotImplementedError: Currently `save` requires model to be a graph network.
    # Consider using `save_weights`, in order to save the weights of the model.
    #model.save(model_path)

if __name__ == '__main__':
    main()