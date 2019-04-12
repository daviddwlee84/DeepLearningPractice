from tqdm import tqdm # For progress bar
import numpy as np
import copy # Shallow and deep copy operations
from collections import defaultdict # For metrics preservation

# ============== Dataset ============== #

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from mlxtend.data import iris_data

def batch_iterator(X, y=None, batch_size=32):
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]

def dataStandardlize(X):
    num_feature = np.shape(X)[1]
    for i in range(num_feature):
        X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
    return X

def loadData(standardlize=True):
    X, y = iris_data()

    y_one_hot = to_categorical(y, num_classes=3)

    if standardlize:
        X = dataStandardlize(X)

    train_X, test_X, train_y, test_y = train_test_split(X, y_one_hot, test_size=0.3, random_state=87)
    return train_X, test_X, train_y, test_y

# ============== Layers ============== #

# -------------- Dense Layer -------------- #

class DenseLayer:
    def __init__(self, n_units, input_dim=None):
        self.output_dim = n_units
        if input_dim:
            self.set_input_dim(input_dim)

        self.layer_input = None
        self.weight = None
        self.bias = None
    
    def set_input_dim(self, input_dim):
        self.input_dim = input_dim

    def forward_prop(self, X):
        self.layer_input = X # for backward_prop
        return np.dot(X, self.weight) + self.bias

    def backward_prop(self, accum_grad):
        # Save weights used during forwards pass
        old_weight = self.weight

        # Calculate gradient w.r.t (with reference to) layer weights
        grad_weight = self.layer_input.T.dot(accum_grad)
        grad_bias = np.sum(accum_grad, axis=0, keepdims=True)

        # Update the layer weights
        self.weight = self.weight_optimizer.update(self.weight, grad_weight)
        self.bias = self.bias_optimizer.update(self.bias, grad_bias)

        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        accum_grad = accum_grad.dot(old_weight.T)
        return accum_grad

    def initialize(self, optimizer):
        # Initialize the weights
        limit = 1 / np.sqrt(self.input_dim)
        self.weight  = np.random.uniform(-limit, limit, (self.input_dim, self.output_dim))
        self.bias = np.zeros((1, self.output_dim))
        # Weight optimizers
        self.weight_optimizer  = copy.copy(optimizer)
        self.bias_optimizer = copy.copy(optimizer)
 
# -------------- Activation Layer -------------- #

class ReLU():
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)

class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)

activation_functions_dict = {
    'relu': ReLU,
    'softmax': Softmax,
}

class ActivationLayer:
    def __init__(self, name, input_dim=None):
        self.layer_input = None
        self.activation_function = activation_functions_dict[name]()
        if input_dim:
            self.set_input_dim(input_dim)

    def set_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward_prop(self, X):
        self.layer_input = X
        return self.activation_function(X)

    def backward_prop(self, accum_grad):
        return accum_grad * self.activation_function.gradient(self.layer_input)

# ============== Model ============== #

# -------------- Optimizer -------------- #

class Adam():
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = None
        self.v = None
        # Decay rates
        self.b1 = b1
        self.b2 = b2

    # w.r.t = with reference to
    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.m is None:
            self.m = np.zeros(np.shape(grad_wrt_w))
            self.v = np.zeros(np.shape(grad_wrt_w))
        
        self.m = self.b1 * self.m + (1 - self.b1) * grad_wrt_w
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_wrt_w, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        self.w_updt = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

        return w - self.w_updt

class RMSprop():
    def __init__(self, learning_rate=0.01, rho=0.9):
        self.learning_rate = learning_rate
        self.Eg = None # Running average of the square gradients at w
        self.eps = 1e-8
        self.rho = rho

    # w.r.t = with reference to
    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.Eg is None:
            self.Eg = np.zeros(np.shape(grad_wrt_w))

        self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(grad_wrt_w, 2)

        # Divide the learning rate for a weight by a running average of the magnitudes of recent
        # gradients for that weight
        return w - self.learning_rate *  grad_wrt_w / np.sqrt(self.Eg + self.eps)

optimizers_dict = {
    'adam': Adam,
    'rmsprop': RMSprop,
}

# -------------- Loss Function -------------- #

class CrossEntropy:
    def __init__(self):
        pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)


loss_functions_dict = {
    'categorical_crossentropy': CrossEntropy,
}

# -------------- Evaluate Metrics -------------- #

class Accuracy:
    def __init__(self):
        self.name = 'acc'

    def score(self, y, p):
        y_true = np.argmax(y, axis=1)
        y_pred = np.argmax(p, axis=1)
        accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
        return accuracy

metrics_dict = {
    'accuracy': Accuracy,
}

# -------------- Nerual Network Model -------------- #

class MLP:
    def __init__(self, layers=None):
        self.layers = []
        if layers is not None:
            for layer in layers:
                self.addLayer(layer)
    
    def addLayer(self, layer):
        if self.layers:
            layer.set_input_dim(self.layers[-1].output_dim)
        
        self.layers.append(layer)

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=None):
        self.optimizer = optimizers_dict[optimizer]()
        self.loss_function = loss_functions_dict[loss]()
        self.metrics = []

        for metric in metrics:
            self.metrics.append(metrics_dict[metric]())

        for layer in self.layers:
            if hasattr(layer, 'initialize'): # initialize the weight of DenseLayer
                layer.initialize(optimizer=self.optimizer)

    def __forward_prop(self, X):
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_prop(layer_output)
        return layer_output
    
    def __backward_prop(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward_prop(loss_grad)

    def eval_on_batch(self, X, y):
        y_pred = self.__forward_prop(X)

        loss = np.mean(self.loss_function.loss(y, y_pred))
        result = [loss]

        for eval_metrics in self.metrics: # combined with other metrics if any
            result.append(eval_metrics.score(y, y_pred))
        
        return result

    def train_on_batch(self, X, y):
        y_pred = self.__forward_prop(X)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        # Calculate the gradient of the loss function wrt y_pred
        loss_grad = self.loss_function.gradient(y, y_pred)
        # Backpropagate. Update weights
        self.__backward_prop(loss_grad=loss_grad)

        result = [loss]
        for eval_metrics in self.metrics:
            result.append(eval_metrics.score(y, y_pred))

        return result

    def fit(self, X, y, epochs=10, batch_size=32, verbose=True):
        self.train_X, self.train_y = X, y # preserve for evaluate part
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch}/{epochs}")

            batch_error = []
            batch_metrics = defaultdict(list)
            if verbose:
                batch_count = 0
                data_size = len(y)
                pbar = tqdm(batch_iterator(X, y, batch_size=batch_size), total=int(np.ceil(data_size/batch_size)))
                pbar_message = ""
            else:
                pbar = batch_iterator(X, y, batch_size=batch_size)

            for X_batch, y_batch in pbar:
                result = self.train_on_batch(X_batch, y_batch)
                batch_error.append(result[0])
                pbar_message = "loss: " + "%.4f" % result[0]
                for i, metric in enumerate(self.metrics):
                    batch_metrics[metric.name].append(result[i+1])
                    pbar_message += " " + metric.name + ": " + "%.4f" % result[i+1]
                
                if verbose:
                    batch_count += len(y_batch)
                    pbar.set_description_str(f"{batch_count}/{data_size}")
                    pbar.set_postfix_str(pbar_message)

            if verbose:
                errors = np.mean(batch_error)
                print("- loss: %.4f" % errors, end="")

                for metric_name, metric_batch_value in batch_metrics.items():
                    metric_value = np.mean(metric_batch_value)
                    print("- %s: %.4f" % (metric_name, metric_value), end="")
                
                print() # as \n
        
    def evaluate(self, X, y, batch_size=32, verbose=True):
        batch_error = []
        batch_metrics = defaultdict(list)

        if verbose:
            batch_count = 0
            data_size = len(y)
            pbar = tqdm(batch_iterator(X, y, batch_size=batch_size), total=int(np.ceil(data_size/batch_size)))
        else:
            pbar = batch_iterator(X, y, batch_size=batch_size)

        for X_batch, y_batch in pbar:
            result = self.eval_on_batch(X_batch, y_batch)
            batch_error.append(result[0])
            for i, metric in enumerate(self.metrics):
                batch_metrics[metric.name].append(result[i+1])
            
            if verbose:
                batch_count += len(y_batch)
                pbar.set_description_str(f"{batch_count}/{data_size}")
            
        errors = np.mean(batch_error)
        result = [errors]

        for metric_batch_value in batch_metrics.values():
            metric_value = np.mean(metric_batch_value)
            result.append(metric_value)
        
        return result
    
    def predict(self, X):
        self.__forward_prop(X)

# My MLP

def MyMLP_adam(train_X, test_X, train_y, test_y, verbose=True):
    myModel = MLP()
    myModel.addLayer(DenseLayer(32, input_dim=4))
    myModel.addLayer(ActivationLayer('relu'))
    myModel.addLayer(DenseLayer(3))
    myModel.addLayer(ActivationLayer('softmax'))

    myModel.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    myModel.fit(train_X, train_y, epochs=10, batch_size=32, verbose=verbose)

    score = myModel.evaluate(test_X, test_y, batch_size=32, verbose=verbose) # loss, acc

    print("Accuracy of MLP From Scratch with Adam optimizer is", score[1])

def MyMLP_rmsprop(train_X, test_X, train_y, test_y, verbose=True):
    myModel = MLP([
        DenseLayer(32, input_dim=4),
        ActivationLayer('relu'),
        DenseLayer(3),
        ActivationLayer('softmax')
    ])

    myModel.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    myModel.fit(train_X, train_y, epochs=10, batch_size=32, verbose=verbose)

    score = myModel.evaluate(test_X, test_y, batch_size=32, verbose=verbose) # loss, acc

    print("Accuracy of MLP From Scratch with RMSprop optimizer is", score[1])


# Keras (for comparison purpose)

from keras.models import Sequential
from keras.layers import Dense, Activation

def keras_adam(train_X, test_X, train_y, test_y, verbose=True):
    model = Sequential()
    model.add(Dense(32, input_dim=4))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_X, train_y, epochs=10, batch_size=32, verbose=verbose)

    score = model.evaluate(test_X, test_y, batch_size=32, verbose=verbose) # loss, acc

    print("Accuracy of Keras with Adam optimizer is", score[1])

def keras_rmsprop(train_X, test_X, train_y, test_y, verbose=True):
    model = Sequential([
        Dense(32, input_dim=4),
        Activation('relu'),
        Dense(3),
        Activation('softmax')
    ])

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_X, train_y, epochs=10, batch_size=32, verbose=verbose)

    score = model.evaluate(test_X, test_y, batch_size=32, verbose=verbose) # loss, acc

    print("Accuracy of Keras with RMSprop optimizer is", score[1])

def main():
    train_X, test_X, train_y, test_y = loadData()

    # Keras test
    keras_adam(train_X, test_X, train_y, test_y, True)
    keras_rmsprop(train_X, test_X, train_y, test_y, True)

    MyMLP_adam(train_X, test_X, train_y, test_y, True)
    MyMLP_rmsprop(train_X, test_X, train_y, test_y, True)

if __name__ == "__main__":
    main()
