import numpy as np
from sklearn.model_selection import train_test_split

RAND_SEED = 87

def load_data(path='TibetanMNIST.npz'):
    data = np.load(path)
    image = data['image']
    label = data['label']

    X_train, y_train, X_test, y_test = train_test_split(image, label, test_size=0.3, random_state=RAND_SEED)
    return X_train, y_train, X_test, y_test
    

def Softmax(x):
    """
    Why mius np.max(x) is for the numerical stability, and it will be canceled out
    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference

class CrossEntropy():
    def __init__(self, use_for_loop:bool = False):
        self.use_for_loop = use_for_loop

    def __call__(self, y_hat, y):
        """
        Assume y is one-hot vector
        """
        # Avoid division by zero
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)

        probs = Softmax(y_hat)
        log_likelihood = 0
        if self.use_for_loop:
            for n in range(y.shape[0]):
                for k in range(y.shape[1]):
                    if y[n, k] == 1:
                        log_likelihood += -np.log(y_hat[n, k])
        else:
            y = np.argmax(y, axis=1)
            log_likelihood = -np.log(probs[range(y.shape[0]), y])

        loss = np.sum(log_likelihood) / y.shape[0]

        return loss

    def gradient(self, y_hat, y):
        return y_hat - y


class Sigmoid():
    def __call__(self, x):
        # To prevent from overflow encountered in exp
        x = np.clip(x, 1e-15, 1 - 1e-15)

        return 1.0 / (1.0 + np.exp(-x))
    
    def gradient(self, x):
        return self(x) * (1.0 - self(x))


class FCNNLayer:
    # TODO: make data to be "row"-based
    def __init__(self, node_num: int, last_layer_num: int, activation=None, custom_W=None, custom_b=None):
        self.__node_num = node_num  # output dimension
        self.__last_layer_num = last_layer_num  # input dimension

        if custom_W is not None:
            self.W_ = custom_W
            assert custom_W.shape == (self.__last_layer_num, self.__node_num)
        else:
            limit = 1 / np.sqrt(self.__last_layer_num)
            self.W_ = np.random.uniform(-limit, limit,
                                        (self.__last_layer_num, self.__node_num))
        if custom_b is not None:
            self.b_ = custom_b
            assert custom_b.shape == (self.__node_num, 1)
        else:
            self.b_ = np.zeros((self.__node_num, 1))

        self.activation = activation
        self.__layer_input = None

    def forward_propagation(self, last_layer):
        self.__layer_input = last_layer
        if self.activation:
            return self.activation(np.dot(self.W_.T, last_layer) + self.b_)
        else:
            return np.dot(self.W_.T, last_layer) + self.b_

    def back_propagation(self, gradient, learning_rate=0.0001):
        if self.activation:
            gradient = self.activation.gradient(gradient)
        
        W_temp = self.W_

        gradient_W = self.__layer_input.dot(gradient.T)
        gradient_b = gradient

        # Update parameters
        self.W_ = self.W_ - learning_rate * gradient_W
        assert self.W_.shape == gradient_W.shape
        self.b_ = self.b_ - learning_rate * gradient_b
        assert self.b_.shape == gradient_b.shape

        accumulated_gradient = W_temp.dot(gradient)
        return accumulated_gradient