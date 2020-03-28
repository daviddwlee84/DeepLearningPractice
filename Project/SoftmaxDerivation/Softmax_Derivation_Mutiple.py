import numpy as np


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


class ReLU():
    def __call__(self, x):
        return np.maximum(x, 0.0)
    
    def gradient(self, x):
        x[x <= 0] = 0.0
        x[x > 0] = 1.0
        return x

class FCNNLayer:
    def __init__(self, node_num: int, last_layer_num: int, activation=None):
        self.__node_num = node_num  # output dimension
        self.__last_layer_num = last_layer_num  # input dimension
        self.W_ = None
        self.b_ = None

        self.activation = activation
        self.__layer_input = None

    def initialize(self, custom_W=None, custom_b=None):
        """
        Initial weights with random values. If given, then use custom weights.
        """
        if custom_W is not None:
            self.W_ = custom_W
        else:
            limit = 1 / np.sqrt(self.__last_layer_num)
            self.W_ = np.random.uniform(-limit, limit,
                                        (self.__last_layer_num, self.__node_num))

        if custom_b is not None:
            self.b_ = custom_b
        else:
            self.b_ = np.zeros((self.__node_num, 1))

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


def test_back_prop():
    # x = np.array([[3, 4, 5], [3, 4, 5]]).T
    # y = np.mat([[0, 1], [1, 0]]).T
    x = np.array([[3, 4, 5]]).T
    y = np.mat([[0, 1, 1]]).T

    layer = FCNNLayer(last_layer_num=3, node_num=2, activation=ReLU())
    init_W = np.array([[1, 1], [2, 2], [3, 3]])
    init_b = np.array([[0], [1]])
    layer.initialize(init_W, init_b)

    layer2 = FCNNLayer(last_layer_num=2, node_num=3, activation=None)
    init_W2 = np.array([[1, 1, 1], [2, 2, 2]])
    init_b2 = np.array([[0], [0], [1]])
    layer2.initialize(init_W2, init_b2)

    output = layer.forward_propagation(x)
    print(output)
    output2 = layer2.forward_propagation(output)
    print(output2)


    loss = CrossEntropy()(output2, y)
    print(loss)
    gradient = CrossEntropy().gradient(output2, y)
    print(gradient)
    accu_loss = layer2.back_propagation(gradient)
    print(accu_loss)
    accu_loss = layer.back_propagation(accu_loss)
    print(accu_loss)


def main():
    x = np.array([[8, 7], [7, 6]]).T
    y = np.array([[0, 1], [1, 0]]).T

    HiddenLayer = FCNNLayer(last_layer_num=2, node_num=3, activation=Sigmoid())
    OutputLayer = FCNNLayer(last_layer_num=3, node_num=2)
    loss_func = CrossEntropy()

    HiddenLayer.initialize()
    OutputLayer.initialize()

    i = 0
    learning_rate = 1e-2
    MAX_ITER = 10000
    tolerance = 0.000001
    while i < MAX_ITER:
        # Forward Propagation
        output1 = HiddenLayer.forward_propagation(x)
        theta = OutputLayer.forward_propagation(output1)
        y_hat = Softmax(theta)

        # Back Propagation
        loss = loss_func(y_hat, y)
        if i % 100 == 0:
            print("Round:", i, "\nCurrent loss:", loss)
        if loss < tolerance:
            # early stop
            break
        gradient = loss_func.gradient(y_hat, y)
        gradient = OutputLayer.back_propagation(gradient, learning_rate=learning_rate)
        HiddenLayer.back_propagation(gradient, learning_rate=learning_rate)
        i += 1

    print("======= Finish Training ======")

    print("After", i, "round training")
    print("Final eight:\n", HiddenLayer.W_,
          "\nFinal bias:\n", HiddenLayer.b_, "\nFinal loss:", loss)
    print("y_hat =\n", y_hat)


if __name__ == '__main__':
    main()
    # test_feed_forward()
    # test_back_prop()
