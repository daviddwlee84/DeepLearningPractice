import numpy as np

def Softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class CrossEntropy():
    def __call__(self, y_hat, y):
        """Computes cross entropy between y (targets encoded as one-hot vectors) and predictions y_hat
        
        Arguments:
            y_hat {ndarray} -- (N, k)
            y {ndarray} -- (N, k)
        
        Returns:
            int -- cross entropy
        """
        # Avoid division by zero
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)

        #return - y.T * np.log(y_hat)
        return - y.T * np.log(y_hat) - (1 - y.T) * np.log(1 - y_hat)

    def gradient(self, y_hat, y):
        return y_hat - y

class FCNNLayer:
    def __init__(self, node_num, last_layer_num):
        self.__node_num = node_num
        self.__last_layer_num = last_layer_num
        self.W_ = None
        self.b_ = None

        self.__layer_input = None
    
    def initialize(self, custom_W=None, custom_b=None):
        if custom_W is not None:
            self.W_ = custom_W
        else:
            limit = 1 / np.sqrt(self.__last_layer_num)
            self.W_ = np.random.uniform(-limit, limit, (self.__last_layer_num, self.__node_num))
        
        if custom_b is not None:
            self.b_ = custom_b
        else:
            self.b_ = np.zeros((self.__node_num, 1))

    def forward_propagation(self, last_layer):
        self.__layer_input = last_layer
        return self.W_.T * last_layer + self.b_

    def back_propagation(self, gradient):
        W_temp = self.W_

        gradient_W = self.__layer_input.dot(gradient.T)
        gradient_b = gradient

        self.W_ = self.W_ - gradient_W
        self.b_ = self.b_ - gradient_b

        accumulated_gradient = gradient.T.dot(W_temp.T)
        return accumulated_gradient

def main():
    x = np.mat([8, 7]).T
    y = np.mat([0, 0, 1]).T

    init_W = np.full((2, 3), 0.5)
    init_b = np.full((3, 1), 1)

    HiddenLayer = FCNNLayer(last_layer_num=2, node_num=3)
    loss_func = CrossEntropy()
    
    HiddenLayer.initialize(custom_W=init_W, custom_b=init_b) # Custom Initialize
    #HiddenLayer.initialize() # Random Initialize
    
    i = 0
    MAX_ITER = 100
    tol = 0.000001
    while i < MAX_ITER:
        # Forward Propagation
        theta = HiddenLayer.forward_propagation(x)
        y_hat = Softmax(theta)

        # Back Propagation
        loss = loss_func(y_hat, y)
        print("Round:", i, "\nCurrent loss:", loss)
        print("Current weight:\n", HiddenLayer.W_, 
        "\nCurrent bias:\n", HiddenLayer.b_)
        if loss < tol:
            break
        gradient = loss_func.gradient(y_hat, y)
        HiddenLayer.back_propagation(gradient)
        i += 1

    print("======= Finish Training ======")

    print("After", i, "round training")
    print("Final eight:\n", HiddenLayer.W_, 
    "\nFinal bias:\n", HiddenLayer.b_, "\nFinal loss:", loss)
    print("y_hat =\n", y_hat)

if __name__ == '__main__':
    main()