import numpy as np

class Perceptron:
    def __init__(self, shape, learning_rate=0.1, initWeight=None, initBias=None):
        if initWeight:
            self.weight = np.array(initWeight)
            if np.shape(self.weight) != (shape, ):
                raise ValueError('You input wrong shape of weight')
        else:
            # initial lize wieght with random number
            self.weight = np.random.uniform(-1, 1, size=shape)
        
        if initBias:
            self.bias = initBias
        else:
            self.bias = np.random.uniform(-1, 1)
        
        self.shape = shape # the input shape
        self.learning_rate = learning_rate

    def __repr__(self):
        status = f"""Perceptron with
        \rCurrent weight
        \r {self.weight}
        \rCurrent bias
        \r {self.bias}
        """
        return status
    
    def __stepFunction(self, x):
        return 1.0 if x > 0.0 else 0.0

    # Predict
    def forward(self, Xi):
        return self.__stepFunction(np.dot(Xi, self.weight) - self.bias)
    
    # Training
    # Stochastic Gradient Descent
    def backward(self, Xi, y):
        yHat = self.forward(Xi)
        delta = y - yHat # prediction error: expected - predicted
        # update weight and bias
        self.weight = self.weight + self.learning_rate * delta * Xi
        self.bias = self.bias - self.learning_rate * delta
    
    def batch_train(self, X, Y, visualize=False):
        for row, Xi in enumerate(X):
            self.backward(Xi, Y[row])
            if visualize:
                print('After round', row, 'training')
                print('updated weight', self.weight, 'bias', self.bias)
    
    def epoch_train(self, X, Y, epochs, visualize=False):
        for epoch in range(epochs):
            self.batch_train(X, Y, visualize=visualize)
            if visualize:
                print('After', epoch, 'epoch training')
                print('updated weight', self.weight, 'bias', self.bias)

def main():
    print("===== AND with init weight =====")
    ANDData = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 1]
    ])
    AND = Perceptron(2, learning_rate=0.1, initWeight=[0.2, -0.3], initBias=0.4)
    print(AND)
    AND.batch_train(ANDData[:, :2], ANDData[:, -1], visualize=True)
    print("\n\nAfter training")
    print(AND) # not sure why bias will contain very very very small decimal

    print("===== OR with random weight =====")
    ORData = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    OR = Perceptron(2)
    print(OR)
    OR.epoch_train(ORData[:, :2], ORData[:, -1], 100)
    print(OR)

    print('Predict')
    for row in ORData[:, :2]:
        print(row, '-->', OR.forward(row))
    

if __name__ == "__main__":
    main()
