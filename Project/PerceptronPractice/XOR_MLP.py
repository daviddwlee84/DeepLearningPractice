# ----------
# 1. create a network of perceptrons with the correct weights
# 2. define a procedure EvalNetwork() which takes in a list of inputs and
# outputs the value of this network.
# ----------

import numpy as np

class Perceptron:
    """
    This class models an artificial neuron with step activation function.
    """

    def __init__(self, weights = np.array([1]), threshold = 0):
        """ Initialize weights and threshold based on input arguments.
        
        Keyword Arguments:
            weights {np.array} -- weights (default: {np.array([1])})
            threshold {int} -- threshold (default: {0})
        """
        self.weights = weights
        self.threshold = threshold

    def activate(self, values):
        """Activation function
        
        Arguments:
            values {list} -- a list of numbers equal to length of weights
        
        Returns:
            int -- the output of a threshold perceptron with given inputs based on
                      perceptron weights and threshold
        """
        # First calculate the strength with which the perceptron fires
        strength = np.dot(values, self.weights)
        
        # Then return 0 or 1 depending on strength compared to threshold  
        return int(strength >= self.threshold)

def EvalNetwork(inputValues, Network):
    """Implement network

    Define a procedure to compute the output of the network, given inputs
    
    Arguments:
        inputValues {list} -- a list of input values e.g. [1, 0]
        Network {list} -- Network that specifies a perceptron network
    
    Returns:
        int -- the output of the Network for the given set of inputs
    """
    #Method1 :
    # p is an instance of Perceptron.
    # inner brackets --> input layer
    # Network[1][0] --> Perceptron([1, -2, 1],   1)  -- Only one element
    #return Network[1][0].activate([p.activate(inputValues) for p in Network[0]])
    
    #Method2 :
    OutputValue = inputValues
    for layer in Network:
        OutputValue = list(map(lambda p: p.activate(OutputValue), layer))
    return OutputValue[0] # single value list

def test(Network):
    """A few tests to make sure that the perceptron class performs as expected.
    
    Arguments:
        Network {list} -- Network list
    """
    print("0 XOR 0 = 0?:", EvalNetwork(np.array([0, 0]), Network))
    print("0 XOR 1 = 1?:", EvalNetwork(np.array([0, 1]), Network))
    print("1 XOR 0 = 1?:", EvalNetwork(np.array([1, 0]), Network))
    print("1 XOR 1 = 0?:", EvalNetwork(np.array([1, 1]), Network))

def main():
    # Set up the perceptron network
    Network = [
        # input layer, declare input layer perceptrons here
        [ Perceptron([1, 0], 1), Perceptron([1, 1], 2), Perceptron([0, 1], 1) ], \
        # output node, declare output layer perceptron here
        [ Perceptron([1, -2, 1], 1) ]
    ]

    test(Network)

if __name__ == "__main__":
    main()
