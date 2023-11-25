import numpy as np

# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
    

class Linear(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        # breakpoint()
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    
    def backward_propagation(self, upstream, learning_rate):
        downstream = np.dot(upstream, self.weights.T)
        weights_error = np.dot(self.input.T, upstream)

        # breakpoint()
        upstream = np.mean(upstream,axis=0)
        # breakpoint()
        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * upstream
        return downstream
    
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self,upstream, learning_rate):
        return self.activation_prime(self.input) * upstream
    
