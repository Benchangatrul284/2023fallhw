import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
    

class Network:
    def __init__(self,layers):
        self.layers = layers
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return np.array(result)

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in tqdm(range(epochs)):
            # apply SGD to one data randomly
            # choose a random sample
            sample = np.random.randint(samples)
            x = x_train[sample].reshape(1,-1)
            y = y_train[sample]
            # breakpoint()
            # forward propagation
            for layer in self.layers:
                x = layer.forward_propagation(x)
            y_pred = x
            err = self.loss(y, y_pred)
            error = self.loss_prime(y, y_pred)
            for layer in reversed(self.layers):
                error = layer.backward_propagation(error, learning_rate)
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))


def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def ReLU(x):
    return np.maximum(0,x)
def ReLU_prime(x):
    return np.where(x>0,1,0)

def standardize(X):
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    return (X-mean)/std


if __name__ == '__main__':
    # read datapoints from csv file
    hw_scores = pd.read_csv(os.path.join('hw1_dataset','Problem 2','Averaged homework scores.csv'))
    final_scores = pd.read_csv(os.path.join('hw1_dataset','Problem 2','Final exam scores.csv'))
    results = pd.read_csv(os.path.join('hw1_dataset','Problem 2','Results.csv'))
    X = standardize(np.array(hw_scores))
    Y = standardize(np.array(final_scores))
    Z = np.array(results)
    data = np.concatenate((X,Y,Z),axis=1)
    train_data = data[:400,:]
    test_data = data[400:,:]
    X_train = train_data[:,:2].reshape(-1, 2)
    Y_train = train_data[:,2].reshape(-1, 1)
    X_test = test_data[:,:2].reshape(-1, 2)
    Y_test = test_data[:,2].reshape(-1, 1)
    # breakpoint()
    # create network
    net = Network([Linear(2, 1),ActivationLayer(sigmoid, sigmoid_prime)])

    # train
    net.use(mse, mse_prime)
    net.fit(X_train, Y_train, epochs=100, learning_rate=0.75)

    # plot the prediction
    plt.scatter(X_test[:,0],X_test[:,1],c=Y_test)
    # calculate the loss on test_set
    out = net.predict(X_test)
    loss = mse(Y_test,out)
    print('loss on test set: ', loss)
    # plot the decision boundary
    x1 = np.linspace(-2,2,100)
    x2 = np.linspace(-2,2,100)
    x1,x2 = np.meshgrid(x1,x2)
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)
    X = np.concatenate((x1,x2),axis=1)
    y = net.predict(X).reshape(100,100)
    plt.contour(np.linspace(-2,2,100),np.linspace(-2,2,100),y,levels=[0.5])
    plt.xlabel('Homework score')
    plt.ylabel('Final score')
    plt.legend()
    plt.savefig('problem2.png')