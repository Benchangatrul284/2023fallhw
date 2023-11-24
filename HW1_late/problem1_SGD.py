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

    # computes local gradient for a given upstream gradient. Returns downstream gradient.
    def backward_propagation(self, output_error, learning_rate):
        downstream = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return downstream
    
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output


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
        for i in (range(epochs)):
            # apply SGD
            err = 0
            for sample in range(samples):
                x = x_train[sample].reshape(1,-1)
                y = y_train[sample].reshape(1,-1)
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
            errs.append(err)

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def plot_loss_curve():
    plt.cla()
    plt.plot(errs, '-bx',label='train')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')
    plt.legend()
    plt.savefig('loss_curve.png')


if __name__ == '__main__':
    errs = []
    # read datapoints from csv file
    hw_scores = pd.read_csv(os.path.join('hw1_dataset','Problem 1','Averaged homework scores.csv'))
    final_scores = pd.read_csv(os.path.join('hw1_dataset','Problem 1','Final exam scores.csv'))
    X = np.array(hw_scores.values)
    Y = np.array(final_scores.values)
    data = np.concatenate((X,Y),axis=1)
    train_data = data[:400,:]
    test_data = data[400:,:]
    X_train = train_data[:,0].reshape(-1, 1)
    Y_train = train_data[:,1].reshape(-1, 1)
    X_test = test_data[:,0].reshape(-1, 1)
    Y_test = test_data[:,1].reshape(-1, 1)
    # breakpoint()
    # create network
    net = Network([Linear(1,2),Linear(2,1)])

    # train
    net.use(mse, mse_prime)
    net.fit(X_train, Y_train, epochs=1000, learning_rate=1e-5)

    # test
    out = net.predict(X_test)
    # calculate the loss on test_set
    loss = mse(Y_test,out)
    print('loss on test set: ', loss)
    plt.scatter(X_test, Y_test,color='blue',label='test')
    plt.plot(X_train, net.predict(X_train).reshape(-1,1), color='red',label= 'linear regression')
    plt.legend()
    plt.savefig('problem1.png')
    plot_loss_curve()