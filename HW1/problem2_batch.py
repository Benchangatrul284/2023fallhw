import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from util import sigmoid,sigmoid_prime,logistic,logistic_prime
from layer import Linear,ActivationLayer
from network import Network

def standardize(X):
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    return (X-mean)/std

def plot_loss_curve():
    plt.cla()
    plt.plot(errs, '-bx',label='train')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')
    # plt.legend()
    plt.savefig('loss_curve.png')

if __name__ == '__main__':
    # read datapoints from csv file
    errs = []
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

    # create network
    net = Network([Linear(2, 1),ActivationLayer(sigmoid, sigmoid_prime)])

    # train
    net.use(logistic, logistic_prime)
    errs = net.fit(X_train, Y_train, epochs=1000, learning_rate=0.75,batch_size = 1,errs = errs)

    # plot the prediction
    plt.scatter(X_test[:,0],X_test[:,1],c=Y_test)
    # calculate the loss on test_set
    out = net.predict(X_test)
    loss = logistic(Y_test,out)
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
    # plt.legend()
    plt.savefig('problem2.png')
    plot_loss_curve()