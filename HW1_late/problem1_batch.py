import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from network import Network
from layer import Linear
from util import mse,mse_prime

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
    net = Network([Linear(1,1)])

    # train
    net.use(mse, mse_prime)
    errs = net.fit(X_train, Y_train, epochs=1000, learning_rate=1e-4,batch_size=1,errs=errs)

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
