import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import argparse

# add parser
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=1000)
args = parser.parse_args()

# linear regression
class regression():
    def __init__(self):
        self.a = tf.Variable(initial_value=1.0, dtype=tf.float32)
        self.b = tf.Variable(initial_value=0.0, dtype=tf.float32)

    def __call__(self, X):
        X = tf.convert_to_tensor(X,dtype=tf.float32)
        y_est = tf.add(self.a,tf.multiply(self.b,X))
        return y_est
    
    def train(self, X_train, y_train, lr, batch_size, num_epochs):
        optimizer = tf.optimizers.SGD(learning_rate=lr)
        num_batches = len(X_train) // batch_size
        # use gpu if available
        with tf.device('/GPU:0') if args.device == 0 else tf.device('/CPU:0'):
            for epoch in tqdm(range(num_epochs)):
                for batch in range(num_batches):
                    start_idx = batch * batch_size
                    end_idx = (batch + 1) * batch_size
                    X_batch = X_train[start_idx:end_idx]
                    y_batch = y_train[start_idx:end_idx]
                    
                    with tf.GradientTape() as tape:
                        y_est = self(X_batch)
                        loss = tf.reduce_mean(tf.square(y_batch - y_est))
                        gradients = tape.gradient(loss, [self.a, self.b])
                        optimizer.apply_gradients(zip(gradients, [self.a, self.b]))

if __name__ == "__main__":
    # read datapoints from csv file
    hw_scores = pd.read_csv(os.path.join('hw1_dataset','Problem 1','Averaged homework scores.csv'))
    final_scores = pd.read_csv(os.path.join('hw1_dataset','Problem 1','Final exam scores.csv'))
    X = np.array(hw_scores.values)
    Y = np.array(final_scores.values)
    data = np.concatenate((X,Y),axis=1)
    train_data = data[:400,:]
    test_data = data[400:,:]
    X_train = train_data[:,0]
    Y_train = train_data[:,1]
    X_test = test_data[:,0]
    Y_test = test_data[:,1]
    
    model = regression()
    model.train(X_train, Y_train, args.lr, args.batch_size, args.num_epochs)
    # caclulate the loss on test_set
    y_est = model(X_test)
    loss = tf.reduce_mean(tf.square(Y_test - y_est))
    print('loss on test set: ', loss.numpy())
    # Plot the result
    plt.scatter(X_test, Y_test,color='blue',label='test')
    plt.plot(X_train, model(X_train), color='red',label= 'linear regression')
    plt.legend()
    plt.savefig('problem1.png')