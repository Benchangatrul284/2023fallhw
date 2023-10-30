import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


def loss_function(y_true,y_est):
    return -y_true*tf.math.log(y_est)-(1-y_true)*tf.math.log(1-y_est)

# logistic regression
class logistic_regression():
    def __init__(self):
        self.w1 = tf.Variable(initial_value=1.0, dtype=tf.double)
        self.w2 = tf.Variable(initial_value=0.0, dtype=tf.double)
        self.b1 = tf.Variable(initial_value=1.0, dtype=tf.double)
        self.w3 = tf.Variable(initial_value=1.0, dtype=tf.double)
        self.w4 = tf.Variable(initial_value=0.0, dtype=tf.double)
        self.b2 = tf.Variable(initial_value=1.0, dtype=tf.double)
        self.activation_hidden = tf.nn.leaky_relu
        self.activation = tf.nn.sigmoid

    def __call__(self,X):
        X = tf.convert_to_tensor(X,dtype=tf.double)
        z1 = self.activation_hidden(self.w1*X[:,0]+self.w2*X[:,1]+self.b1)
        z2 = self.w3*z1+self.w4*z1+self.b2
        return self.activation(z2)
        
    
    def train(self, X_train, y_train, lr, batch_size, num_epochs):
        optimizer = tf.optimizers.SGD(learning_rate=lr)
        num_batches = len(X_train) // batch_size
        # use gpu if available
        with tf.device('/GPU:0'):
            for epoch in tqdm(range(num_epochs)):
                for batch in range(num_batches):
                    start_idx = batch * batch_size
                    end_idx = (batch + 1) * batch_size
                    X_batch = X_train[start_idx:end_idx]
                    y_batch = y_train[start_idx:end_idx]
                    
                    with tf.GradientTape() as tape:
                        y_est = self(X_batch)
                        loss = loss_function(y_batch,y_est)
                        loss = tf.reduce_mean(loss)
                        gradients = tape.gradient(loss, [self.w1, self.w2,self.b1])
                        optimizer.apply_gradients(zip(gradients, [self.w1, self.w2,self.b1]))


def standardize(X):
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    return (X-mean)/std


def train(lr,batch_size,num_epochs,plot=False):
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
    X_train = train_data[:,:2]
    Y_train = train_data[:,2]
    X_test = test_data[:,:2]
    Y_test = test_data[:,2]
    model = logistic_regression()
    model.train(X_train, Y_train, lr, batch_size, num_epochs)
    # calculate the loss on test_set
    y_est = model(X_test)
    loss = loss_function(Y_test,y_est)
    print('loss on test set: ', tf.reduce_mean(loss).numpy())
    if plot:
        # plot the prediction
        plt.scatter(X_test[:,0],X_test[:,1],c=Y_test)
        # plot the decision boundary
        x1 = np.linspace(-2,2,100)
        x2 = np.linspace(-2,2,100)
        x1,x2 = np.meshgrid(x1,x2)
        x1 = x1.reshape(-1,1)
        x2 = x2.reshape(-1,1)
        X = np.concatenate((x1,x2),axis=1)
        y = model(X).numpy().reshape(100,100)
        plt.contour(np.linspace(-2,2,100),np.linspace(-2,2,100),y,levels=[0.5])
        plt.xlabel('Homework score')
        plt.ylabel('Final score')
        plt.legend()
        plt.savefig('problem2.png')
    return tf.reduce_mean(loss).numpy()

if __name__ == "__main__":
    lr_list = np.linspace(1,10,100)
    loss_list = []
    for l in lr_list:
        loss_list.append(train(l,64,100))
    plt.plot(lr_list,loss_list)
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.savefig('problem2_loss-lr.png')
   