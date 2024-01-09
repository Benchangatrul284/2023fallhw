import numpy as np

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
    def fit(self, x_train, y_train, epochs, learning_rate, batch_size,errs):
        # sample dimension first
        samples = len(x_train)

        # calculate the number of batches
        num_batches = samples // batch_size
        
        # training loop
        for i in range(epochs):
            idx = np.random.permutation(samples)
            x_train = x_train[idx]
            y_train = y_train[idx]
            # breakpoint()
            err = 0
            for j in range(num_batches):
                # get the batch
                x_batch = x_train[j*batch_size:(j+1)*batch_size].reshape(batch_size,-1)
                y_batch = y_train[j*batch_size:(j+1)*batch_size].reshape(batch_size,-1)
                # breakpoint()
                # forward propagation
                for layer in self.layers:
                    x_batch = layer.forward_propagation(x_batch)
                y_pred = x_batch

                # calculate error
                err += self.loss(y_batch, y_pred)
                
                # backpropagation
                error = self.loss_prime(y_batch, y_pred)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # print error for each epoch
            errs.append(err/num_batches)
            print('epoch %d/%d   error=%f' % (i+1, epochs, err/num_batches))

        return errs