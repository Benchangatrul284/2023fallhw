import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model import CVAE
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--latent_dim', type = int, default = 256)
parser.add_argument('--figsize', type = int, default = 28)
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--resume', type = str, default = '')
parser.add_argument('--save', type = str, default = './checkpoint')
parser.add_argument('--pics_dir', type = str, default = './pic_2')
parser.add_argument('--dataset_dir', type = str, default = '../dataset_flowers/rose')
parser.add_argument('--lr', type = float, default = 1e-4)
args = parser.parse_args()


def plot_losses(reconstruction_losses, kl_losses):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(reconstruction_losses)
    plt.title('Reconstruction Loss')

    plt.subplot(1, 2, 2)
    plt.plot(kl_losses)
    plt.title('KL Divergence')

    plt.tight_layout()
    plt.savefig('loss.png')

### The code that can plot like figure 1
def plot_latent_space(vae, n=5):
    # display an n*n 2D manifold of digits
    digit_size = args.figsize
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n, 3))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            rest_latent = np.ones((1,args.latent_dim-2))*yi
            z_sample = np.array([[xi, yi]])
            z_sample = np.concatenate((z_sample, rest_latent), axis=1)
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size, 3)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(args.figsize, args.figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure)
    # plt.show()
    plt.savefig('out.png')
    
    return


def log_normal_pdf(sample, mean, logvar, raxis=1):
    '''
    Compute the log pdf of the normal distribution.
    '''
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)


def compute_loss(model, x):
    '''
    Compute the loss of the model.
    return the total loss, reconstruction loss, KL loss
    '''
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_decoded = model.decode(z)
    cross_ent = tf.keras.losses.binary_crossentropy(x, x_decoded)
    logfx_z = -tf.reduce_sum(cross_ent)
    logpz = log_normal_pdf(z, 0., 0.)
    loggz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logfx_z + logpz - loggz_x), -tf.reduce_mean(logfx_z), -tf.reduce_mean(logpz - loggz_x)


@tf.function
def train_step(model, x, optimizer):
    '''
    Train the model for one step.
    '''
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def generate_and_save_images(model, epoch):
    '''
    Generate and save images after 10 epochs
    '''
    fig = plt.figure(figsize=(args.figsize,args.figsize))
    z_sample_test = tf.random.normal(shape=(16, args.latent_dim),seed=1)
    x_decoded_test = model.decoder.predict(z_sample_test)
    plt.imshow(x_decoded_test[0].reshape(args.figsize,args.figsize, 3))
    plt.savefig(os.path.join(args.pics_dir,'image_at_epoch_{:04d}.png'.format(epoch)))




def train():
    '''
    Train the model for args.epochs epochs.
    '''
    generate_and_save_images(model, 0) # Generate and save initial image
    for epoch in range(1, args.epochs + 1):
        reconstruction_loss = 0
        kl_loss = 0
        for train_x in train_dataset:
            train_step(model, train_x, optimizer)
            total_loss, r, k = compute_loss(model, train_x)
            reconstruction_loss += r
            kl_loss += k
        reconstruction_loss /= num_train_samples
        kl_loss /= num_train_samples
        
        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result() # calculate the ELBO(Evidence Lower Bound)
        
        print('Epoch: {}, Test set ELBO: {}'.format(epoch, elbo))
        reconstruction_losses.append(reconstruction_loss)
        kl_losses.append(kl_loss)
        plot_losses(reconstruction_losses, kl_losses)
        if epoch % 10 == 0:
            generate_and_save_images(model, epoch)
    

if __name__ == '__main__':
    reconstruction_losses = []
    kl_losses = []
    dataset = keras.utils.image_dataset_from_directory(
        args.dataset_dir, label_mode=None, seed=1, image_size=(args.figsize, args.figsize), batch_size=args.batch_size,
    )
    dataset = dataset.map(lambda x: x/255.0)

    # Calculate the number of batches in the dataset
    num_samples = tf.data.experimental.cardinality(dataset).numpy()
    # Calculate the number of training samples (80% of the total)
    num_train_samples = int(0.8 * num_samples)
    # Calculate the number of testing samples (the rest)
    num_test_samples = num_samples - num_train_samples
    # Split the dataset
    train_dataset = dataset.take(num_train_samples)
    test_dataset = dataset.skip(num_train_samples)
    
    optimizer = tf.keras.optimizers.Adam(args.lr)
    model = CVAE(args.latent_dim)
    
    train()
    plot_latent_space(model)