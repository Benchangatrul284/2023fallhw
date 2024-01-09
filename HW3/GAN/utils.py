import matplotlib.pyplot as plt
import os
def plot_losses(gen_losses, disc_losses):
    plt.cla()
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gen_losses,label="Generator")
    plt.plot(disc_losses,label="Discriminator")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('losses.png')

def generate_and_save_images(model, epoch, test_input, pics_dir):
    # Notice `training` is set to False.
    plt.cla()
    plt.figure(figsize=(5,5))
    predictions = model(test_input, training=False)
    # imshow can accept images in [0, 1] range directly
    # if we need to show [0, 255] range, the number should be integer
    plt.imshow((predictions[0, :, :, :] / 2.0) + 0.5)
    plt.savefig(os.path.join(pics_dir, "image_at_epoch_{:04d}.jpg".format(epoch)))
    return