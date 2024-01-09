from tensorflow.keras.layers import Conv2D, Flatten, Dropout, Input
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Input, Dense, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.models import Sequential

from tensorflow.keras.initializers import RandomNormal

def make_discriminator_model(figsize):
    model = Sequential()
    layers = 16
    # Initial Conv2D layer for downsampling
    model.add(Conv2D(layers, (3, 3), strides=(2, 2), padding='same', input_shape=[figsize, figsize, 3]))
    model.add(Dropout(0.2))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    # Gradual downsampling
    downsampling_layers = [
        (layers * 2, (3, 3)),
        (layers * 4, (3, 3)),
        (layers * 4, (3, 3)),
        (layers * 4, (3, 3)),
        # Add more layers or adjust strides for larger input sizes
    ]
    current_size = figsize // 2  # Adjust based on the initial Conv2D layer and strides above
    for filters, kernel_size in downsampling_layers:
        if current_size >= 4:  # We want to stop when the feature maps are 4x4
            model.add(Conv2D(filters, kernel_size, strides=(2, 2), padding='same'))
            model.add(Dropout(0.2))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization())
            current_size //= 2

    # Flatten the output and add a Dense layer
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('tanh'))
    return model



# Define the initializer
initializer = RandomNormal(mean=0., stddev=0.02)

def make_generator_model(latent_dim, figsize):
    model = Sequential()
    layers = 16
    # Initial Dense layer for a modest upscale
    model.add(Dense(layers * 4 * 4, input_dim=latent_dim, kernel_initializer=initializer))
    # model.add(Dropout(0.2))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, layers)))

    # Gradual upsampling and refinement
    model.add(Conv2DTranspose(layers * 2, (3, 3), strides=(2, 2), padding='same', kernel_initializer=initializer))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # Gradual upsampling and refinement
    model.add(Conv2DTranspose(layers * 4, (3, 3), strides=(2, 2), padding='same', kernel_initializer=initializer))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # Adjusting the number of upsampling layers based on the final image size
    # Adding more layers for larger figsize
    upsampling_layers = [
        (layers * 4, (3, 3)),
        (layers * 4, (3, 3)),
        (layers * 8, (3, 3)),
        (layers * 8, (3, 3)),
        (layers * 8, (3, 3)),
        (layers * 4, (3, 3)),
        (layers * 4, (3, 3)),
        (3, (3, 3)),
        # Add more layers or adjust strides for larger output sizes
    ]

    current_size = 16  # Adjust based on the initial dense layer and strides above
    for filters, kernel_size in upsampling_layers:
        if current_size < figsize:
            model.add(Conv2DTranspose(filters, kernel_size, strides=(2, 2), padding='same', kernel_initializer=initializer))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.2))
            current_size *= 2

    # Ensure the final layer has the correct size
    if current_size != figsize:
        final_stride = (figsize // current_size, figsize // current_size)
        model.add(Conv2DTranspose(64, (3, 3), strides=final_stride, padding='same', kernel_initializer=initializer))
    
    # general convolutional layers
    model.add(Conv2DTranspose(layers, (3, 3), padding='same', kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(layers, (3, 3), padding='same', kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # Output layer
    model.add(Conv2DTranspose(3, (3, 3), activation='tanh', padding='same', kernel_initializer=initializer))
    
    return model


if __name__ == '__main__':
    generator = make_generator_model(latent_dim=256, figsize=32)
    (generator.summary())
    discriminator = make_discriminator_model(figsize=32)
    (discriminator.summary())
