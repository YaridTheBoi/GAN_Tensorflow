import tensorflow as tf
from keras import layers

def make_generator_model(image_size, noise_size):
    print(image_size[0], image_size[1])
    model = tf.keras.Sequential()
    model.add(layers.Dense((image_size[0]//4)*(image_size[1]//4)*256, use_bias=False, input_shape=(noise_size,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((image_size[0]//4, image_size[1]//4, 256)))
    assert model.output_shape == (None, image_size[0]//4, image_size[1]//4, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, image_size[0]//4, image_size[1]//4, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, image_size[0]//2, image_size[1]//2, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, image_size[0], image_size[1], 3)

    return model


def make_discriminator_model(image_size):
    print(image_size[0], image_size[1])
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[image_size[0], image_size[1], 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

