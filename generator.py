import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
import os
from utils import make_discriminator_model, make_generator_model

IMAGE_SIZE = 120
noise_dim = 200

discriminator = make_discriminator_model(IMAGE_SIZE)
generator = make_generator_model(IMAGE_SIZE, noise_dim)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

noise = tf.random.normal([1, noise_dim])
generated_image = generator(noise, training=False)


plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()
