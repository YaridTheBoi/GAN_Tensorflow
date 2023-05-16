import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import os
import time
from keras import layers
import cv2
import numpy as np
from utils import make_generator_model, make_discriminator_model

checkpoint_dir = './training_checkpoints'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

data_dir = './data'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.listdir(data_dir):
    print("Data folder is empty. Provide some photos, then run trianer again")
    quit()


train_images = []

for file in os.scandir(data_dir):
    image = cv2.imread(file.path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(image)
    train_images.append(image)


train_images = np.array(train_images)

IMAGE_SIZE = 120
EPOCHS = 1000
noise_dim = 200

#BUFFER_SIZE = 60000
#BATCH_SIZE = 256

BUFFER_SIZE = 16
BATCH_SIZE = 2

#(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

print(train_images.shape)
train_images = train_images.reshape( -1, IMAGE_SIZE, IMAGE_SIZE, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]


train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)


        # Save the model every 15 epochs
        if (epoch + 1) % 200 == 0 or (epoch + 1) == epochs:
            checkpoint.save(file_prefix = checkpoint_prefix)


        

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


print("START")

generator = make_generator_model(IMAGE_SIZE, noise_dim)
discriminator = make_discriminator_model(IMAGE_SIZE)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)


train(train_dataset, EPOCHS)
