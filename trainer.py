import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import os
import time
from keras import layers
import cv2
import numpy as np
from utils import make_generator_model, make_discriminator_model
from dotenv import load_dotenv

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


def train(dataset, epochs, checkpoints):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)


        # Save the model every 15 epochs
        if (epoch + 1) % checkpoints == 0 or (epoch + 1) == epochs:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


print("START")

if __name__ == '__main__':

    checkpoint_dir = './training_checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    data_dir = './data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.listdir(data_dir):
        print("Data folder is empty. Provide some photos, then run trianer again.\nREMEMBER TO USE SQUARE, BLACK-WHITE IMAGES (FOR NOW)")
        quit()


    train_images = []
    BUFFER_SIZE = 0
    for file in os.scandir(data_dir):
        image = cv2.imread(file.path)
        IMAGE_SIZE = image.shape[0]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        train_images.append(image)
        BUFFER_SIZE += 1


    train_images = np.array(train_images)
    ans = ''
    print('\n')
    if os.path.exists('.env'):
        load_dotenv()
        ans = input("""Do you want to use previous settings:
EPOCHS: {},
BATCH SIZE: {},
NOISE DIMENSION: {},
EPOCH CHECKPOINT: {}  
y/n: """.format(os.getenv("EPOCHS"), os.getenv("BATCH_SIZE") , os.getenv("NOISE_SIZE"), os.getenv("CHECKPOINT_EPOCH")))
        
        if(ans == 'y'):
            EPOCHS = int(os.getenv("EPOCHS"))
            BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
            noise_dim = int(os.getenv("NOISE_SIZE"))
            CHECKPOINT_EPOCH = int(os.getenv("CHECKPOINT_EPOCH"))

    if (ans != 'y'):
        print('\n')
        EPOCHS = int(input("How many epochs you want to train network?: "))
        BATCH_SIZE = int(input("How big should batch be? You have {} photos: ".format(BUFFER_SIZE)))
        noise_dim = int(input("What noise size you want to use?: "))
        CHECKPOINT_EPOCH = int(input("How often do you want to make a checkpoint(in EPOCHS)?: "))

        with open('.env', 'w') as config:
            config.write("NOISE_SIZE={}\n".format(noise_dim))
            config.write("IMAGE_SIZE={}\n".format(IMAGE_SIZE))
            config.write("EPOCHS={}\n".format(EPOCHS))
            config.write("CHECKPOINT_EPOCH={}\n".format(CHECKPOINT_EPOCH))
            config.write("BATCH_SIZE={}\n".format(BATCH_SIZE))
            config.write("BUFFER_SIZE={}\n".format(BUFFER_SIZE))
    
    print(train_images.shape)
    train_images = train_images.reshape( -1, IMAGE_SIZE, IMAGE_SIZE, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 


    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

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


    train(train_dataset, EPOCHS,CHECKPOINT_EPOCH )
