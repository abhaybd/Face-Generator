from keras.models import Model
from keras.layers import Dense, Reshape, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
import numpy as np
import tensorflow as tf
from datetime import datetime
import os

image_width, image_height = 28, 28
noise_shape = (100,)

def build_discriminator():
      image = Input(shape=(image_height, image_width, 1))
      x = Conv2D(128, kernel_size=(3,3))(image)
      x = LeakyReLU(alpha=0.2)(x)
      x = Conv2D(128, kernel_size=(3,3))(x)
      x = LeakyReLU(alpha=0.2)(x)
      x = MaxPooling2D()(x)
      x = Conv2D(256, kernel_size=(3,3))(x)
      x = LeakyReLU(alpha=0.2)(x)
      x = Conv2D(256, kernel_size=(3,3))(x)
      x = LeakyReLU(alpha=0.2)(x)
      x = MaxPooling2D()(x)
      x = GlobalAveragePooling2D()(x)
      x = Dense(1, activation='sigmoid')(x)
      discriminator = Model(inputs=image, outputs=x)
      return discriminator

def build_generator():
      noise = Input(shape=noise_shape)
      x = Dense(256, activation='relu', input_shape=noise_shape)(noise)
      x = BatchNormalization(momentum=0.8)(x)
      x = Dense(512, activation='relu')(x)
      x = BatchNormalization(momentum=0.8)(x)
      x = Dense(1024, activation='relu')(x)
      x = BatchNormalization(momentum=0.8)(x)
      x = Dense(image_width * image_height, activation='sigmoid')(x)
      x = Reshape((image_height, image_width, 1))(x)
      return Model(noise, x)

optimizer = Adam(0.00002)

discriminator = build_discriminator()
discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

generator = build_generator()
generator.compile(optimizer=optimizer, loss='binary_crossentropy')

discriminator.trainable = False

# Input is the noise
noise_input = Input(shape=noise_shape)
# The generator takes the noise and outputs an image
generated_image = generator(noise_input)
# The descriminator takes the image and outputs the validity (is it authentic or faked?)
validity = discriminator(generated_image)

combined = Model(noise_input, validity)
combined.compile(optimizer=optimizer, loss='binary_crossentropy')

x_train = np.load('dataset.npy')
x_train = x_train.astype(np.float32)/255

def write_log(callback, names, logs, epoch):
      for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, epoch)
        callback.writer.flush()

epochs = 1000
batch_size=32
half_batch = batch_size//2

date = datetime.today().strftime('%m-%d_%H%M')
callback = TensorBoard(os.path.join('logs',date))
callback.set_model(combined)
train_names = ['g_loss', 'd_loss', 'd_acc']

for epoch in range(epochs):
      losses = []
      for i in range(x_train.shape[0]//batch_size):
            # Train discriminator
            image_indexes = np.random.randint(0, x_train.shape[0], half_batch)
            images = x_train[image_indexes]
            noises = np.random.normal(0, 1, (half_batch, *noise_shape))
            
            generated_images = generator.predict(noises)
            
            discriminator_loss_real = discriminator.train_on_batch(images, np.ones((half_batch,1)))
            discriminator_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((half_batch,1)))
            discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
            
            # Train generator
            noises = np.random.normal(0, 1, (batch_size, *noise_shape))
            y = np.ones((batch_size,1))
            
            generator_loss = combined.train_on_batch(noises, y)
            
            discriminator_loss[1] *= 100
            losses.append((generator_loss, *discriminator_loss))
      avg_g_loss = np.average([loss[0] for loss in losses])
      avg_d_loss = np.average([loss[1] for loss in losses])
      avg_d_acc = np.average([loss[2] for loss in losses])
      write_log(callback, train_names, [avg_g_loss, avg_d_loss, avg_d_acc], epoch)
      print('\rEpoch: {: 5d} [G loss: {: 10.6f}] [D loss: {: 10.6f} acc.: {: 10.2f}%]'.format(epoch, avg_g_loss, avg_d_loss, avg_d_acc), end='')

generator.save_weights('generator.h5')
discriminator.save_weights('discriminator.h5')
combined.save_weights('combined.h5')