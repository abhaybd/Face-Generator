# Based on implementation from here: https://github.com/cyjeffliu/Generate-Faces/blob/master/dlnd_face_generation.ipynb

from keras.models import Sequential, Model
from keras.layers import Conv2D, GaussianNoise, LeakyReLU, Dropout
from keras.layers import BatchNormalization, Flatten, Dense, Activation
from keras.layers import Conv2DTranspose, Input, Reshape
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import os
from datetime import datetime
import math
import numpy as np
from PIL import Image
from utils import rescale, write_log
from tqdm import tqdm

np.random.seed(42)

image_shape = (64,64,1)
noise_shape = (100,)

def build_discriminator():
      discriminator = Sequential()
      discriminator.add(GaussianNoise(0.01, input_shape=image_shape))
      discriminator.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))
      discriminator.add(LeakyReLU(alpha=0.01))
      discriminator.add(Dropout(0.5))
      
      discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
      discriminator.add(BatchNormalization())
      discriminator.add(LeakyReLU(alpha=0.01))
      discriminator.add(Dropout(0.5))
      
      discriminator.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
      discriminator.add(BatchNormalization())
      discriminator.add(LeakyReLU(alpha=0.01))
      discriminator.add(Dropout(0.5))
      
      discriminator.add(Conv2D(512, kernel_size=3, strides=2, padding='same'))
      discriminator.add(BatchNormalization())
      discriminator.add(LeakyReLU(alpha=0.01))
      discriminator.add(Dropout(0.5))
      
      discriminator.add(Conv2D(1024, kernel_size=3, strides=2, padding='same'))
      discriminator.add(BatchNormalization())
      discriminator.add(LeakyReLU(alpha=0.01))
      discriminator.add(Dropout(0.5))
      
      discriminator.add(Flatten())
      discriminator.add(Dense(1))
      discriminator.add(Activation('sigmoid'))
      return discriminator

def build_generator():
      generator = Sequential()
      generator.add(Dense(8*8*1024, input_shape=noise_shape))
      generator.add(Reshape((8,8,1024)))
      generator.add(BatchNormalization())
      generator.add(LeakyReLU(alpha=0.01))
      
      generator.add(Conv2DTranspose(512, kernel_size=3, strides=2, padding='same'))
      generator.add(BatchNormalization())
      generator.add(LeakyReLU(alpha=0.01))
      
      generator.add(Conv2DTranspose(256, kernel_size=3, strides=2, padding='same'))
      generator.add(BatchNormalization())
      generator.add(LeakyReLU(alpha=0.01))
      
      generator.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
      generator.add(BatchNormalization())
      generator.add(LeakyReLU(alpha=0.01))
      
      generator.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
      generator.add(BatchNormalization())
      generator.add(LeakyReLU(alpha=0.01))
      
      generator.add(Conv2DTranspose(1,kernel_size=3, strides=1, padding='same'))
      generator.add(Activation('tanh'))
      return generator

discriminator = build_discriminator()
generator = build_generator()

# Create optimizers
d_optimizer = Adam(lr=0.0005, beta_1=0.3)
g_optimizer = Adam(lr=0.0005, beta_1=0.3)

# Compile and setup discriminator
discriminator.compile(d_optimizer, loss='binary_crossentropy')
discriminator.trainable = False

noise_input = Input(shape=noise_shape)
generated_image = generator(noise_input)
validity = discriminator(generated_image)

# Build and compile a combined model, with the discriminator stacked on top of the generator
combined = Model(noise_input, validity)
combined.compile(optimizer=g_optimizer, loss='binary_crossentropy')

# Set number of epochs, batch size, and calculate half batch
epochs = 100
batch_size=32
half_batch = batch_size//2

if not os.path.isdir('logs/facev2'):
      os.makedirs('logs/facev2')

# Init TensorBoard callback
date = datetime.today().strftime('%m-%d_%H%M')
callback = TensorBoard(os.path.join('logs/facev2',date))
callback.set_model(combined)
train_names = ['g_loss', 'd_loss']

# Make checkpoint directory if not already there
if not os.path.isdir('checkpoints/facev2/%s' % date):
      os.makedirs('checkpoints/facev2/%s' % date)

# Make image directory
if not os.path.isdir('generated_images/facev2/%s' % date):
      os.makedirs('generated_images/facev2/%s' % date)

write_image_period = 1 # Write images every 20 epochs
num_images_to_write = 10 # Generate these many. MUST BE <= half_batch

# Initial loss is infinity so any subsequent loss will be better.
best_g_loss = math.inf

DATASET_PATH = 'resources/CelebA/processed-64'
all_images = os.listdir(DATASET_PATH)
num_images = len([img for img in all_images if img.endswith('.png')])

def get_batch(batch_size):
      image_indexes = np.random.randint(0, len(all_images), batch_size)
      images = np.array([np.expand_dims(np.array(Image.open(os.path.join(DATASET_PATH,all_images[i]))),axis=2) for i in image_indexes])
      return rescale(images.astype(np.float32), -1, 1, data_min=0, data_max=255)

for epoch in range(epochs):
      losses = []
      for _ in tqdm(range(num_images // batch_size)):
            # Train discriminator
            
            # Get random valid images
            images = get_batch(half_batch)
            
            # Generate random invalid images
            noises = np.random.uniform(-1, 1, size=(half_batch, *noise_shape))
            generated_images = generator.predict(noises)
            
            # Train on each half batch
            # Invalid images' ground truth is zeros
            discriminator_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((half_batch,1)))
            # Valid images' ground truth is ones
            discriminator_loss_real = discriminator.train_on_batch(images, np.ones((half_batch,1))*0.9)
            discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
            
            # Train generator
            
            # Create noise to generate images
            noises = np.random.uniform(-1, 1, size=(batch_size, *noise_shape))
            
            # Train using the combined model, since the error relies on the discriminator
            generator_loss = combined.train_on_batch(noises, np.ones((batch_size, 1)))
            
            # Add to list of losses
            losses.append((generator_loss, discriminator_loss))
      # Calculate various losses and accuracies
      avg_g_loss = np.average([loss[0] for loss in losses])
      avg_d_loss = np.average([loss[1] for loss in losses])
      
      noises = np.random.uniform(-1, 1, size=(half_batch, *noise_shape))
      generated_images = generator.predict(noises)
      
      if epoch % write_image_period == 0:
            for i,img in enumerate(generated_images[:num_images_to_write]):
                  image = rescale(img.squeeze(), 0, 255, data_min=-1, data_max=1)
                  image = Image.fromarray(np.round(image).astype(np.uint8))
                  image.save('generated_images/facev2/%s/epoch%03d_%d.png'%(date,epoch,i))
      
      try:
            generator.save('checkpoints/facev2/{}/g_epoch{:04d}.h5'.format(date, epoch))
            discriminator.save('checkpoints/facev2/{}/d_epoch{:04d}.h5'.format(date, epoch))
            best_g_loss = avg_g_loss
      except:
            pass
      # Write log to TensorBoard
      write_log(callback, train_names, [avg_g_loss, avg_d_loss], epoch)
      # Print to console
      print('Epoch: {: 5d} [G loss: {: 10.6f}] [D loss: {: 10.6f}]'.format(epoch, avg_g_loss, avg_d_loss))