from keras.models import Model
from keras.layers import Dense, Reshape, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
import numpy as np
from tqdm import tqdm

image_width, image_height = 28, 28
noise_shape = (100,)

def build_discriminator():
      image = Input(shape=(image_height, image_width, 1))
      x = Conv2D(128, kernel_size=(3,3), activation='relu')(image)
      x = Conv2D(128, kernel_size=(3,3), activation='relu')(x)
      x = MaxPooling2D()(x)
      x = Conv2D(256, kernel_size=(3,3), activation='relu')(x)
      x = Conv2D(256, kernel_size=(3,3), activation='relu')(x)
      x = MaxPooling2D()(x)
      x = GlobalAveragePooling2D()(x)
      x = Dense(1, activation='sigmoid')(x)
      discriminator = Model(inputs=image, outputs=x)
      return discriminator

def build_generator():
      noise = Input(shape=noise_shape)
      x = Dense(256, activation='relu', input_shape=noise_shape)(noise)
      x = Dense(512, activation='relu')(x)
      x = Dense(1024, activation='relu')(x)
      x = Dense(image_width * image_height, activation='sigmoid')(x)
      x = Reshape((image_height, image_width, 1))(x)
      return Model(noise, x)

discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

generator = build_generator()
generator.compile(optimizer='adam', loss='binary_crossentropy')

discriminator.trainable = False

# Input is the noise
noise_input = Input(shape=noise_shape)
# The generator takes the noise and outputs an image
generated_image = generator(noise_input)
# The descriminator takes the image and outputs the validity (is it authentic or faked?)
validity = discriminator(generated_image)

combined = Model(noise_input, validity)
combined.compile(optimizer='adam', loss='binary_crossentropy')

x_train = np.load('dataset.npy')
x_train = x_train.astype(np.float32)/255

epochs = 30
batch_size=32
half_batch = batch_size//2

for epoch in range(epochs):
      # Train discriminator
      image_indexes = np.random.randint(0, x_train.shape[0], half_batch)
      images = x_train(image_indexes)
      noises = np.random.normal(0.5, 0.5, (half_batch, *noise_shape))
      
      generated_images = generator.predict(noises)
      
      descriminator_loss_real = discriminator.train_on_batch(images, np.ones((half_batch,1)))
      descriminator_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((half_batch,1)))
      descriminator_loss = 0.5 * np.add(descriminator_loss_real, descriminator_loss_fake)
      
      # Train generator
      noises = np.random.normal(0.5, 0.5, (batch_size, *noise_shape))
      y = np.ones((batch_size,1))
      
      generator_loss = combined.train_on_batch(noises, y)
      
      print('Epoch: {} [G loss: {: 10.2f}] [D loss: {: 10.2f} acc.: {: 10.2f}]%'.format(epoch, generator_loss, *descriminator_loss))

generator.save_weights('generator.h5')
discriminator.save_weights('discriminator.h5')
combined.save_weights('combined.h5')