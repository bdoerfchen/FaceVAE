###
#
# Basic implementation of a VAE
# adapted from fchollet: https://keras.io/examples/generative/vae/

import os
os.environ["KERAS_BACKEND"] = "tensorflow"


from VAE import VAE
from Sampling import Sampling


import numpy as np
import tensorflow as tf
import keras
from keras import layers
gpus = tf.config.list_physical_devices('GPU')

print("Start")
data_path = "D:\\Downloads\\ffhq256\\images256x256"
image_shape = (256, 256, 3)

# Load data
training_data

# Define Encoder
latent_dim = 100

encoder_inputs = keras.Input(shape=image_shape)
x = layers.Conv2D(8, 3, activation="relu", strides=2, padding="same")(encoder_inputs) #output_shape=(128,128,8)
x = layers.Conv2D(16, 5, activation="relu", strides=2, padding="same")(x) #output_shape=(64,64,16)
x = layers.Flatten()(x)
x = layers.Dense(2*latent_dim, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()


# Define Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(2*latent_dim)(latent_inputs)
x = layers.Dense(64 * 64 * 16, activation="relu")(x)
x = layers.Reshape((64, 64, 16))(x)
x = layers.Conv2DTranspose(16, 5, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(8, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# Define VAE
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())




# Train
vae.fit(ffhq256, epochs=30, batch_size=128)