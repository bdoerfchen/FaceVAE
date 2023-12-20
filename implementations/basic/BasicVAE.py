###
#
# Basic implementation of a VAE
# adapted from fchollet: https://keras.io/examples/generative/vae/

from .VAE import VAE
from .Sampling import Sampling

import numpy as np
import tensorflow as tf
import keras
from keras import layers

class BasicVAE:

    def __init__(self, dataset, image_shape, latent_size=100, optimizer: keras.optimizers.Optimizer=keras.optimizers.Adam()) -> None:
        self.dataset = dataset
        self.image_shape = image_shape
        self.optimizer = optimizer
        self.latent_size = latent_size

        self.__compile()
        pass
    
    def __compile(self):

        encoder_inputs = keras.Input(shape=self.image_shape)
        x = layers.Conv2D(8, 3, activation="relu", strides=2, padding="same")(encoder_inputs) #output_shape=(128,128,8)
        x = layers.Conv2D(16, 5, activation="relu", strides=2, padding="same")(x) #output_shape=(64,64,16)
        x = layers.Flatten()(x)
        x = layers.Dense(2*self.latent_size, activation="relu")(x)
        z_mean = layers.Dense(self.latent_size, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_size, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # Define Decoder
        latent_inputs = keras.Input(shape=(self.latent_size,))
        x = layers.Dense(2*self.latent_size)(latent_inputs)
        x = layers.Dense(int(self.image_shape[0]/4 * self.image_shape[1]/4 * 16), activation="relu")(x)
        x = layers.Reshape((int(self.image_shape[0]/4), int(self.image_shape[1]/4), 16))(x)
        x = layers.Conv2DTranspose(16, 5, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(8, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        # Define VAE
        self.vae = VAE(encoder, decoder)
        self.vae.compile(optimizer=self.optimizer)




    def fit(self, epochs=30, batch_size=32):
        return self.vae.fit(self.dataset, epochs=epochs, batch_size=batch_size)

    def encode(self, image):
        return self.vae.encoder(image)

    def reconstruct(self, image):
        return self.vae(image)

    def generate(self, latent):
        self.vae.decoder(latent)

