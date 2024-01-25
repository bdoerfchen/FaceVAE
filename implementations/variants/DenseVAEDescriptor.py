import keras
import numpy as np

from .VAEDescriptor import VAEDescriptor

class DenseVAEDescriptor(VAEDescriptor):
    
    def __init__(self, img_shape, encoder_layer_units : list, latent_size: int = 20) -> None:
        self.img_shape = img_shape
        self.encoder_layer_units = encoder_layer_units
        self.latent_size = latent_size

    def createModel(self) -> (keras.models.Model, keras.models.Model):
        img_shape = self.img_shape
        encoder_layer_units = self.encoder_layer_units
        latent_size = self.latent_size


        #   [ Encoder ]
        encoder_input = keras.layers.Input(shape=img_shape, name="encoder_input")
        encoder_flatten = keras.layers.Flatten()(encoder_input)

        x = keras.layers.Dense(encoder_layer_units[0], activation="relu")(encoder_flatten)
        for units in encoder_layer_units[1:]:
            x = keras.layers.Dense(units, activation="relu")(x)

        encoder_mean = keras.layers.Dense(units=latent_size, name="encoder_mean")(x)
        encoder_log_variance = keras.layers.Dense(units=latent_size, name="encoder_log_variance")(x)

        encoder = keras.models.Model(encoder_input, [encoder_mean, encoder_log_variance], name="encoder_model")
        encoder.summary()

        #   -----------
        #   [ Decoder ]
        decoder_input = keras.layers.Input(shape=(latent_size), name="decoder_input")

        x = keras.layers.Dense(encoder_layer_units[-1], activation="relu")(decoder_input)
        for units in list(reversed(encoder_layer_units))[1:]:
            x = keras.layers.Dense(units, activation="relu")(x)

        decoder_output = keras.layers.Dense(keras.backend.prod(img_shape))(x)
        decoder_reshape = keras.layers.Reshape(target_shape=img_shape)(decoder_output)

        decoder = keras.models.Model(decoder_input, decoder_reshape, name="decoder_model")
        decoder.summary()
        
        return (encoder, decoder)
