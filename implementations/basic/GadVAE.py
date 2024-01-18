import os
import keras
from keras.utils import CustomObjectScope
import tensorflow as tf
import numpy as np

# https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/


class GadVAE():

    def __init__(self, img_shape=(255, 255, 3), latent_size=100, optimizer : keras.optimizers.Optimizer = keras.optimizers.Adam()) -> None:
        self.createModel(img_shape=img_shape, latent_size=latent_size, optimizer=optimizer)
        pass

    def createModel(self, img_shape : list, latent_size : int, optimizer : keras.optimizers.Optimizer):
        #   [ Encoder ]
        encoder_input = keras.layers.Input(shape=img_shape, name="encoder_input")
            
        encoder_conv_1 = keras.layers.Conv2D(8, 3, activation="relu", strides=2, padding="same")(encoder_input)
        encoder_conv_2 = keras.layers.Conv2D(16, 5, activation="relu", strides=2, padding="same")(encoder_conv_1)

        beforeflattened_shape = keras.backend.int_shape(encoder_conv_2)[1:]
        conv_flatten = keras.layers.Flatten()(encoder_conv_2)

        prelatent_dense = keras.layers.Dense(4*latent_size, name="encoder_prelatent", activation="relu")(conv_flatten)

        encoder_mean = keras.layers.Dense(units=latent_size, name="encoder_mean")(prelatent_dense)
        encoder_log_variance = keras.layers.Dense(units=latent_size, name="encoder_log_variance")(prelatent_dense)

        latent_space = KLDivergenceLayer()([encoder_mean, encoder_log_variance]) # Does nothing but computes the kldivergence and adds loss

        encoder = keras.models.Model(encoder_input, latent_space, name="encoder_model")
        encoder.summary()

        #   -----------
        #   [ Decoder ]
        decoder_input = keras.layers.Input(shape=(latent_size), name="decoder_input")

        decoder_postlatent = keras.layers.Dense(4*latent_size, name="decoder_postlatent", activation="relu")(decoder_input)

        # Restore convoluted layers
        decoder_latent_redense = keras.layers.Dense(units=np.prod(beforeflattened_shape), name="decoder_latent_redense", activation="relu")(decoder_postlatent)
        decoder_reshape = keras.layers.Reshape(target_shape=beforeflattened_shape)(decoder_latent_redense)

        decoder_deconv_1 = keras.layers.Conv2DTranspose(16, 5, activation="relu", strides=2, padding="same")(decoder_reshape)
        decoder_deconv_2 = keras.layers.Conv2DTranspose(8, 3, activation="relu", strides=2, padding="same")(decoder_deconv_1)
        decoder_output = keras.layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(decoder_deconv_2)

        decoder = keras.models.Model(decoder_input, decoder_output, name="decoder_model")
        decoder.summary()

        #   -------
        #   [ VAE ]
        vae_input = encoder_input
        vae_latent = encoder(vae_input)
        vae_latent_sampled = VAESamplingLayer(name="vae_latent_sampled")(vae_latent) #keras.layers.Lambda(sampling, name="vae_latent_sampled")(vae_latent)
        vae_decoder_output = decoder(vae_latent_sampled)
        vae = keras.models.Model(vae_input, vae_decoder_output, name="VAE")
        vae.summary()
        vae.compile(optimizer=optimizer, loss=vae_reconstruction_loss)

        self.model = vae
        self.encoder = encoder
        self.decoder = decoder

    def fit(self, x, *args, **kwargs):
        self.model.fit(x, *args, **kwargs)

    def save(self, directory):
        self.model.save(os.path.join(directory, "vae.keras"), save_format="keras")
        self.encoder.save(os.path.join(directory, "vae_encoder.keras"), save_format="keras")
        self.decoder.save(os.path.join(directory, "vae_decoder.keras"), save_format="keras")

    def load_from_directory(directory: str):
        assert os.path.exists(directory)
        gadVae = GadVAE()
        with CustomObjectScope({
                'KLDivergenceLayer': KLDivergenceLayer,
                'VAESamplingLayer': VAESamplingLayer,
                'vae_reconstruction_loss': vae_reconstruction_loss
            }):
            gadVae.model = keras.saving.load_model(os.path.join(directory, "vae.keras"), safe_mode=False)
            gadVae.encoder = keras.saving.load_model(os.path.join(directory, "vae_encoder.keras"), safe_mode=False)
            gadVae.decoder = keras.saving.load_model(os.path.join(directory, "vae_decoder.keras"), safe_mode=False)
        return gadVae

    


def vae_reconstruction_loss(y_true, y_predict):
    reconstruction_loss_factor = 1000
    diff = y_true-y_predict
    reconstruction_loss = keras.backend.square(diff)
    reconstruction_loss = keras.backend.mean(reconstruction_loss, axis=[1, 2, 3])
                
    l = reconstruction_loss_factor * reconstruction_loss
    return l


class VAESamplingLayer(keras.layers.Layer):
    
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(VAESamplingLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mean, log_variance = inputs

        batch = keras.backend.shape(mean)[0]
        dim = keras.backend.shape(mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return mean + keras.backend.exp(0.5 * log_variance) * epsilon
        
        
class KLDivergenceLayer(keras.layers.Layer):

    """Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mean, log_variance = inputs

        kl_batch = keras.backend.sum(-0.5 * (1 + log_variance - keras.backend.square(mean) - keras.backend.exp(log_variance)), axis=-1)
        self.add_loss(keras.backend.mean(kl_batch), inputs=inputs)

        return inputs

