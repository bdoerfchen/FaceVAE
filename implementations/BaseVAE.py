import os
import keras
from keras.utils import CustomObjectScope
import tensorflow as tf
import numpy as np

from .descriptors.VAEDescriptor import VAEDescriptor

# https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/


class BaseVAE(keras.models.Model):
    """The base class for all VAE models

    They need to be initialized with an descriptor, describing their internal structure and thus determining the VAE variant
    """

    def __init__(self, descriptor: VAEDescriptor, name: str = "VAE") -> None:
        """Create a new VAE instance

        Args:
            descriptor (VAEDescriptor): An object describing the structure of the model, determines the kind of VAE
            name (str): The name of the model
        """

        encoder, decoder = descriptor.createModel()

        #   -------
        #   [ VAE ]
        vae_input = encoder.layers[0]
        vae_latent = encoder(vae_input)
        vae_latent_sampled = VAESamplingLayer(name="vae_latent_sampled")(vae_latent)
        vae_decoder_output = decoder(vae_latent_sampled)
        vae_mse_output = MSEReconstructionLossLayer(name="mse_vae_output")([vae_input, vae_decoder_output])
        vae = keras.models.Model()

        super.__init__(self, vae_input, vae_mse_output, name=name)
        pass

    def save(self, directory, file = "vae"):
        self.model.save(os.path.join(directory, file + ".keras"), save_format="keras")

    def load_from_directory(directory: str, file = "vae"):
        assert os.path.exists(directory)
        vae = BaseVAE()
        with CustomObjectScope({
                'KLDivergenceLossLayer': KLDivergenceLossLayer,
                'MSEReconstructionLossLayer': MSEReconstructionLossLayer,
                'VAESamplingLayer': VAESamplingLayer
            }):
            vae.model = keras.saving.load_model(os.path.join(directory, "vae.keras"), safe_mode=False)
        return vae




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

class MSEReconstructionLossLayer(keras.layers.Layer):
    
    """ Layer that calculates the mean squared error between input and output. 
    It expects [y_true, y_predict] and returns y_predict.
    """
    
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(MSEReconstructionLossLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        y_true, y_predict = inputs

        reconstruction_loss = keras.backend.square((y_true-y_predict)*255)
        reconstruction_loss = keras.backend.mean(reconstruction_loss, axis=[0, 1, 2, 3])
        self.add_loss(reconstruction_loss, inputs=inputs)
        return y_predict
        
        
class KLDivergenceLossLayer(keras.layers.Layer):

    """Identity transform layer that adds KL divergence
    to the final model loss.
    """
    KL_LOSS_FACTOR = 1

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLossLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mean, log_variance = inputs

        kl_batch = keras.backend.sum(-0.5 * (1 + log_variance - keras.backend.square(mean) - keras.backend.exp(log_variance)), axis=[1]) * 255
        self.add_loss(keras.backend.mean(kl_batch * KLDivergenceLossLayer.KL_LOSS_FACTOR, axis=-1), inputs=inputs)

        return inputs

