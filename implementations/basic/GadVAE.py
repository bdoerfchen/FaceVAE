import os
import keras
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
        encoder_flatten = keras.layers.Flatten()(encoder_conv_2)

        encoder_mean = keras.layers.Dense(units=latent_size, name="encoder_mean")(encoder_flatten)
        encoder_log_variance = keras.layers.Dense(units=latent_size, name="encoder_log_variance")(encoder_flatten)

        latent_space = KLDivergenceLayer()([encoder_mean, encoder_log_variance])

        encoder = keras.models.Model(encoder_input, latent_space, name="encoder_model")
        encoder.summary()

        #   -----------
        #   [ Decoder ]
        decoder_input = keras.layers.Input(shape=(latent_size), name="decoder_input")
        decoder_latent_redense = keras.layers.Dense(units=np.prod(beforeflattened_shape), name="decoder_latent_redense")(decoder_input)
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
        vae_latent_sampled = keras.layers.Lambda(sampling, name="vae_latent_sampled")(vae_latent)
        vae_decoder_output = decoder(vae_latent_sampled)
        vae = keras.models.Model(vae_input, vae_decoder_output, name="VAE")
        vae.summary()
        vae.compile(optimizer=optimizer, loss=vae_loss_func(vae_latent))

        self.model = vae
        self.encoder = encoder
        self.decoder = decoder

    def fit(self, x, batch_size=64, epochs=20):
        self.model.fit(x, batch_size=batch_size, epochs=epochs)

    def save(self, directory):
        self.model.save(os.path.join(directory, "vae.keras"), save_format="keras")
        self.encoder.save(os.path.join(directory, "vae_encoder.keras"), save_format="keras")
        self.decoder.save(os.path.join(directory, "vae_decoder.keras"), save_format="keras")

    def load_from_directory(directory: str):
        assert os.path.exists(directory)
        gadVae = GadVAE()
        gadVae.model = keras.models.load_model(os.path.join(directory, "vae.keras"))
        gadVae.encoder = keras.models.load_model(os.path.join(directory, "vae_encoder.keras"))
        gadVae.decoder = keras.models.load_model(os.path.join(directory, "vae_decoder.keras"))
        return gadVae

    

def vae_loss_func(encoder_latent):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        diff = y_true-y_predict
        reconstruction_loss = keras.backend.square(diff)
        reconstruction_loss = keras.backend.mean(reconstruction_loss, axis=[1, 2, 3])
                
        l = reconstruction_loss_factor * reconstruction_loss
        return l

    def vae_kl_loss(y_true, y_predict):
        #print(encoder_latent)
        mean = encoder_latent[0]
        log_variance = encoder_latent[1]
        return encoder_latent[0] + encoder_latent[1]
        #kl_loss = -0.5 * keras.backend.sum(1.0 + log_variance - keras.backend.square(mean) - keras.backend.exp(log_variance), axis=1)
        #return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)
        loss = reconstruction_loss# + kl_loss

        return loss

    return vae_loss

def sampling(mean_log_variance):
    mean, log_variance = mean_log_variance
    epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean), mean=0.0, stddev=1.0)
    random_sample = mean + keras.backend.exp(log_variance/2) * epsilon
    return random_sample

        
class KLDivergenceLayer(keras.layers.Layer):

    """Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mean, log_variance = inputs

        kl_batch = -0.5 * keras.backend.sum(1 + log_variance - keras.backend.square(mean) - keras.backend.exp(log_variance), axis=-1)

        self.add_loss(keras.backend.mean(kl_batch), inputs=inputs)

        return inputs