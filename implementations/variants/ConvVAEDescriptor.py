import keras
from .VAEDescriptor import VAEDescriptor

class ConvVAEDescriptor(VAEDescriptor):
    
    def __init__(self, img_shape, latent_size) -> None:
        self.img_shape = img_shape
        self.latent_size = latent_size

    def createModel(self) -> (keras.models.Model, keras.models.Model):
        img_shape = self.img_shape
        latent_size = self.latent_size


        #   [ Encoder ]
        encoder_input = keras.layers.Input(shape=img_shape, name="encoder_input")
            
        encoder_conv_1 = keras.layers.Conv2D(8, 3, activation="relu", strides=2, padding="same")(encoder_input)
        encoder_conv_2 = keras.layers.Conv2D(16, 5, activation="relu", strides=2, padding="same")(encoder_conv_1)

        beforeflattened_shape = keras.backend.int_shape(encoder_conv_2)[1:]
        conv_flatten = keras.layers.Flatten()(encoder_conv_2)

        prelatent_dense = keras.layers.Dense(4*latent_size, name="encoder_prelatent", activation="relu")(conv_flatten)

        encoder_mean = keras.layers.Dense(units=latent_size, name="encoder_mean")(prelatent_dense)
        encoder_log_variance = keras.layers.Dense(units=latent_size, name="encoder_log_variance")(prelatent_dense)

        latent_space = KLDivergenceLossLayer()([encoder_mean, encoder_log_variance]) # Does nothing but computes the kldivergence and adds loss

        encoder = keras.models.Model(encoder_input, latent_space, name="encoder_model")

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
        
        return (encoder, decoder)
