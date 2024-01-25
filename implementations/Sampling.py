# from fchollet: https://keras.io/examples/generative/vae/

import tensorflow as tf
from keras import layers

class Sampling(layers.Layer):
    """Uses (mean, log_variance) to sample"""

    def call(self, inputs):
        mean, log_variance = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_variance) * epsilon
