
# %% Packages

from tensorflow.keras import Model, layers
import tensorflow as tf
from tensorflow.keras import backend as K

# %% Settings

tf.compat.v1.disable_eager_execution()

# %% Code

class CreateEncoder:

    def __init__(
        self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim
    ):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.reconstruction_loss_weight = 1_000

        self.model = None
        self._model_input = None
        self._shape_before_bottleneck = None
        self._num_conv_layers = len(conv_filters)
        self._build_encoder()

    def summary(self):
        self.model.summary()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        self._model_input = encoder_input
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self.model = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return layers.Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Adds a conv block to a graph of layers, consisting of conv 2d + Relu + batch"""
        layer_number = layer_index + 1
        conv_layer = layers.Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = layers.ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = layers.BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _sample_normal_distribution(self, args):
        mu, log_variance = args
        epsilon = K.random_normal(shape=K.shape(mu))
        sampled_point = mu + K.exp(log_variance / 2) * epsilon
        return sampled_point

    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck with Gaussian sampling"""
        self._shape_before_bottleneck = x.shape[1:]
        x = layers.Flatten()(x)
        self.mu = layers.Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = layers.Dense(self.latent_space_dim, name="log_variance")(x)

        x = layers.Lambda(
            function=self._sample_normal_distribution,
            name="encoder_output")([self.mu, self.log_variance])
        return x
