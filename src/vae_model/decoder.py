# %% Packages

from ast import literal_eval as make_tuple
import numpy as np
from numpy.core.fromnumeric import shape
from tensorflow.keras import Model, layers

# %% Code


class CreateDecoder:
    def __init__(
        self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim
    ):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = [make_tuple(x) for x in conv_kernels]
        self.conv_strides = [make_tuple(x) for x in conv_strides]
        self.latent_space_dim = latent_space_dim

        self.model = None
        self._model_input = None
        self._shape_before_bottleneck = self._conv_arithmetic()
        self._num_conv_layers = len(conv_filters)
        self._build_decoder()

    def summary(self):
        self.model.summary()

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layers = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layers)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.model = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return layers.Input(shape=self.latent_space_dim, name="decoder_input")

    def _calculate_input_size(self, input_size, number_of_iteration, position):
        conv_strides_tuple = self.conv_strides[number_of_iteration]
        conv_strides_number = conv_strides_tuple[position]
        input_size = int(input_size / conv_strides_number)
        return input_size

    def _conv_arithmetic(self):
        input_width = self.input_shape[0]
        input_height = self.input_shape[1]

        for i in range(len(self.conv_filters)):
            input_width = self._calculate_input_size(input_width, i, 0)
            input_height = self._calculate_input_size(input_height, i, 1)

        shape_before_bottleneck = [input_width, input_height, self.conv_filters[-1]]
        return shape_before_bottleneck

    def _add_dense_layer(self, decoder_input):
        shape_before_bottleneck = self._conv_arithmetic()
        number_of_neurons = np.prod(shape_before_bottleneck)
        dense_layers = layers.Dense(number_of_neurons)(decoder_input)
        return dense_layers

    def _add_reshape_layer(self, dense_layer):
        return layers.Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """Add conv transpose blocks"""
        # Loop through all the conv layers in reverse order and stop at the first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = layers.Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}",
        )
        x = conv_transpose_layer(x)
        x = layers.ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = layers.BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = layers.Conv2DTranspose(
            filters=self.input_shape[-1],
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}",
        )
        x = conv_transpose_layer(x)
        output_layer = layers.Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer
