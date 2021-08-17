# %% Packages

import tensorflow as tf
from tensorflow.keras import backend as K
from vae_model.decoder import CreateDecoder
from vae_model.encoder import CreateEncoder

# %% Code


class VAE:
    def __init__(self, config):
        self.config = config

        self.encoder = CreateEncoder(**self.config.get("network"))
        self.decoder = CreateDecoder(**self.config.get("network"))
        self.reconstruction_loss_weight = self.config.parameters.reconstruction_loss_weight
        self.model = self._build_and_compile_model()

    def _build_and_compile_model(self):
        model = self._build_model()
        model = self._compile_model(model)
        return model

    def _build_model(self):
        model_input = self.encoder._model_input
        model_output = self.decoder.model(self.encoder.model(model_input))
        model = tf.keras.Model(model_input, model_output, name="autoencoder")
        return model

    def _compile_model(self, model):
        learning_rate = self.config.get("parameters.learning_rate")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = self._calculate_combined_loss
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstructed_loss = K.mean(K.square(error), axis=[1, 2, 3])
        return reconstructed_loss

    def _calculated_KL_loss(self, y_target, y_predicted):
        log_variance = self.encoder.log_variance
        mu = self.encoder.mu
        kl_loss = -0.5 * K.sum(
            1 + log_variance - K.square(mu) - K.exp(log_variance), axis=1
        )
        return kl_loss

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculated_KL_loss(y_target, y_predicted)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_loss
