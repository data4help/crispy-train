
# %% Packages

import numpy as np
import tensorflow as tf
from vae_model.model import VAE
from base_classes.ml_pipeline import PipelineModel
from visualization.functions import plot_evaluation

# %% Pipeline

class VAEPipeline(PipelineModel):

    def __init__(self, config, name):
        super().__init__(config, name)
        self.name = name
        self.config = config.get_config(self.name)
        self.vae = VAE(self.config)
        self.model = self.vae.model

    def train(self, x_train: np.array, y_train: np.array) -> None:
        batch_size = self.config.get_int("parameters.batch_size")
        epochs = self.config.get_int("parameters.number_of_epochs")
        patience = self.config.get_int("parameters.patience")

        es_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=patience)
        tb_callback = tf.keras.callbacks.TensorBoard("./logs", update_freq="batch")

        self.model.fit(
            x_train[:10_000], x_train[:10_000],
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            callbacks=[es_callback, tb_callback]
        )
        self.save(self.model)

    def evaluate(self, x_test: np.array, y_test: np.array):
        self.model.load_weights(self.weights_path())
        reconstructed_images, latent_representation = self.vae.reconstruct(x_test, self.model)
        plot_evaluation(self.config, x_test, y_test, reconstructed_images, latent_representation)

    def predict(self):
        pass
