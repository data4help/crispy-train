# %% Packages

import numpy as np
import tensorflow as tf
from vae_model.model import VAE
from base_classes.ml_pipeline import PipelineModel
from visualization.functions import plot_evaluation

# %% Pipeline


class VAEPipeline(PipelineModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.save_path = config.paths.get_string("save_path")
        self.vae = VAE(self.config)
        self.model = self.vae.model

    def train(self, x_train: np.array) -> None:
        batch_size = self.parameters.get_int("batch_size")
        epochs = self.parameters.get_int("number_of_epochs")
        es_callback, tb_callback = self._create_callbacks()

        # self.model.fit(
        #     x_train,
        #     x_train,
        #     batch_size=batch_size,
        #     epochs=epochs,
        #     shuffle=True,
        #     callbacks=[es_callback, tb_callback],
        # )
        self._save(self.save_path, self.model, self.config)

    def evaluate(self, x_test: np.array, y_test: np.array) -> None:
        weights_path = f"{self.save_path}/sound_vae_weights.h5"
        self.model.load_weights(weights_path)

        reconstructed_images, latent_representation = self.vae.reconstruct(
            x_test, self.model
        )
        plot_evaluation(
            self.config, x_test, y_test, reconstructed_images, latent_representation
        )

    def predict(self):
        pass
