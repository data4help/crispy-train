# %% Packages

import os
import pickle
from abc import abstractmethod
import tensorflow as tf

# %% Class


class PipelineModel:
    def __init__(self, config):
        self.parameters = config.parameters
        self.paths = config.paths

    def _save(self, path, model, config):

        # Save model weights
        model_path = os.path.join(path, "sound_vae_weights.h5")
        model.save_weights(model_path)

        # Save config
        config_path = os.path.join(path, "config.pkl")
        with open(config_path, "wb") as f:
            pickle.dump(config, f)

    def _create_folder(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

    def _create_callbacks(self):

        # Early stopping
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=self.parameters.patience
        )

        # Tensorboard
        log_path = self.paths.get_string("logs_path")
        self._create_folder(log_path)
        tb_callback = tf.keras.callbacks.TensorBoard(log_path, update_freq="batch")
        return es_callback, tb_callback

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self):
        pass
