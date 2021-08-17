# %% Packages

import os
import pickle
import matplotlib.pyplot as plt
from pyhocon import ConfigTree
from vae_model.model import VAE
from tensorflow.python.keras.engine.functional import Functional
from abc import abstractmethod

# %% Class


class MLTask:
    def __init__(self, config):
        self.parameters = config.get_config("parameters")
        self.paths = config.get_config("paths")

    def clear_output_path(self) -> None:
        """This method clears the output path in order to mitigate the case in which files from a previous run still remain and
        unintentionally alter the results
        """
        output_path = self.paths.get_string("output_path")
        file_names = [x for x in os.listdir(output_path) if not x.startswith(".")]
        for file in file_names:
            full_file_path = os.path.join(output_path, file)
            os.remove(full_file_path)

    def save_config(self):
        """This method saves the config file used to instantiate this class in order to re-call
        this class when used in the streamlit app.
        """
        config_path = os.path.join(self.paths.get_string("output_path"), "config.pickle")
        with open(config_path, "wb") as f:
            pickle.dump(self.config, f)

    def load_network_config(self) -> ConfigTree:
        """This method loads the config which was used for the VAE. Importing
        this is necessary in order to instantiates the model

        :return: Config file including all the parameters needed to instantiat the model
        :rtype: ConfigTree
        """
        config_path = os.path.join(self.paths.get_string("model_path"), "config.pkl")
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        return config

    def load_model(self) -> Functional:
        """This method instantiates the VAE and then loads the pretrained weights

        :return: Tensorflow model with pretrained weights
        :rtype: Functional
        """
        config = self.load_network_config()
        self.vae = VAE(config)
        model = self.vae.model
        weights_path = os.path.join(self.paths.get_string("model_path"), self.name, "weights.h5")
        model.load_weights(weights_path)
        return model

    @abstractmethod
    def run(self):
        pass
