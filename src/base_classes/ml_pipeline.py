
# %% Packages

import os
from abc import abstractmethod

# %% Class

class PipelineModel:

    def __init__(self, config, name):
        self.save_path = config.get_string(f"{name}.paths.save_path")

    def weights_path(self):
        weights_path = os.path.join(self.save_path, "weights.h5")
        return weights_path

    def save(self, model):
        weight_path = os.path.join(self.save_path, "weights.h5")
        model.save_weights(weight_path)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self):
        pass

