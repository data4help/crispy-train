# %% Packages

from abc import abstractmethod

# %% Class


class PipelineModel:
    def __init__(self, config):
        self.parameters = config.parameters
        self.paths = config.paths

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
