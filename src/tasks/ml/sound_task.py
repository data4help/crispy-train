
# %% Packages

from ml_classes.task import MLTask
from pipeline.image_pipeline import VAEPipeline

# %%

class SoundVAETask(MLTask):
    """This task builds a variational autoencoder for sound"""
    name = "sound_vae"

    def __init__(self, config):
        self.pipeline = VAEPipeline(config, self.name)

    def run(self):

        # Load the data
        x_train, y_train, x_test, y_test = self._load_data()

        # Train the pipeline
        # self.pipeline.train(x_train, y_train)

        # Evaluate the pipeline
        self.pipeline.evaluate(x_test, y_test)

    def _load_data(self):
        a = 1
