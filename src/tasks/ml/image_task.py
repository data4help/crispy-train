
# %% Packages

from ml_classes.task import MLTask
from tensorflow.keras.datasets import mnist
from pipeline.image_pipeline import VAEPipeline

# %%

class ImageVAETask(MLTask):
    """This task builds a variational autoencoder for images"""
    name = "image_vae"

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
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype("float32") / 255
        x_train = x_train.reshape(x_train.shape + (1,))

        x_test = x_test.astype("float32") / 255
        x_test = x_test.reshape(x_test.shape + (1,))
        return x_train, y_train, x_test, y_test
