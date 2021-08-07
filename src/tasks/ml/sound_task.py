# %% Packages

from ml_classes.task import MLTask
from pipeline.vae_pipeline import VAEPipeline
from sklearn.model_selection import train_test_split

# %% Classes


class SoundVAETask(MLTask):
    """This task builds a variational autoencoder for sound"""

    name = "sound_vae"

    def __init__(self, config):
        super().__init__(config)
        self.pipeline = VAEPipeline(config)
        self.config = config

    def run(self):

        # Load the data
        x_train, x_test, y_train, y_test = self.create_train_test_target_features()

        # Train the pipeline
        self.pipeline.train(x_train)

        # Evaluate the pipeline
        self.pipeline.evaluate(x_test, y_test)

    def create_train_test_target_features(self):

        train_size = self.parameters.get("train_size")
        random_state = self.parameters.get("random_state")

        x, y = self._load_data()  # TODO: Adjust that statement
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_size, random_state=random_state, shuffle=True
        )
        return x_train, x_test, y_train, y_test
