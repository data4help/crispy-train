# %% Packages

import os
import numpy as np
from tqdm import tqdm
from typing import Tuple, List
from ml_classes.task import MLTask
from pipeline.vae_pipeline import VAEPipeline
from sklearn.model_selection import train_test_split

# %% Classes


class SoundVAETask(MLTask):
    """This task builds a variational autoencoder for sound"""

    name = "sound_vae"

    def __init__(self, config):
        super().__init__(config)
        self.pipeline = VAEPipeline(config, self.name)
        self.config = config

    def run(self):

        # Load the data
        x_train, x_test, y_train, y_test = self.create_train_test_target_features()

        # Train the pipeline
        self.pipeline.train(x_train, x_train, retrain=False)

        # Evaluate the pipeline
        self.pipeline.evaluate(x_test, y_test)

    def load_sound_data(self) -> Tuple[np.array, List[str]]:
        """This method loads the spectograms and their genre information. Whereas the spectograms are saved
        in the numpy array, the label information are saved in a list

        :return: The first element are the spectograms saved in a numpy file, the second element is the list with genres
        :rtype: Tuple[np.array, List[str]]
        """
        input_path = self.paths.get_string("input_path")
        file_names = [x for x in os.listdir(input_path) if not x.startswith(".")]
        x, y = [], []
        for file_name in tqdm(file_names):
            full_file_path = os.path.join(input_path, file_name)
            file = np.load(full_file_path)
            x.append(file)
            y.append(file_name)
        x = np.array(x)
        x = x[..., np.newaxis]
        return x, y

    def create_train_test_target_features(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """This method splits the data into train and test data. This is done using the standard sklearn function
        with hyper-parameter from the config file

        :return: Train and test data for target and features
        :rtype: Tuple[np.array, np.array, np.array, np.array]
        """

        train_size = self.parameters.get("train_size")
        random_state = self.parameters.get("random_state")

        x, y = self.load_sound_data()
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_size, random_state=random_state, shuffle=True
        )
        return x_train, x_test, y_train, y_test
