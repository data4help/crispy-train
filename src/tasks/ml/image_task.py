# %% Packages

import os
import numpy as np
from PIL import Image
from typing import Tuple, List
from ml_classes.task import MLTask
from pipeline.vae_pipeline import VAEPipeline

# %% Classes


class ImageVAETask(MLTask):
    """This task builds a variational autoencoder for sound"""

    name = "image_vae"

    def __init__(self, config):
        super().__init__(config)
        self.pipeline = VAEPipeline(config, self.name)
        self.config = config

    def run(self):

        # Load the data
        image_arrays, labels = self.load_images()

        # Train the model
        self.pipeline.train(image_arrays, image_arrays, retrain=False)

        # Evaluate the model
        self.pipeline.evaluate(image_arrays, labels)

    def load_images(self) -> Tuple[np.array, List[str]]:
        """This method loads the images and adds a fourth dimension for the grayscale. Furthermore, the images are scaled to be between 0 and 1.

        :return: All images in an array format.
        :rtype: np.array
        """
        input_path = self.paths.get_string("input_path")
        image_list = [os.path.join(input_path, x) for x in os.listdir(input_path) if not x.startswith(".")]
        image_files = np.array([np.asarray(Image.open(file))/255 for file in image_list]).astype(np.float32)
        labels = ["Bottles"] * len(image_files)

        image_files = image_files[..., np.newaxis]
        return image_files, labels
