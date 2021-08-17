
# %% Packages

import os
import numpy as np
from PIL.Image import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ml_classes.task import MLTask
from tqdm import tqdm

# %% Classes


class AugmentImages(MLTask):
    """This task resizes the images and augments them"""

    name = "augment_images"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.new_height = self.parameters.get_int("new_height")
        self.new_width = self.parameters.get_int("new_width")

    def run(self):

        # Prepare dirs
        self.clear_output_path()

        # Delete existing files and create necessary folders
        self.load_and_augment_images()

    def load_and_augment_images(self):
        """This method loads the image already in the target size and then augments the image through different zooming levels.
        Finally the image is made square by padding the remaining width with zeros.
        """
        zoom_lower_level = self.parameters.get("zoom_lower_level")
        zoom_upper_level = self.parameters.get("zoom_upper_level")
        pad_to_square = int((self.new_height - self.new_width) / 2)

        image_file_names = os.listdir(self.paths.input_path)
        valid_image_file_name = [x for x in image_file_names if x.endswith(".bmp")]
        datagen = ImageDataGenerator(zoom_range=[zoom_lower_level, zoom_upper_level])

        for file_name in tqdm(valid_image_file_name):
            image = self.load_image(file_name)
            it = datagen.flow(image, batch_size=1)

            for i in range(self.parameters.number_of_augmented_versions):
                batch = it.next()
                augmented_image = batch[0, :, :, :].astype(np.uint8)
                squared_image = np.pad(augmented_image, pad_width=((0, 0), (pad_to_square, pad_to_square), (0, 0)))
                im = array_to_img(squared_image)
                self.save_image(im, file_name, i)

    def load_image(self, file_name: str) -> np.array:
        """This method loads the image from the file path. The image is loaded in grayscale as well is resized to the target size
        specified in the config file. Finally, the image is given a batch dimension in the beginning in order to be processed
        by the tensorflow package

        :param file_name: File name of the image
        :type file_name: str
        :return: Loaded image, turned into a numpy array
        :rtype: np.array
        """
        full_input_path = os.path.join(self.paths.input_path, file_name)
        loaded_image = load_img(full_input_path, grayscale=True, target_size=(self.new_height, self.new_width))
        image = img_to_array(loaded_image)
        image = image[np.newaxis, ...]
        return image

    def save_image(self, im: Image, file_name: str, number: int):
        """This method takes the created image and then saves it under the specified file name. Since we are creating multiple versions
        of each image, these are specified through the number input

        :param im: Augmented image
        :type im: Image
        :param file_name: Name under which the image is going to be saved
        :type file_name: str
        :param number: Number which distinguishes between the different versions of the augmentation
        :type number: int
        """
        pure_file_name = file_name.split(".bmp")[0]
        adjusted_file_name = f"{pure_file_name}_v{number}.png"

        full_output_path = os.path.join(self.paths.output_path, adjusted_file_name)
        im.save(full_output_path)
