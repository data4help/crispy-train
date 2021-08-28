# %% Packages

import os
import numpy as np
from vae_model.model import VAE
from base_classes.ml_pipeline import PipelineModel
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tensorflow.keras import backend as K
from sklearn.decomposition import PCA
from typing import Tuple
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.engine.functional import Functional

# %% Pipeline


class VAEPipeline(PipelineModel):
    def __init__(self, config, name):
        super().__init__(config)

        self.config = config
        self.name = name
        self.vae = VAE(self.config)
        self.model = self.vae.model

    def train(self, x_train: np.array, y_train: np.array, retrain: bool) -> None:
        """This method trains the VAE. For that it first extracts the necessary model parameters such as batch size and number of
        epochs. Afterwards it instantiates the callbacks for the run. We are using the early stopping callback as well as the
        tensorboard functionalities. After the training is completed the model is saved.

        :param x_train: Train data
        :type x_train: np.array
        :param x_train: Target data. In most cases that will be identical to the train data, but not always
        :type x_train: np.array
        """

        weights_path = os.path.join(self.paths.get_string("model_path"), self.name, "weights.h5")
        if retrain:
            batch_size = self.parameters.get_int("batch_size")
            epochs = self.parameters.get_int("number_of_epochs")
            es_callback, tb_callback = self._create_callbacks()

            self.model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=True,
                callbacks=[es_callback, tb_callback],
            )
            self.model.save_weights(weights_path)
        else:
            self.model.load_weights(weights_path)

    def evaluate(self, data: np.array, labels: np.array) -> None:
        reconstructed_images, latent_representation = self.reconstruct(
            data, self.model
        )
        self.plot_evaluation(data, labels, reconstructed_images, latent_representation)

    def _create_folder(self, path: str) -> None:
        """This method creates a certain folder if that folder does not exist yet

        :param path: Folder path which should be created if it does not exist
        :type path: str
        """
        if not os.path.isdir(path):
            os.mkdir(path)

    def _create_callbacks(self) -> Tuple[EarlyStopping, TensorBoard]:
        """This method creates the early stopping callback as well as the callback for the creation of the
        tensorboard

        :return: The instances for the early stopping as well as tensorboard callback
        :rtype: Tuple[EarlyStopping, TensorBoard]
        """

        # Early stopping
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=self.parameters.patience
        )

        # Tensorboard
        log_path = os.path.join(self.paths.get_string("logs_path"), self.name)
        self._create_folder(log_path)
        tb_callback = tf.keras.callbacks.TensorBoard(log_path, update_freq="batch")
        return es_callback, tb_callback

    def reconstruct(self, data: np.array, model: Functional) -> Tuple[np.array, np.array]:
        """This method feeds the data through the entire decoder. Meaning that when having a trained VAE, the exact same image should come out.
        During its way through the VAE, the latent representations of the bottleneck are extracted and saved as well. The recreated images, as well as
        their latent representations are then outputted.

        :param data: The data which we would like to re-create and extract the latent representations from.
        :type data: np.array
        :param model: The tensorflow model which is the VAE
        :type model: Functional
        :return: The re-created images as well as the latent representations.
        :rtype: np.array
        """

        # With a Keras function
        encoder = K.function(model.layers[0].input, model.layers[1].output)
        decoder = K.function(model.layers[2].input, model.layers[2].output)

        all_latent_representation = []
        all_reconstructed_images = []
        for image in tqdm(data):
            image = image[np.newaxis, ...]
            latent_representation = encoder(image)
            reconstructed_images = decoder(latent_representation)

            all_latent_representation.append(latent_representation)
            all_reconstructed_images.append(reconstructed_images)

        array_latent_representation = np.concatenate(np.array(all_latent_representation))
        array_reconstructed_images = np.concatenate(np.array(all_reconstructed_images))

        latent_representation_path = os.path.join(self.paths.get_string("output_path"), "latent_representations.pickle")
        with open(latent_representation_path, "wb") as f:
            pickle.dump(array_latent_representation, f)

        return array_reconstructed_images, array_latent_representation

    def plot_evaluation(self, data: np.array, labels: np.array, reconstructed_images: np.array, latent_representation: np.array):
        """This method creates two kinds of plots. The first plot compares the inputted images with the outputted images. That comparison helps to see whether
        the VAE is properly working, since a working model should output something that resembles the original fairly well.

        :param data: Original data
        :type data: np.array
        :param labels: Labels, if there are some to work with
        :type labels: np.array
        :param reconstructed_images: The re-constructed versions of the original images
        :type reconstructed_images: np.array
        :param latent_representation: The latent representations which were collected on the way.
        :type latent_representation: np.array
        """
        self.plot_reconstructed_images(data, labels, reconstructed_images)
        self.plot_latent_representation(latent_representation, labels)

    def plot_reconstructed_images(
            self,
            data: np.array,
            labels: np.array,
            reconstructed_images: np.array,
            ):
        """This method plots the reconstructed images right below the original counterpart

        :param data: Original data
        :type data: np.array
        :param labels: Labels, of the image
        :type labels: np.array
        :param reconstructed_images: The reconstructed images
        :type reconstructed_images: np.array
        """

        number_of_example_images = self.parameters.get_int("evaluation_examples")
        fname = os.path.join(self.paths.get_string("images_path"), self.name, "reconstructed_image.png")

        sample_real_images = data[:number_of_example_images]
        sample_real_labels = labels[:number_of_example_images]
        sample_reconstructed_images = reconstructed_images[:number_of_example_images]

        fig, axs = plt.subplots(figsize=(20, 10), nrows=2, ncols=number_of_example_images)
        axs = axs.ravel()
        for i, (label, image) in enumerate(
            zip(sample_real_labels*2, np.concatenate((sample_real_images, sample_reconstructed_images)))
        ):
            axs[i].imshow(image, cmap="gray")
            axs[i].set_title(label)
            axs[i].axis("off")
        fig.savefig(fname=fname)
        plt.close()

    def plot_latent_representation(self, latent_representation: np.array, labels: np.array):
        """This function plots the latent representation of the data

        :param latent_representation: Latent representation of the entire dataset
        :type latent_representation: np.array
        :param labels: The labels of the original data.
        :type labels: np.array
        """

        fname = os.path.join(self.image_path, self.name, "latent_representation.png")
        comp_latent = PCA(n_components=2).fit_transform(latent_representation)

        fig, axs = plt.subplots(figsize=(10, 10))
        sns.scatterplot(
            x=comp_latent[:, 0],
            y=comp_latent[:, 1],
            hue=labels,
            ax=axs,
            alpha=0.5,
            palette="Set2",
        )
        fig.savefig(fname=fname)
        plt.close()
