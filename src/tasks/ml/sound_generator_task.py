
# %% Packages

import os
import pickle
import librosa
import seaborn as sns
import numpy as np
from vae_model.model import VAE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from ml_classes.task import MLTask
from pyhocon import ConfigTree
from typing import Tuple, Dict
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.keras import backend as K
import soundfile as sf
import matplotlib.pyplot as plt

# %% Classes


class SoundGeneratorTask(MLTask):
    """
    This task uses the trained sound vae and generates all the relevant components for
    the streamlit application. This means that it creates a spectogram of a randomly chosen
    point in the latent space. This spectogram is then shown how it stands in contrast
    to all other points in the latent space. Furthermore, the sound-file is shown.
    """

    name = "sound_generator"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vae = None
        self.latent_scaler = None
        self.pca_model = None
        self.output_path = self.paths.get_string("output_path")
        self.hop_length = self.parameters.get_int("hop_length")
        self.sample_rate = self.parameters.get_int("sample_rate")
        self.min_max_dict = self.load_spectogram_dict()

        self.model = self.load_model()
        self.file_names, self.latent_representations = self.create_latent_representation(reload=False)
        self.pca_model, self.latent_scaler = self.fit_scaler()
        self.save_config()

        SMALL_SIZE = 20
        MEDIUM_SIZE = 25
        BIGGER_SIZE = 30
        plt.rc("font", size=BIGGER_SIZE)
        plt.rc("axes", titlesize=SMALL_SIZE)
        plt.rc("axes", labelsize=MEDIUM_SIZE)
        plt.rc("xtick", labelsize=SMALL_SIZE)
        plt.rc("ytick", labelsize=SMALL_SIZE)
        plt.rc("legend", fontsize=SMALL_SIZE)
        plt.rc("figure", titlesize=BIGGER_SIZE)

    def run(self):

        # Take random sound
        file_name, latent_representation = self.randomly_choose_file()

        # Generate the scatterplot and spectogram
        spectogram = self.generate_spectogram(latent_representation)

        # Generate the scatterplot
        self.generate_scatterplot(latent_representation)

        # Generate the sound signal
        self.convert_spectogram_into_audio(file_name, spectogram)

    def save_config(self):
        """This method saves the config file used to instantiate this class in order to re-call
        this class when used in the streamlit app.
        """
        with open(f"{self.output_path}/{self.name}/config.pickle", "wb") as f:
            pickle.dump(self.config, f)

    def generate_scatterplot(self, latent_representation: np.array):
        """This method gives an overview where the randomly chosen point is put within the scatter
        plot. A density plot is shown to give an idea of the distribution of the
        latent representations of the different genres.

        :param latent_representation: Latent representation of the randomly chosen sound snippet
        :type latent_representation: np.array
        """

        def get_handles_labels(axs):
            legend = axs.legend_
            handles = legend.legendHandles
            labels = [t.get_text() for t in legend.get_texts()]
            return handles, labels

        sns.set_theme(style="darkgrid")
        fig, axs = plt.subplots(figsize=(10, 10))

        scaled_latent_representation = self.pca_and_scale(latent_representation[np.newaxis, ...])
        scaled_latent_representations = self.pca_and_scale(self.latent_representations)
        category_names = [x.split("_")[0].capitalize() for x in self.file_names]

        histplot = sns.histplot(x=scaled_latent_representations[:, 0], y=scaled_latent_representations[:, 1], hue=category_names, ax=axs)
        histplot_handles, histplot_labels = get_handles_labels(axs)

        axs.scatter(x=scaled_latent_representation[:, 0], y=scaled_latent_representation[:, 1], label="Chosen", color="red", s=150)
        scatter_handles, scatter_labels = axs.get_legend_handles_labels()

        axs.legend(scatter_handles+histplot_handles, scatter_labels+histplot_labels, loc="upper left")

        fig.savefig(fname=f"{self.output_path}/{self.name}/scatterplot.png", bbox_inches="tight")
        plt.close()

    def pca_and_scale(self, latent_representation: np.array) -> np.array:
        """This method reduces the dimensionality of the latent representation and also scales the data using the minmaxscaler

        :param latent_representation: Latent representation of the data
        :type latent_representation: np.array
        :return: Scaled and dimensionality reduced version of the latent representation
        :rtype: np.array
        """
        pca_latent = self.pca_model.transform(latent_representation)
        scaler_latent = self.latent_scaler.transform(pca_latent)
        return scaler_latent

    def generate_spectogram(self, latent_representation: np.array):
        """This method uses the latent representation of the sound, decodes that representation into a spectgram
        and visualizes that result

        :param latent_representation: [description]
        :type latent_representation: np.array
        """
        def forceAspect(ax,aspect=1):
            im = ax.get_images()
            extent =  im[0].get_extent()
            ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

        reshaped_latent_representation = latent_representation[np.newaxis, ...]
        spectogram = self.decode_latent_space(reshaped_latent_representation, self.model)
        reshaped_spectogram = spectogram[0, :, :, 0]
        fig, axs = plt.subplots(figsize=(10, 10))
        axs.imshow(reshaped_spectogram)
        forceAspect(axs)
        fig.savefig(f"{self.output_path}/{self.name}/sound_spectogram.png", bbonx_inches="tight")
        plt.close()
        return reshaped_spectogram

    def randomly_choose_file(self) -> Tuple[str, np.array]:
        """This method is randomly choosing a file and then extracts the corresponding label as well as latent representation

        :param file_names: List of file names
        :type file_names: List[str]
        :param latent_representations: List of latent vector representations
        :type latent_representations: np.array
        :return: Once the name of the file as well as the corresponding latent representation
        :rtype: Tuple[str, np.array]
        """
        number_of_files = len(self.latent_representations)
        random_number = np.random.randint(low=0, high=number_of_files)

        file_name = self.file_names[random_number]
        latent_representation = self.latent_representations[random_number]

        genre = file_name.split("_")[0].capitalize()
        with open(f"{self.output_path}/{self.name}/genre.txt", "w") as f:
            f.write(genre)

        return file_name, latent_representation

    def convert_spectogram_into_audio(self, file_name: str, reconstructed_spectogram: np.array):
        """This method creates from the spectogram a sound. It does so by reversing the
        normalization.

        :param file_name: Name of the sound snippet to re-create the sound
        :type file_name: str
        :param reconstructed_spectogram: The reconstructed spectogram which was build from the decoder
        :type reconstructed_spectogram: np.array
        """
        log_spectogram = reconstructed_spectogram
        denormalized_spectogram = self.denormalize(file_name, log_spectogram)
        spectogram = librosa.db_to_amplitude(denormalized_spectogram)
        signal = librosa.griffinlim(spectogram, hop_length=self.hop_length)

        save_path = f"{self.output_path}/{self.name}/sound.wav"
        sf.write(save_path, signal, self.sample_rate)

    def load_spectogram_dict(self) -> Dict:
        """Loading the dictionary for the denormalization

        :return: The dictionary which is used for denormalizing the recreated spectograms
        :rtype: Dict
        """
        with open("./data/processed/min_max/min_max_values.pkl", "rb") as f:
            min_max_dict = pickle.load(f)
        return min_max_dict

    def denormalize(self, file_name: str, norm_array: np.array) -> np.array:
        """This we trained the neural network with normalized sound snippets, we have
        to reverse those in order to get the original sound back. Not doing that would
        distort the sound to a level at which one cannot hear what was played

        :param file_name: Name of the original file
        :type file_name: str
        :param norm_array: The normalized array
        :type norm_array: np.array
        :return: The denormalized array
        :rtype: np.array
        """
        min_max_values = self.min_max_dict[file_name.split(".npy")[0]]
        min_value, max_value = min_max_values["min"], min_max_values["max"]
        denormalized_array = norm_array * (max_value - min_value) + min_value
        return denormalized_array

    def load_config(self) -> ConfigTree:
        """This method loads the config which was used for the VAE. Importing
        this is necessary in order to instantiates the model

        :return: Config file including all the parameters needed to instantiat the model
        :rtype: ConfigTree
        """
        config_path = f"{self.paths.save_path}/config.pkl"
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        return config

    def load_model(self) -> Functional:
        """This method instantiates the VAE and then loads the pretrained weights

        :return: Tensorflow model with pretrained weights
        :rtype: Functional
        """
        config = self.load_config()
        self.vae = VAE(config)
        model = self.vae.model
        weights_path = f"{self.paths.save_path}/sound_vae_weights.h5"
        model.load_weights(weights_path)
        return model

    def create_latent_representation(self, reload: bool) -> np.array:
        """This method creates latent representations of the sound data. That is done
        by feeding the initial sound into the network and extracting the values at the
        bottleneck of the VAE.

        :param reload: Indication whether it should be reloaded
        :type reload: bool
        :return: Array with the latent representations
        :rtype: np.array
        """
        latent_path = os.path.join(self.paths.save_path, "latent_representations_w_target.pkl")
        if reload:
            data, file_name = self._load_data()
            _, latent_representation = self.vae.reconstruct(data, self.model)
            with open(latent_path, "wb") as f:
                pickle.dump([file_name, latent_representation], f)
        else:
            with open(latent_path, "rb") as f:
                file_name, latent_representation = pickle.load(f)
        return file_name[::-1], latent_representation[::-1]

    def encode_original(self, original: np.array, model: Functional) -> np.array:
        """This method encodes the image from the original image down to the
        latent space

        :param original: Original data which should be encoded
        :type original: np.array
        :param model: Trained Keras model which can be used to encode
        :type model: Functional
        :return: Latent representation of the original
        :rtype: np.array
        """
        encoder = K.function(model.layers[0].input, model.layers[1].output)
        latent_representation = encoder(original)
        return latent_representation

    def decode_latent_space(self, latent_representation: np.array, model: Functional) -> np.array:
        """This method builds the image from the latent representation

        :param latent_representation: The latent representation of the input
        :type latent_representation: np.array
        :param model: The Variational Encoder
        :type model: Functional
        :return: The resulting image
        :rtype: np.array
        """
        decoder = K.function(model.layers[2].input, model.layers[2].output)
        reconstructed_image = decoder(latent_representation)
        return reconstructed_image

    def fit_scaler(self) -> Tuple[PCA, MinMaxScaler]:
        """This method fits the pca algorithm as well as the minmaxscaler on the
        latent representations of the data

        :return: Fitted parameters
        :rtype: PCA, MinMaxScaler
        """

        # Fit the pca model
        pca_model = PCA(n_components=2)
        pca_model.fit(self.latent_representations)

        # Fit the minmax scaler
        pca_transformed_data = pca_model.transform(self.latent_representations)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(pca_transformed_data)

        scaler.transform(pca_transformed_data).shape

        return pca_model, scaler
