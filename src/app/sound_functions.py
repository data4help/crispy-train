# %% Packages

import pickle
import librosa
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict
import soundfile as sf
import librosa
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# %% Classes

def load_min_max_dict() -> Dict:
    """Loading the dictionary for the denormalization

    :return: The dictionary which is used for denormalizing the recreated spectograms
    :rtype: Dict
    """
    with open("./data/sound/min_max/min_max_dict.pickle", "rb") as f:
        min_max_dict = pickle.load(f)
    return min_max_dict

@st.cache
def reduce_dimensionality(latent_representation: np.array) -> PCA:
    """This method reduces the dimensionality of the sound data. That is necessary in order to better visualize whether the different
    music genres are showing different characteristics. Therefore we apply a PCA dimensionality reduction technique, using two components.

    :param latent_representation: Latent representations of the trainings-data
    :type latent_representation: np.array
    :return: The fitted PCA model
    :rtype: PCA
    """
    pca_model = PCA(n_components=2)
    pca_model.fit(latent_representation)
    return pca_model

def convert_spectogram_into_audio(
        min_max_dict: Dict, file_name: str, log_spectogram: np.array, save_path: str
        ):
    """This method creates from the spectogram a sound. It does so by reversing the
    normalization.

    :param file_name: Name of the sound snippet to re-create the sound
    :type file_name: str
    :param reconstructed_spectogram: The reconstructed spectogram which was build from the decoder
    :type reconstructed_spectogram: np.array
    """
    denormalized_spectogram = denormalize(min_max_dict, file_name, log_spectogram)
    spectogram = librosa.db_to_amplitude(denormalized_spectogram)
    signal = librosa.griffinlim(spectogram[0, :, :, 0], hop_length=256)
    sf.write(save_path, signal, 22050)

def denormalize(min_max_dict: Dict, file_name: str, norm_array: np.array) -> np.array:
    """This we trained the neural network with normalized sound snippets, we have
    to reverse those in order to get the original sound back. Not doing that would
    distort the sound to a level at which one cannot hear what was played

    :param min_max_dict: Dictionary containing the minimum and maximum value of the individual spectogram 
    :type min_max_dict: Dict
    :param file_name: Name of the original file
    :type file_name: str
    :param norm_array: The normalized array
    :type norm_array: np.array
    :return: The denormalized array
    :rtype: np.array
    """
    min_max_values = min_max_dict[file_name.split(".npy")[0]]
    min_value, max_value = min_max_values["min"], min_max_values["max"]
    denormalized_array = norm_array * (max_value - min_value) + min_value
    return denormalized_array

def list_separation(file_names: List[str]) -> Dict:
    """This method helps to differentiate the different genres. Namely it gives the indices values of the different music genres.
    For that we first have to extract the genre name of each file, before grouping them into their specific category

    :param file_names: List of all file names
    :type file_names: List[str]
    :return: Dictionary with an indication which index belongs to which genre
    :rtype: Dict
    """
    category_names = [x.split("_")[0].capitalize() for x in file_names]
    dict_index_position = {}
    for category in set(category_names):
        index_position = [i for i, x in enumerate(category_names) if x==category]
        dict_index_position[category] = index_position
    return dict_index_position

def sound_generate_scatterplot(
        latent_representations: np.array,
        latent_representation: np.array,
        file_names: List[str],
        pca_model: PCA
        ) -> Figure:
    """This method creates the scatterplot of all latent representations of the entire trainings-data as well as indicating where the chosen
    latent representation is. This is done by loading the latent representations of the entire data before then plotting another scatterplot
    with the single observation on top of it.

    :param latent_representations: Latent representations of the entire trainings data
    :type latent_representations: np.array
    :param latent_representation: Latent representation of the chosen single audio file
    :type latent_representation: np.array
    :param file_names: All file names from the entire trainings data. This is needed to indicate where each genre is.
    :type file_names: List[str]
    :param pca_model: A fitted PCA model in order to reduce the high dimension of the latent representations
    :type pca_model: PCA
    :return: A matplotlib figure of the two scatterplots.
    :rtype: Figure
    """

    sns.set_theme(style="darkgrid")
    fig, axs = plt.subplots(figsize=(10, 10))

    pca_lp = pca_model.transform(latent_representation[np.newaxis, ...])
    pca_lps = pca_model.transform(latent_representations)
    category_names = [x.split("_")[0].capitalize() for x in file_names]

    pca_lps = pca_lps[::-1]
    category_names = category_names[::-1]

    sns.histplot(
        x=pca_lps[:, 0],
        y=pca_lps[:, 1],
        hue=category_names,
        ax=axs,
    )
    histplot_handles, histplot_labels = get_handles_labels(axs)

    axs.scatter(
        x=pca_lp[:, 0],
        y=pca_lp[:, 1],
        label="Chosen",
        color="red",
        marker="x",
        s=150,
    )
    scatter_handles, scatter_labels = axs.get_legend_handles_labels()

    axs.legend(
        scatter_handles + histplot_handles,
        scatter_labels + histplot_labels,
        loc="upper left",
    )
    return fig

def force_aspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(
        abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect
    )

def generate_spectogram(spectogram: np.array) -> Figure:
    """This method uses the latent representation of the sound, decodes that representation into a spectgram
    and visualizes that result

    :param latent_representation: [description]
    :type latent_representation: np.array
    :return: A matplotlib figure of the spectogram.
    :rtype: Figure
    """

    reshaped_spectogram = spectogram[0, :, :, 0]
    fig, axs = plt.subplots(figsize=(10, 10))
    axs.imshow(reshaped_spectogram)
    force_aspect(axs)
    return fig

def read_audio(file_name):
    with open(file_name, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

def get_handles_labels(axs):
    legend = axs.legend_
    handles = legend.legendHandles
    labels = [t.get_text() for t in legend.get_texts()]
    return handles, labels
