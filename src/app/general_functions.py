
# %% Packages

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import streamlit as st
from utils.config import load_config
from vae_model.model import VAE
from tensorflow.python.keras.engine.functional import Functional

# %% Functions

def matplotlib_settings():
    """This method loads in the standard matplotlib settings in order to get the right sizes
    """
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

def reshape_latent_representation(array: np.array) -> np.array:
    """This method adds a axis to the numpy array in order to conform with the input shape
    the decoder is expected from an array

    :param array: Array with only two dimensions
    :type array: np.array
    :return: Array with an additional batch dimension equal to one
    :rtype: np.array
    """
    array = array[np.newaxis, ...]
    return array

def load_model(vae_type: str) -> Functional:
    """This method loads the different configurations files as well as the VAE. It then initializes
    the VAE and returns the model. The model is then loading the pre-trained weights.

    :param vae_type: String whether we are looking for an image or sound vae
    :type vae_type: str
    :return: Tensorflow model which is using pre-trained weights
    :rtype: Functional
    """
    config = load_config("./src/config.conf", f"{vae_type}_vae")
    vae = VAE(config)
    model = vae.model

    weights_path = f"./model/{vae_type}_vae/weights.h5"
    model.load_weights(weights_path)
    return model

def load_decoder(vae_type: str) -> Functional:
    """This method is breaking the trained VAE into two parts and only takes the decoder part of it.

    :param vae_type: Indication whether to load the sound or image decoder
    :type vae_type: str
    :return: The decoder part of the VAE
    :rtype: Functional
    """
    model = load_model(vae_type)
    decoder = K.function(model.layers[2].input, model.layers[2].output)
    return decoder

@st.cache
def load_latent_representation(vae_type: str, label: bool) -> np.array:
    """This method loads the latent representations of the trainings data. We differentiate between cases where the label
    should also be loaded. This is necessary within the sound vae, since we do have different genres of sound. Since
    we are training only with one kind of images, we do not have these labels for images.

    :param vae_type: String indicating whether we are dealing with sound or images
    :type vae_type: str
    :param label: A boolean indicating whether we would like to load labels.
    :type label: bool
    :return: The latent representations of the trainings image or sound data with potentially a label
    :rtype: np.array
    """
    latent_path = f"./data/{vae_type}/latent_representation/latent_representations.pickle"

    if label:
        with open(latent_path, "rb") as f:
            file_names, latent_representations = pickle.load(f)
        return file_names, latent_representations

    else:
        with open(latent_path, "rb") as f:
            latent_representations = pickle.load(f)
        return latent_representations
