
# %% Packages

import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# %% Functions

def image_generate_scatterplot(latent_representations, x: float = 0.0, y: float = 0.0) -> Figure:
    """This functions plot the latent representations of all images and puts the chosen latent representation on top of it. This is done by simply overlaying
    two scatterplots.

    :param latent_representations: Latent representations of all images
    :type latent_representations: [type]
    :param x: First latent vector value, defaults to 0.0
    :type x: float, optional
    :param y: Second latent vector value, defaults to 0.0
    :type y: float, optional
    :return: A matplotlib figure showing the scatterplot
    :rtype: Figure
    """
    fig, axs = plt.subplots(figsize=(10, 10))
    axs.scatter(
        latent_representations[:, 0],
        latent_representations[:, 1],
        color="blue",
        label="Latent Space of Existing Bottles",
        edgecolors="white",
        alpha=0.5
    )
    axs.scatter(
        x,
        y,
        color="red",
        label="Chosen Latent Space Vector",
        s=200,
        marker="x"
    )
    axs.legend()
    return fig

def generate_decoded_image(image_array: np.array) -> Figure:
    """The function simply plots the results of the decoder, a.k.a the image

    :param image_array: The decoded latent representation
    :type image_array: np.array
    :return: A matplotlib figure showing the decoded latent representation
    :rtype: Figure
    """
    image_array = image_array[0, :, :, 0]
    fig, axs = plt.subplots(figsize=(10, 10))
    axs.imshow(image_array)
    axs.axis("off")
    return fig
