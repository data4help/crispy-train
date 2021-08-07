# %% Packages

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pyhocon import ConfigTree

# %% Functions


def plot_evaluation(config, x_test, y_test, reconstructed_images, latent_representation):

    plot_reconstructed_images(config, x_test, reconstructed_images, y_test)
    plot_latent_representation(config, latent_representation, y_test)


def plot_reconstructed_images(
        config: ConfigTree,
        images: np.array,
        reconstructed_images: np.array,
        y_test: np.array
        ):

    number_of_example_images = config.get_int("parameters.evaluation_examples")
    fname = os.path.join(config.paths.images_path, "reconstructed_images.png")

    sample_real_images = images[:number_of_example_images]
    sample_real_labels = y_test[:number_of_example_images]
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


def plot_latent_representation(
        config: ConfigTree, latent_representation: np.array, labels: np.array
        ):

    fname = os.path.join(config.paths.images_path, "latent_representation.png")
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
