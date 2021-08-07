
# %% Packages

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %% Functions


def plot_evaluation(config, x_test, y_test, reconstructed_images, latent_representation):
    image_path = config.paths.images_path

    plot_reconstructed_images(config, x_test, reconstructed_images)
    plot_latent_representation(config, latent_representation, y_test)


def plot_reconstructed_images(config, images: np.array, reconstructed_images: np.array,):

    number_of_example_images = config.get_int("parameters.evaluation_examples")
    fname = os.path.join(config.paths.images_path, "reconstructed_images.png")

    sample_real_images = images[:number_of_example_images]
    sample_reconstructed_images = reconstructed_images[:number_of_example_images]

    fig, axs = plt.subplots(figsize=(20, 10), nrows=2, ncols=number_of_example_images)
    axs = axs.ravel()
    for i, image in enumerate(np.concatenate((sample_real_images, sample_reconstructed_images))):
        axs[i].imshow(image, cmap="gray")
        axs[i].axis("off")
    fig.savefig(fname=fname)


def plot_latent_representation(config, latent_representation: np.array, labels: np.array):

    fname = os.path.join(config.paths.images_path, "latent_representation.png")

    fig, axs = plt.subplots(figsize=(10, 10))
    sns.scatterplot(
        x=latent_representation[:, 0],
        y=latent_representation[:, 1],
        hue=labels,
        ax=axs,
        alpha=0.5,
        palette="Set2")
    fig.savefig(fname=fname)

