# %% Packages


import sys
sys.path.insert(0, "./src")

import random
import numpy as np
import streamlit as st
from app.streamlit_helper import _get_state
from app.image_functions import image_generate_scatterplot, generate_decoded_image
from app.sound_functions import (reduce_dimensionality, sound_generate_scatterplot, generate_spectogram,
list_separation, load_min_max_dict, convert_spectogram_into_audio, read_audio)
from app.general_functions import load_decoder, load_latent_representation, reshape_latent_representation, matplotlib_settings

# %% Page handler

def main():
    """This method enables streamlit to deal with multiple pages
    """
    matplotlib_settings()

    state = _get_state()
    pages = {
        "Image VAE": image_page,
        "Sound VAE": sound_page,
    }

    st.sidebar.title(":floppy_disk: Page states")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


# %% Image-Page: Pre-loaded components

vae_type = "image"
image_decoder = load_decoder(vae_type)
latent_representations = load_latent_representation(vae_type, label=False)

# %% Image-Page: Constants

max_absolute_value = np.ceil(np.abs(np.max(latent_representations)))
min_value = -max_absolute_value.astype(float)
max_value = max_absolute_value.astype(float)
value = 0.0
step = 0.01

vae_type = "sound"
sound_decoder = load_decoder(vae_type)
file_names, latent_representations = load_latent_representation(vae_type, label=True)
dict_category_numbers = list_separation(file_names)
pca_model = reduce_dimensionality(latent_representations)
min_max_dict = load_min_max_dict()
sound_save_path = "./data/sound/results/sound.wav"

# %% Image-Page: VAE

def image_page(state):

    st.title("Bottle generation")

    # Create the sliders
    x = st.slider("Latent Space Factor 1", min_value=min_value, max_value=max_value, value=value, step=step)
    y = st.slider("Latent Space Factor 2", min_value=min_value, max_value=max_value, value=value, step=step)

    # Create both boxes
    c1, c2 = st.columns((2, 2))

    # Plotting the latent representations
    fig_image_scatter = image_generate_scatterplot(latent_representations, x, y)
    c1.pyplot(fig_image_scatter)

    # Plotting the image
    array = np.array([x, y])
    reshaped_array = reshape_latent_representation(array)
    image_array = image_decoder(reshaped_array)
    fig_image = generate_decoded_image(image_array)
    c2.pyplot(fig_image)

# %% Sound-Page: Constants

def sound_page(state):

    st.title("Sound generation using a Variational Autoencoder")

    # Button section
    st.header("Choose sound snippet")
    b1, b2, b3 = st.columns(3)

    # Left-Upper
    techno_button = b1.button("Create techno sample")
    rock_button = b2.button("Create rock sample")
    piano_button = b3.button("Create piano sample")

    if techno_button:
        choice = random.choice(dict_category_numbers["Techno"])
    elif rock_button:
        choice = random.choice(dict_category_numbers["Rock"])
    elif piano_button:
        choice = random.choice(dict_category_numbers["Piano"])
    else:
        choice = np.random.randint(len(file_names))

    file_name = file_names[choice]
    latent_representation = latent_representations[choice]
    category = file_name.split("_")[0].capitalize()

    # Image section
    c1, c2 = st.columns(2)

    # Right-Upper
    c1.header("Spectogram")
    reshaped_array = reshape_latent_representation(latent_representation)
    spectogram = sound_decoder(reshaped_array)
    fig_spectogram = generate_spectogram(spectogram)
    c1.pyplot(fig_spectogram)

    # Lower-Left
    c2.header("Latent representation")
    fig_sound_scatter = sound_generate_scatterplot(latent_representations, latent_representation, file_names, pca_model)
    c2.pyplot(fig_sound_scatter)

    # Bottem centered
    st.header(f"Sound snippet from the genre {category}")
    signal = convert_spectogram_into_audio(min_max_dict, file_name, spectogram, sound_save_path)
    st.audio(read_audio(sound_save_path))

if __name__ == "__main__":
    main()
