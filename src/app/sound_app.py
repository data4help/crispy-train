
# %% Packages


import sys
sys.path.insert(0, "./src")

import pickle
from pyhocon import ConfigTree
import streamlit as st
from src.tasks.ml.sound_generator_task import SoundGeneratorTask

# %% Pathing

file_path = "./data/results/sound_generator"

# %% Load generator class

def load_generator_config() -> ConfigTree:
    """This method loads the config file which was used to generate the sound outputs

    :return: Config file
    :rtype: ConfigTree
    """
    with open(f"{file_path}/config.pickle", "rb") as f:
        generator_config = pickle.load(f)
    return generator_config

generator_config = load_generator_config()
sound_generator = SoundGeneratorTask(generator_config)

# %% Application

st.title("Sound generation using a Variational Autoencoder")

# Space out the maps so the first one is 2x the size of the other three
c1, c2 = st.columns((2, 2))

# Left-Upper
c1.header("Scatterplot")
with c1:
    st.image(f"{file_path}/scatterplot.png")

c2.header("Spectogram")
with c2:
    st.image(f"{file_path}/sound_spectogram.png")

# Lower columns
c3, c4 = st.columns((2, 2))

# Lower-left
c3.header("Reset Button")
with c3:
    button = st.button("Create new sound snippet")
if button:
    sound_generator.run()

# Lower-right
c4.header("Play sound")
with c4:
    st.audio(f"{file_path}/sound.wav", format="audio/wav")

with open(f"{file_path}/genre.txt") as f:
    genre = f.readlines()
st.header(f"The snippet is most likely from the {genre[0]} genre")
