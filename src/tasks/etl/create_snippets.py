
# %% Packages

import os
import re
from pydub import AudioSegment
from pydub.utils import make_chunks
from ml_classes.task import MLTask
from tqdm import tqdm

# %%

class CreateSoundSnippets(MLTask):
    """This task chops a longer track into smaller sound pieces"""
    name = "make_chunks"

    def __init__(self, config):
        self.config = config
        self.wav_input_path = self.config.get_string("wav_path_input_path")
        self.wav_output_path = self.config.get_string("wav_path_output_path")

    def run(self):

        # Deleting potentially already existing files
        self.delete_if_non_empty()

        # Chop the long sound-track
        self.chopping_sound()

    def delete_if_non_empty(self):
        """This function clears all potentially already existing audio snippets
        """
        dir = self.wav_output_path
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

    def chopping_sound(self):
        """This file takes the (large) inputted sound file and then chops
        it down into smaller pieces. Every single snippet is then
        saved as a wav file itself. To keep better track, an intuitive
        naming system is used. Through that the files are staying ordered.
        """
        chunk_length_ms = self.config.get_int("chunk_length_ms")

        myaudio = AudioSegment.from_file(self.wav_input_path, "wav")
        chunks = make_chunks(myaudio, chunk_length_ms)
        number_of_digits_of_length = len(str(len(chunks)))

        for i, chunk in tqdm(enumerate(chunks)):
            padding = number_of_digits_of_length - len(str(i))
            chunk_name = f"{padding*'0'}{str(i)}.wav"
            full_chunk_path = os.path.join(self.wav_output_path, chunk_name)
            chunk.export(full_chunk_path, format="wav")
