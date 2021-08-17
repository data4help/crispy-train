
# %% Packages

import os
from pydub import AudioSegment
from pydub.utils import make_chunks
from ml_classes.task import MLTask
from tqdm import tqdm

# %% Classes


class CreateSoundSnippets(MLTask):
    """This task chops a longer track into smaller sound pieces"""

    name = "make_chunks"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def run(self):

        # Delete existing files and create necessary folders
        self.clear_output_path()

        # Chop the long sound-track
        self.chopping_sound()

    def chopping_sound(self) -> None:
        """This file takes the (large) inputted sound file and then chops
        it down into smaller pieces. Every single snippet is then
        saved as a wav file itself. To keep better track, an intuitive
        naming system is used. Through that the files are staying ordered.
        """
        input_path = self.paths.get_string("input_path")
        output_path = self.paths.get_string("output_path")
        chunk_length_ms = self.parameters.get_int("chunk_length_ms")

        mix_files = [x for x in os.listdir(input_path) if x.endswith(".wav")]
        for file in mix_files:
            file_path = os.path.join(input_path, file)
            myaudio = AudioSegment.from_file(file_path, "wav")
            chunks = make_chunks(myaudio, chunk_length_ms)
            number_of_digits_of_length = len(str(len(chunks)))
            category_name = file.split("_mix")[0]

            for i, chunk in tqdm(enumerate(chunks)):
                padding = number_of_digits_of_length - len(str(i))
                chunk_name = f"{category_name}_{padding*'0'}{str(i)}.wav"
                file_output_path = os.path.join(output_path, chunk_name)
                chunk.export(file_output_path, format="wav")
