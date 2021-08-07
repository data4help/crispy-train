
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
        self.prepare_dirs()

        # Chop the long sound-track
        self.chopping_sound()

    def chopping_sound(self) -> None:
        """This file takes the (large) inputted sound file and then chops
        it down into smaller pieces. Every single snippet is then
        saved as a wav file itself. To keep better track, an intuitive
        naming system is used. Through that the files are staying ordered.
        """
        chunk_length_ms = self.parameters.get_int("chunk_length_ms")

        self.categories = self.detect_categories()
        for category in self.categories:
            category_path = os.path.join(self.paths.input_path, category)

            for root, _, file in os.walk(category_path):
                file_name = [x for x in file if not x.startswith(".")]
                audio_path = os.path.join(root, file_name[0])
                myaudio = AudioSegment.from_file(audio_path, "wav")
                chunks = make_chunks(myaudio, chunk_length_ms)
                number_of_digits_of_length = len(str(len(chunks)))

                for i, chunk in tqdm(enumerate(chunks)):
                    padding = number_of_digits_of_length - len(str(i))
                    chunk_name = f"{category}_{padding*'0'}{str(i)}.wav"
                    full_chunk_path = os.path.join(f"{self.paths.output_path}/{category}", chunk_name)
                    chunk.export(full_chunk_path, format="wav")
