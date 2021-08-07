# %% Packages

import os
import numpy as np
from tqdm import tqdm
from abc import abstractmethod

# %% Class


class MLTask:
    def __init__(self, config):
        self.parameters = config.get_config("parameters")
        self.paths = config.get_config("paths")

    def detect_categories(self):
        categories = [x for x in os.listdir(self.paths.input_path) if not x.startswith(".")]
        return categories

    def prepare_dirs(self) -> None:
        categories = self.detect_categories()
        for folder in categories:
            full_folder_path = os.path.join(self.paths.output_path, folder)
            if not os.path.isdir(full_folder_path):
                os.mkdir(full_folder_path)
            else:
                file_names = os.listdir(full_folder_path)
                for file in file_names:
                    os.remove(os.path.join(full_folder_path, file))

    def _load_data(self):
        x, y = [], []
        categories = self.detect_categories()
        for category in tqdm(categories):
            files = os.listdir(f"{self.paths.input_path}/{category}")
            for file in tqdm(files):
                full_file_path = os.path.join(self.paths.input_path, category, file)
                spectogram = np.load(full_file_path)
                x.append(spectogram)
                y.append(file)
        x = np.array(x)
        x = x[..., np.newaxis]
        return x, y

    @abstractmethod
    def run(self):
        pass
