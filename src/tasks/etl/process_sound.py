
# %% Packages

import os
import pickle
import numpy as np
import librosa
from tqdm import tqdm
from ml_classes.task import MLTask

# %% Classes


class PreprocessSound(MLTask):
    """This task preprocesses the sound for a variational autoencoder which
    works with sound data."""

    name = "process_sound"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.sr = self.parameters.get_int("sample_rate")
        self.duration = self.parameters.get("duration")
        self.mono = self.parameters.get("mono")
        self.frame_size = self.parameters.get_int("frame_size")
        self.hop_length = self.parameters.get_int("hop_length")

        self.input_path = self.paths.get_string("input_path")
        self.output_path = self.paths.get_string("output_path")
        self.min_max_value_dir = self.paths.get_string("min_max_values_save_dir")

        self.number_expected_samples = int(self.sr * self.duration)
        self.min_max_values = {}

    def run(self):

        # Clean folders from all potentially existing files
        # self.prepare_dirs()

        # Create spectogram features and min max values
        categories = self.detect_categories()
        for category in categories:
            full_folder_path = os.path.join(self.input_path, category)
            for file in tqdm(os.listdir(full_folder_path)):
                file_path = os.path.join(full_folder_path, file)
                self.process_file(file_path)
        self.save_min_max_values(self.min_max_values)

    def process_file(self, file_path: str) -> None:
        signal = self._loader(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self._logspectogramextractor(signal)
        norm_feature = self._normalize(feature)
        if self._no_nan(norm_feature):
            file_name = self._save_feature(norm_feature, file_path)
            self._store_min_max_value(file_name, feature.min(), feature.max())

    def save_min_max_values(self, min_max_values: dict) -> None:
        save_path = os.path.join(self.min_max_value_dir, "min_max_values.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(min_max_values, f)

    def create_folders(self) -> None:
        for folder in os.listdir(self.audio_dir):
            full_folder_path = os.path.join(self.features_dir, folder)
            self._create_potentially_missing_folder(full_folder_path)

    def _no_nan(self, feature: np.array) -> bool:
        if not np.any(np.isnan(feature)):
            return True
        else:
            return False

    def _is_padding_necessary(self, signal: np.array) -> bool:
        if len(signal) < self.number_expected_samples:
            return True
        else:
            return False

    def _apply_padding(self, signal: np.array) -> np.array:
        number_missing_samples = self.number_expected_samples - len(signal)
        padded_signal = self._padder(signal, number_missing_samples)
        return padded_signal

    def _store_min_max_value(self, file_name: str, min_value: np.array, max_value: np.array) -> None:
        self.min_max_values[file_name] = {
            "min": min_value,
            "max": max_value
        }

    def _save_feature(self, feature: np.array, file_path: str) -> str:
        file_name_w_extension = os.path.split(file_path)[1]
        file_name = file_name_w_extension.split(".")[0]
        category = file_name.split("_")[0]
        save_path = os.path.join(self.output_path, category, f"{file_name}.npy")
        np.save(save_path, feature)
        return file_name

    def _loader(self, file_path: str) -> np.array:
        signal = librosa.load(file_path, sr=self.sr, duration=self.duration, mono=self.mono)[0]
        return signal

    def _padder(self, array: np.array, num_missing_items: int, mode: str = "constant") -> np.array:
        padded_array = np.pad(array, (0, num_missing_items), mode=mode)
        return padded_array

    def _logspectogramextractor(self, signal: np.array) -> np.array:
        stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length)[:-1]
        spectogram = np.abs(stft)
        log_spectogram = librosa.amplitude_to_db(spectogram)
        return log_spectogram

    def _normalize(self, array: np.array) -> np.array:
        norm_array = (array - array.min()) / (array.max() - array.min())
        return norm_array
