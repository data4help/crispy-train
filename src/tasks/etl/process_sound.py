
# %% Packages

from ml_classes.task import MLTask
from pipeline.preprocessing_pipeline import Loader
#from pipeline.preprocessing_pipeline import Preprocessing

# %%

class PreprocessSound(MLTask):
    """This task builds a variational autoencoder for sound"""
    name = "process_sound"

    def __init__(self, config):
        self.config = config


    def run(self):

        Loader(self.config)

