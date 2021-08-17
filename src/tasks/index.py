
from tasks.ml.sound_task import SoundVAETask
from tasks.ml.image_task import ImageVAETask
from tasks.etl.augment_images import AugmentImages
from tasks.etl.create_snippets import CreateSoundSnippets
from tasks.etl.process_sound import PreprocessSound

tasks = [
    SoundVAETask, CreateSoundSnippets, PreprocessSound,
    ImageVAETask, AugmentImages
]
task_index = {}

for task in tasks:
    task_index[task.name] = task
