
from tasks.ml.image_task import ImageVAETask
from tasks.ml.sound_task import SoundVAETask
from tasks.etl.create_snippets import CreateSoundSnippets
from tasks.etl.process_sound import PreprocessSound

tasks = [ImageVAETask, SoundVAETask, CreateSoundSnippets, PreprocessSound]
task_index = {}

for task in tasks:
    task_index[task.name] = task
