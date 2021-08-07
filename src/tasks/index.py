from tasks.ml.sound_task import SoundVAETask
from tasks.ml.sound_generator_task import SoundGeneratorTask
from tasks.etl.create_snippets import CreateSoundSnippets
from tasks.etl.process_sound import PreprocessSound

tasks = [
    SoundGeneratorTask, SoundVAETask, CreateSoundSnippets, PreprocessSound,
]
task_index = {}

for task in tasks:
    task_index[task.name] = task
