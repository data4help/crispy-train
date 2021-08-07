
from utils.config import load_config
from utils.args import parse_args
from tasks.index import task_index

args = parse_args()
task = args.task

config_path = args.config
config = load_config(config_path, task)

print(f"Running a {task} VAE.")
if task in task_index.keys():  # check if the provided task is included in the list of tasks
    task_obj = task_index[task]  # collect the task object from the task index
    task = task_obj(config)  # initialize the task object from the provided environment config
    task.run()  # run the task run() function
else:
    raise ValueError(f"The task '{task}' is not in the task index.")
