
from pyhocon import ConfigFactory

def load_config(config_path, task):
    try:
        config = ConfigFactory.parse_file(config_path)
        specific_config = config.get_config(task)
        return specific_config
    except Exception as e:
        print(f"The specified configs cannot be loaded: {e}.")
        raise
