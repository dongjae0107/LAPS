import sys
import shutil

from utils.tools import setup_experiment

from utils.config import Config
from utils.mapper import Mapper

def run():
    config = Config()

    if len(sys.argv) > 1:
        config.load_from_file(sys.argv[1])
    else:
        sys.exit("Please provide a config file as an argument.")

    # Set up experiment
    setup_experiment(config)
    shutil.copy(sys.argv[1], config.experiment_path)

    # Run mapper
    mapper = Mapper(config)
    mapper.mapping()

if __name__ == "__main__":
    run()