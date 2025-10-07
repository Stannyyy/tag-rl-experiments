# Import packages
import sys
import datetime
from experiment import Experiment
from config import Config

# Config
config = Config()

# Experiment
experiment = sys.argv[1]
if experiment == "new":
    experiment = f"experiment-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
experiment = Experiment(config, experiment)
experiment.continue_experiment()
