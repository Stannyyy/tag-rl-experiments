# Import packages
import sys
import datetime
from experiment5 import Experiment

# Experiment
experiment = sys.argv[1]
if experiment == "new":
    experiment = f"experiment-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
experiment = Experiment(experiment)
experiment.continue_experiment()
