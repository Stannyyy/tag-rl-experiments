# Import packages
import datetime
from experiment4 import Experiment

# Experiment
experiment = "new"#sys.argv  [0]
if experiment == "new":
    experiment = f"/experiment-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
experiment = Experiment(experiment)
experiment.continue_experiment()
