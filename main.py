# Import packages
import datetime
from experiment5 import Experiment

# Experiment
experiment = "/experiment-20250815-0935"#sys.argv  [0]
if experiment == "new":
    experiment = f"/experiment-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
experiment = Experiment(experiment)
experiment.continue_experiment()
