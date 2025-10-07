# Import packages
import os
from tensorboard import program

# Define Tensorboard class
class TensorBoardLogs:

    def __init__(self, experiment_name):

        # Arena variables
        self._experiment_name = experiment_name

    def tensorboard_command(self):
        print("tensorboard --logdir " + self._experiment_name + "/logs --port 6006")

    def start_tensorboard_live(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self._experiment_name+'/logs', '--port', '6006'])
        url = tb.launch()
        print(f"TensorBoard is running at {url}")

    @staticmethod
    def show_all_tensorboards():
        folder = os.listdir(os.getcwd())
        experiments = [f for f in folder if "experiment-" in f]
        port = 6007
        for experiment in experiments:
            print(os.path.join(experiment, 'logs'))
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', experiment+'/logs', '--port', str(port)])
            url = tb.launch()
            print(f"TensorBoard is running at {url}")
            port += 1