# Import packages
import os
from tensorboard import program

# Define Tensorboard class
class TensorBoardLogs():

    def __init__(self, experimentName):

        # Arena variables
        self.experiment = experimentName

    def tensorboard_command(self):
        print("tensorboard --logdir " + self.experiment + "/logs --port 6006")

    def start_tensorboard_live(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.experiment+'/logs', '--port', '6006'])
        url = tb.launch()
        print(f"TensorBoard is running at {url}")

    def show_all_tensorboards(self):
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