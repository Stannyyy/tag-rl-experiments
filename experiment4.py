from player import Player, RandomPlayer, StillPlayer
from moderator import Moderator
from arena import Arena
import os
from config import Config
import pickle

class Experiment():
    def __init__(self, experiment):

        # Import config
        Config.__init__(self)

        # Experiment definition
        self._experiment = experiment

        # Player definition
        p00 = Player(experiment, name='Pietje Puk', addLSTM=False)
        p01 = Player(experiment, name='Pietje Puk with memory', addLSTM=True, sequenceLength=5)
        p02 = Player(experiment, name='Pietje Puk with very short memory', addLSTM=True, sequenceLength=1)
        self._players = [p01, p00, p02]

        # Initialize results paths
        if not os.path.exists(os.getcwd() + experiment):
            os.mkdir(os.getcwd() + experiment)
        if not os.path.exists(os.getcwd() + experiment + '/checkpoints'):
            os.mkdir(os.getcwd() + experiment + '/checkpoints')
        if not os.path.exists(os.getcwd() + experiment + '/results'):
            os.mkdir(os.getcwd() + experiment + '/results')
        if not os.path.exists(os.getcwd() + experiment + '/state'):
            os.mkdir(os.getcwd() + experiment + '/state')

        # Initialize paths
        for player in self._players:
            name = player.name
            if not os.path.exists(os.getcwd() + experiment + f'/checkpoints/{name}'):
                os.mkdir(os.getcwd() + experiment + f'/checkpoints/{name}')
            for training_phase in ["part1", "part2", "part3"]:
                if not os.path.exists(os.getcwd() + experiment + f'/checkpoints/{name}/{training_phase}'):
                    os.mkdir(os.getcwd() + experiment + f'/checkpoints/{name}/{training_phase}')
    def continue_experiment(self):
        total = int(self.numEpisodes/self.numEpisodesBeforePrint)
        for p in self._players:
            part1_done = len(os.listdir(os.getcwd()+f'{self._experiment}/checkpoints/{p._name}/part1'))
            if part1_done == total:
                pass
            else:

                # If state exists, load state
                if os.path.exists(p._state_path):
                    arn = pickle.load(open(p._state_path, "rb", -1))
                    for p in arn.modertr.players:
                        p.reload()
                else:
                    mdrtr = Moderator([p, p], experiment=self._experiment)
                    arn = Arena(mdrtr, training_phase="part1")

                arn.play_and_learn()