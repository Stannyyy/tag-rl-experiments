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
        p00 = Player(experiment, name='Pietje Puk')
        p01 = Player(experiment, name='Liesje Lot', bootstrapValueEpsilon=0.0001)
        p02 = Player(experiment, name='Naima Nima', bootstrapValueEpsilon=0.01)
        p03 = Player(experiment, name='Samir Smit', layers=[25, 25, 25])
        p04 = Player(experiment, name='Biesje Bos', layers=[1000, 1000, 1000])
        p05 = Player(experiment, name='Diego Delo', layers=[100, 100])
        p06 = Player(experiment, name='Arie Aaron', layers=[100, 100, 100, 100])
        p07 = Player(experiment, name='Fatima Flo', learningRate=0.01)
        p08 = Player(experiment, name='Omari Oost', learningRate=0.0001)
        p09 = Player(experiment, name='Lida Leeuw', discountFactor=0.995)
        p10 = Player(experiment, name='Kim Klasen', discountFactor=0.95)
        self._players = [p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]

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
            # Part 1 - Against still player
            part1_done = len(os.listdir(os.getcwd()+f'{self._experiment}/checkpoints/{p._name}/part1'))
            if part1_done == total:
                pass
            else:

                # If state exists, load state
                if os.path.exists(p._state_path):
                    arn = pickle.load(open(p._state_path, "rb", -1))
                    p = arn.modertr.players[0]
                    p.reload()
                else:
                    ps = StillPlayer(name='Stable Sef')
                    mdrtr = Moderator([p, ps], experiment=self._experiment)
                    arn = Arena(mdrtr, training_phase="part1")

                arn.play_and_learn()

            # Part 2 - Against random player
            p.reload()
            p.new_part("part1", "part2")
            p.set_eps(p.minEpsilon)
            part2_done = len(os.listdir(os.getcwd() + f'{self._experiment}/checkpoints/{p._name}/part2'))
            if part2_done == total:
                pass
            else:

                # If state exists, load state
                if os.path.exists(p._state_path):
                    arn = pickle.load(open(p._state_path, "rb", -1))
                    p = arn.modertr.players[0]
                    p.reload()
                else:
                    pr = RandomPlayer(name='Randy Rado')
                    mdrtr = Moderator([p, pr], experiment=self._experiment)
                    arn = Arena(mdrtr, training_phase="part2")

                arn.play_and_learn()

            # Part 3 - against oneself
            p.reload()
            p.new_part("part2", "part3")
            p.set_eps(p.minEpsilon)
            part3_done = len(os.listdir(os.getcwd() + f'{self._experiment}/checkpoints/{p._name}/part3'))
            if part3_done == total:
                pass
            else:

                # If state exists, load state
                if os.path.exists(p._state_path):
                    arn = pickle.load(open(p._state_path, "rb", -1))
                    for player in arn.modertr.players:
                        player.reload()
                else:
                    mdrtr = Moderator([p, p], experiment=self._experiment)
                    arn = Arena(mdrtr, training_phase="part3")

                arn.play_and_learn()