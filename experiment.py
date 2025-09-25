from player import Player, RandomPlayer, StillPlayer
from moderator import Moderator
from arena import Arena
import os
from config import Config
import pickle
import shutil
import datetime
import numpy as np
import pandas as pd
import sys
from logs import TensorBoardLogs

class Experiment():
    def __init__(self, experiment):

        # Import config
        Config.__init__(self)

        # Experiment definition
        self._experiment = experiment
        self._total_n_training = int(self.numEpisodes / self.numEpisodesPerRound)
        self._total_n_competition = int(self.numPlayers)

        # Player definition
        # p00 = Player(experiment, name='learningRate=0.01', learningRate=0.01, useProbabilities=True)
        # p01 = Player(experiment, name='learningRate=0.001', learningRate=0.001, useProbabilities=True)
        # p02 = Player(experiment, name='learningRate=0.0001', learningRate=0.0001, useProbabilities=True)
        # p03 = Player(experiment, name='learningRate=0.00001', learningRate=0.00001, useProbabilities=True)
        # p04 = Player(experiment, name='learningRate=0.01;preselect', learningRate=0.01, useProbabilities=True, preselectBatch=True)
        # p05 = Player(experiment, name='learningRate=0.001;preselect', learningRate=0.001, useProbabilities=True, preselectBatch=True)
        # p06 = Player(experiment, name='learningRate=0.0001;preselect', learningRate=0.0001, useProbabilities=True, preselectBatch=True)
        # p07 = Player(experiment, name='learningRate=0.00001;preselect', learningRate=0.00001, useProbabilities=True, preselectBatch=True)
        p08 = Player(experiment, name='learningRate=0.00001;preselect;10epochs', learningRate=0.00001, useProbabilities=True, preselectBatch=True)

        self._players = [p08]#[p00, p01, p02, p03, p04, p05, p06, p07]

        # Initialize results paths
        experiment_path = os.path.join(os.getcwd(), experiment)
        os.makedirs(experiment_path, exist_ok=True)
        os.makedirs(os.path.join(experiment_path, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(experiment_path, 'results'), exist_ok=True)
        os.makedirs(os.path.join(experiment_path, 'state'), exist_ok=True)
        os.makedirs(os.path.join(experiment_path, 'code'), exist_ok=True)
        os.makedirs(os.path.join(experiment_path, 'competition'), exist_ok=True)

        # Add a version of the code to the code base
        shutil.copy(__file__, os.path.join(experiment_path, 'code', 'experiment.py'))
        for file in ['arena.py', 'config.py', 'game.py', 'main.py', 'model.py', 'moderator.py', 'player.py']:
            shutil.copy(os.path.join(os.getcwd(), file), os.path.join(experiment_path, 'code', file))

        # Initialize paths
        for player in self._players:
            name = player.name
            os.makedirs(os.path.join(experiment_path, 'checkpoints', name), exist_ok=True)
            for training_phase in ["part1"]:
                os.makedirs(os.path.join(experiment_path, 'checkpoints', name, training_phase), exist_ok=True)
            for competition_phase in [p.name for p in self._players]:
                os.makedirs(os.path.join(experiment_path, 'competition', name, competition_phase), exist_ok=True)

        # Initialize record of completed parts
        self._training_done, self._competition_done = self.admin_of_completed_tasks()

    def admin_of_completed_tasks(self):
        training_done = []
        competition_done = []
        for p in self._players:
            # The amount of checkpoints say something about the progress of training
            training_done += [len(
                [f for f in os.listdir(os.path.join(os.getcwd(), self._experiment, 'checkpoints', p._name, 'part1')) if
                 '-next-state' not in f]) >= self._total_n_training]

            # The existence of competition results say something about the progress of competition
            competition_done += [[os.path.exists(os.path.join(os.getcwd(), self._experiment, 'competition', p.name, p2.name, 'results.csv')) for p2 in self._players]]

        return training_done, competition_done

    def continue_experiment(self):

        # Show tensorboard command (use start_tensorboard_live for continuous updates: prone to failures)
        tf_logs = TensorBoardLogs(self._experiment)
        tf_logs.tensorboard_command()

        # Every player has a training program and competition
        # Update admin on which player has completed which program and/or competition
        self._training_done, self._competition_done = self.admin_of_completed_tasks()

        for ix, p in enumerate(self._players):
            print(p.name)

            if self._training_done[ix] is False:

                # If there is an existing training, continue there
                if os.path.exists(p._state_path):
                    arn = pickle.load(open(p._state_path, "rb", -1))
                    p.step = arn.step
                    p._eps = arn._eps
                    p._cnt = arn.cnt
                    p.reload()
                    mdrtr = Moderator([p, p], experiment=self._experiment)
                    arn.modertr = mdrtr
                else:
                    mdrtr = Moderator([p, p], experiment=self._experiment)
                    arn = Arena(mdrtr, training_phase="part1")

                # Play and learn in the arena!
                arn.play_and_learn(mode='parallel')

                # Update admin
                self._training_done, self._competition_done = self.admin_of_completed_tasks()

            if all(self._competition_done[ix]) == False:
                for icomp, comp_done in enumerate(self._competition_done[ix]):
                    # Compete against all players who completed training and whom the current player did not yet play against
                    if (comp_done is False) and (self._training_done[ix] is True) and (self._training_done[icomp] is True):

                        # Initialize competition
                        ps = [self._players[ix], self._players[icomp]]
                        for p in ps:
                            p.reload()
                        start = datetime.datetime.now().strftime('%Y%m%d-%H%M')
                        start_lens_runner = []; start_lens_tagger = []; unique_players = list(set(ps))
                        for p in unique_players:
                            start_lens_runner += [len(p._reward_store_runner)]
                            start_lens_tagger += [len(p._reward_store_tagger)]

                        # Start the competition
                        print(f"{ps[0].name} is competing against {ps[1].name}")
                        mdrtr = Moderator(ps, experiment=self._experiment)
                        arn = Arena(mdrtr, training_phase="part2")
                        arn.competition()

                        results = {'player1': ps[0].name, 'player2': ps[1].name, 'start_time': start, 'end_time': datetime.datetime.now().strftime('%Y%m%d-%H%M')}
                        for i, p in enumerate(unique_players):
                            start_len_runner = start_lens_runner[i]
                            start_len_tagger = start_lens_tagger[i]
                            results[f'player{i+1}_tagger_mean'] = np.mean(p.reward_store_tagger[start_len_tagger:])
                            results[f'player{i+1}_runner_mean'] = np.mean(p.reward_store_runner[start_len_runner:])
                            results[f'player{i+1}_tagger_median'] = np.median(p.reward_store_tagger[start_len_tagger:])
                            results[f'player{i+1}_runner_median'] = np.median(p.reward_store_runner[start_len_runner:])
                            results[f'player{i+1}_tagger_len'] = len(p.reward_store_tagger[start_len_tagger:])
                            results[f'player{i+1}_runner_len'] = len(p.reward_store_runner[start_len_runner:])
                            results[f'player{i+1}_tagger_all'] = str(p.reward_store_tagger[start_len_tagger:])
                            results[f'player{i+1}_runner_all'] = str(p.reward_store_runner[start_len_runner:])

                        df = pd.DataFrame(results, index=[0])
                        df.to_csv(os.path.join(os.getcwd(), self._experiment, 'competition', ps[0].name, ps[1].name, 'results.csv'), index=False)
                        df.to_csv(os.path.join(os.getcwd(), self._experiment, 'competition', ps[1].name, ps[0].name, 'results.csv'), index=False)

        # Update admin
        self._training_done, self._competition_done = self.admin_of_completed_tasks()
        if all(self._training_done) and all([all(c) for c in self._competition_done]):
            sys.exit(1)