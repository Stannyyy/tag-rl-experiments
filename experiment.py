from player import Player, RandomPlayer, StillPlayer
from moderator import Moderator
from arena import Arena
import os
import pickle
import shutil
import numpy as np
import pandas as pd
import sys
from logs import TensorBoardLogs

class Experiment:
    def __init__(self, config, experiment_name):

        # Import config
        self._config = config

        # Experiment name as attribute
        self._experiment_name = experiment_name

        # Initialize results paths
        self._path = os.path.join(os.getcwd(), self._experiment_name)

        os.makedirs(self._path, exist_ok=True)
        os.makedirs(os.path.join(self._path, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self._path, 'state'), exist_ok=True)
        os.makedirs(os.path.join(self._path, 'code'), exist_ok=True)
        os.makedirs(os.path.join(self._path, 'competition'), exist_ok=True)
        os.makedirs(os.path.join(self._path, 'display-games'), exist_ok=True)

        # Player definition
        p00 = Player(config, self._path, name='Pietje-use-same-batch', redo_batch=False)
        p01 = Player(config, self._path, name='Jantje-use-different-batch', redo_batch=True)

        self._experiment_players = [p00, p01]

        # Add a version of the code to the code base
        shutil.copy(__file__, os.path.join(self._path, 'code', 'experiment.py'))
        for file in ['arena.py', 'config.py', 'game.py', 'main.py', 'model.py', 'moderator.py', 'player.py']:
            shutil.copy(os.path.join(os.getcwd(), file), os.path.join(self._path, 'code', file))

        # Initialize paths
        for player in self._experiment_players:
            player_name = player.name
            os.makedirs(os.path.join(self._path, 'checkpoints', player_name), exist_ok=True)
            os.makedirs(os.path.join(self._path, 'checkpoints', player_name, 'part1'), exist_ok=True)
            for player2_name in [player1.name for player1 in self._experiment_players]:
                os.makedirs(os.path.join(self._path, 'competition', player_name, player2_name), exist_ok=True)
                os.makedirs(os.path.join(self._path, 'display-games', player_name, player2_name), exist_ok=True)

        # Initialize record of completed parts
        self._training_done, self._competition_done = self.admin_of_completed_tasks()

    def admin_of_completed_tasks(self):
        training_done = []
        competition_done = []
        for player1 in self._experiment_players:
            # The amount of checkpoints say something about the progress of training
            training_done += [len(
                [f for f in os.listdir(os.path.join(self._path, 'checkpoints', player1.name, 'part1')) if
                 '-next-state' not in f]) >= int(self._config.number_of_episodes_total / self._config.number_of_episodes_per_round)]

            # The existence of competition results say something about the progress of competition
            competition_done += [
                [os.path.exists(os.path.join(self._path, 'competition', player1.name, player2.name, 'results.csv')) 
                 for player2 in self._experiment_players]]

        return training_done, competition_done

    def continue_experiment(self):

        # Show tensorboard command (use start_tensorboard_live for continuous updates: prone to failures)
        tf_logs = TensorBoardLogs(self._experiment_name)
        tf_logs.tensorboard_command()

        # Every player has a training program and competition
        # Update admin on which player has completed which program and/or competition
        self._training_done, self._competition_done = self.admin_of_completed_tasks()

        for ix, player in enumerate(self._experiment_players):
            print(player.name)

            if self._training_done[ix] is False:

                # If there is an existing training, continue there
                if os.path.exists(player.state_path):
                    arena = pickle.load(open(player.state_path, "rb", -1))
                    player.reload(arena)
                    moderator = Moderator(self._config, [player, player], display_games_path = os.path.join(self._path, 'display-games'))
                    arena.moderator = moderator
                else:
                    moderator = Moderator(self._config, [player, player], display_games_path = os.path.join(self._path, 'display-games'))
                    arena = Arena(self._config, moderator)
                    player.arena = arena

                # Play and learn in the arena!
                arena.play_and_learn()

                # Update admin
                self._training_done, self._competition_done = self.admin_of_completed_tasks()

            if all(self._competition_done[ix]) == False:
                for icomp, comp_done in enumerate(self._competition_done[ix]):
                    # Compete against all players who completed training and whom the current player did not yet play against
                    if (comp_done is False) and (self._training_done[ix] is True) and (self._training_done[icomp] is True):

                        # Initialize competition
                        players = [self._experiment_players[ix], self._experiment_players[icomp]]
                        for player in players:
                            arena = pickle.load(open(player.state_path, "rb", -1))
                            player.reload(arena)
                        start_lens_runner = []; start_lens_tagger = []; unique_players = list(set(players))
                        for player in unique_players:
                            start_lens_runner += [len(player.reward_store_runner)]
                            start_lens_tagger += [len(player.reward_store_tagger)]

                        # Start the competition
                        print(f"{players[0].name} is competing against {players[1].name}")
                        moderator = Moderator(self._config, players, display_games_path = os.path.join(self._path, 'display-games'))
                        arena = Arena(self._config, moderator)
                        arena.competition()

                        results = {'player1': players[0].name, 'player2': players[1].name, 'duration': arena.episode_times[-1]}
                        for i, p in enumerate(unique_players):
                            start_len_runner = start_lens_runner[i]
                            start_len_tagger = start_lens_tagger[i]
                            results[f'player{i+1}_tagger_mean'] = np.mean(player.reward_store_tagger[start_len_tagger:])
                            results[f'player{i+1}_runner_mean'] = np.mean(player.reward_store_runner[start_len_runner:])
                            results[f'player{i+1}_tagger_median'] = np.median(player.reward_store_tagger[start_len_tagger:])
                            results[f'player{i+1}_runner_median'] = np.median(player.reward_store_runner[start_len_runner:])
                            results[f'player{i+1}_tagger_len'] = len(player.reward_store_tagger[start_len_tagger:])
                            results[f'player{i+1}_runner_len'] = len(player.reward_store_runner[start_len_runner:])
                            results[f'player{i+1}_tagger_all'] = str(player.reward_store_tagger[start_len_tagger:])
                            results[f'player{i+1}_runner_all'] = str(player.reward_store_runner[start_len_runner:])

                        df = pd.DataFrame(results, index=[0])
                        df.to_csv(os.path.join(self._path, 'competition', players[0].name, players[1].name, 'results.csv'), index=False)
                        df.to_csv(os.path.join(self._path, 'competition', players[1].name, players[0].name, 'results.csv'), index=False)

        # Update admin
        self._training_done, self._competition_done = self.admin_of_completed_tasks()
        if all(self._training_done) and all([all(c) for c in self._competition_done]):
            sys.exit(1)