from player import Player, RandomPlayer, StillPlayer
from moderator import Moderator
from arena import Arena
import os
import pickle
import shutil
import numpy as np
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
        os.makedirs(os.path.join(self._path, 'display-games'), exist_ok=True)

        # Player definition
        p00 = Player(config, self._path, name='Pietje-use-same-batch', redo_batch=False)
        p01 = Player(config, self._path, name='Jantje-use-different-batch', redo_batch=True)
        p02 = Player(config, self._path, name='Pietje-use-same-batch-load-adam', redo_batch=False)
        p03 = Player(config, self._path, name='Jantje-use-different-batch-load-adam', redo_batch=True)

        self._experiment_players = [p02, p03]
        self._competition_reference_player = p00.name
        self._competition_professional_player = r'C:\Users\Stann\PycharmProjects\tag-rl-experiment\experiment-20251006-1212\checkpoints\Jantje-use-different-batch\part1\cp-1000001.weights.h5'

        # Admin
        self._training_done = {}
        self._competition_done = {}
        self._competition_to_do = {}

        # Add a version of the code to the code base
        shutil.copy(__file__, os.path.join(self._path, 'code', 'experiment.py'))
        for file in ['arena.py', 'config.py', 'game.py', 'main.py', 'model.py', 'moderator.py', 'player.py']:
            shutil.copy(os.path.join(os.getcwd(), file), os.path.join(self._path, 'code', file))

        # Initialize paths
        for player in self._experiment_players:
            player_name = player.name
            os.makedirs(os.path.join(self._path, 'checkpoints', player_name), exist_ok=True)
            os.makedirs(os.path.join(self._path, 'display-games', player_name, player_name), exist_ok=True)

    def admin_of_completed_tasks(self):

        # Initialize
        self._training_done = {}
        self._competition_done = {}
        self._competition_to_do = {}

        # Check available reference player checkpoints for competition
        reference_checkpoints = os.listdir(os.path.join(self._path, 'checkpoints', 'Jantje-use-different-batch')) # TEMPORARY REPLACE
        for player in self._experiment_players:
            self._training_done[player.name] = False
            # The amount of checkpoints say something about the progress of training
            player_checkpoints = [f for f in os.listdir(os.path.join(self._path, 'checkpoints', player.name))
                                  if '-next-state' not in f]
            if (len(player_checkpoints)
                >= int(player.config.number_of_episodes_total / player.config.number_of_episodes_per_round)):
                self._training_done[player.name] = True

            # Compare reference_checkpoints, player_checkpoints and player.competition_done
            self._competition_done[player.name] = os.listdir(os.path.join(self._path, 'display-games', player.name))
            self._competition_to_do[player.name] = [f'reference-{checkpoint.replace(".keras", ".weights.h5")}' for checkpoint in player_checkpoints
                                                    if (f"reference-{checkpoint.replace('.keras', '.gif')}" not in self._competition_done.get(player.name))
                                                    and (checkpoint.replace('.keras', '.weights.h5') in reference_checkpoints)]
            self._competition_to_do[player.name] += [f'professional-{checkpoint}' for checkpoint in player_checkpoints
                                                     if (f"professional-{checkpoint.replace('.keras', '.gif')}" not in self._competition_done.get(player.name))]

    def continue_experiment(self):

        # Show tensorboard command (use start_tensorboard_live for continuous updates: prone to failures)
        tf_logs = TensorBoardLogs(self._experiment_name)
        tf_logs.tensorboard_command()

        # Every player has a training program and competition
        # Update admin on which player has completed its training and/or competition
        self.admin_of_completed_tasks()

        for ix, player in enumerate(self._experiment_players):
            print(player.name)

            if self._training_done.get(player.name) is False:

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

            self.admin_of_completed_tasks()

            for competition in self._competition_to_do.get(player.name):
                # Initialize competition
                arena = pickle.load(open(player.state_path, "rb", -1))
                if 'professional' in competition:
                    player.reload(arena,
                                  checkpoint_path_overwrite=os.path.join(self._path, 'checkpoints', player.name,
                                                                         competition.replace('professional-', '')))
                    professional_player = Player(self._config, self._path, name="Pro")
                    professional_player.reload(arena, checkpoint_path_overwrite = self._competition_professional_player)
                    players = [player, professional_player]
                if 'reference' in competition:
                    player.reload(arena,
                                  checkpoint_path_overwrite=os.path.join(self._path, 'checkpoints', player.name,
                                                                         competition.replace('reference-', '').replace('.weights.h5', '.keras')))
                    checkpoint_path_overwrite = os.path.join(self._path, 'checkpoints', self._competition_reference_player,
                                                             competition.replace('reference-', ''))
                    reference_player = Player(self._config, self._path, name="Ref")
                    reference_player.reload(arena, checkpoint_path_overwrite=checkpoint_path_overwrite)
                    players = [player, reference_player]

                # Start the competition
                print(f"{players[0].name} is competing against {players[1].name}")
                moderator = Moderator(self._config, players, display_games_path = os.path.join(self._path, 'display-games'))
                arena = Arena(self._config, moderator)
                game_path = os.path.join(self._path, 'display-games', player.name, competition.split('.')[0])
                arena.competition(game_path)
                episode_count = int(competition.split('cp-')[1].split('.')[0])
                competitor = competition.split('-')[0]
                player.add_competition_to_tensorboard(episode_count=episode_count, competitor=competitor)

            # Write to tensorboard
            player.write_summary_to_tensorboard()

        # Update admin
        self.admin_of_completed_tasks()
        if all(self._training_done.values()) and len([j for i in self._competition_to_do.values() for j in i]) == 0:
            sys.exit(1)