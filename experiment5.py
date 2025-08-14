from player import Player, RandomPlayer, StillPlayer
from moderator import Moderator
from arena import Arena
import os
from config import Config
import pickle
import shutil
from tensorboard import program
import datetime
import numpy as np
import pandas as pd

class Experiment():
    def __init__(self, experiment):

        # Import config
        Config.__init__(self)

        # Experiment definition
        self._experiment = experiment

        # Player definition
        p00 = Player(experiment, name='Pietje Puk with exploration', maxEpsilon=1)
        p01 = Player(experiment, name='Pietje Puk with small curiosity', curiosity=True, curiosity_beta=0.01, maxEpsilon=0.01)
        p02 = Player(experiment, name='Pietje Puk with medium curiosity', curiosity=True, curiosity_beta=0.1, maxEpsilon=0.01)
        p03 = Player(experiment, name='Pietje Puk with large curiosity', curiosity=True, curiosity_beta=1, maxEpsilon=0.01)
        p04 = Player(experiment, name='Pietje Puk without exploration', maxEpsilon=0.01)
        p05 = Player(experiment, name='Pietje Puk with exploration with small curiosity', curiosity=True, curiosity_beta=0.01, maxEpsilon=1)
        p06 = Player(experiment, name='Pietje Puk with exploration with medium curiosity', curiosity=True, curiosity_beta=0.1, maxEpsilon=1)
        p07 = Player(experiment, name='Pietje Puk with exploration with large curiosity', curiosity=True, curiosity_beta=1, maxEpsilon=1)
        self._players = [p00, p01, p02, p03, p04, p05, p06, p07]

        # Initialize results paths
        if not os.path.exists(os.getcwd() + experiment):
            os.mkdir(os.getcwd() + experiment)
        if not os.path.exists(os.getcwd() + experiment + '/checkpoints'):
            os.mkdir(os.getcwd() + experiment + '/checkpoints')
        if not os.path.exists(os.getcwd() + experiment + '/results'):
            os.mkdir(os.getcwd() + experiment + '/results')
        if not os.path.exists(os.getcwd() + experiment + '/state'):
            os.mkdir(os.getcwd() + experiment + '/state')
        if not os.path.exists(os.getcwd() + experiment + '/code'):
            os.mkdir(os.getcwd() + experiment + '/code')
        if not os.path.exists(os.getcwd() + experiment + '/final-results'):
            os.mkdir(os.getcwd() + experiment + '/final-results')

        # Add a version of the code to the code base
        shutil.copy(__file__, os.getcwd() + experiment + '/code/experiment.py')
        for file in ['arena.py', 'config.py', 'game.py', 'main.py', 'model.py', 'moderator.py', 'player.py']:
            shutil.copy(os.getcwd() + '/' + file, os.getcwd() + experiment + '/code/' + file)

        # Initialize paths
        for player in self._players:
            name = player.name
            if not os.path.exists(os.getcwd() + experiment + f'/checkpoints/{name}'):
                os.mkdir(os.getcwd() + experiment + f'/checkpoints/{name}')
            for training_phase in ["part1", "part2", "part3"]:
                if not os.path.exists(os.getcwd() + experiment + f'/checkpoints/{name}/{training_phase}'):
                    os.mkdir(os.getcwd() + experiment + f'/checkpoints/{name}/{training_phase}')
    def continue_experiment(self):
        self.start_tensorboard()

        total = int(self.numEpisodes/self.numEpisodesBeforePrint)
        for p in self._players:
            part1_done = len([f for f in os.listdir(os.getcwd()+f'{self._experiment}/checkpoints/{p._name}/part1') if '-next-state' not in f])
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

        # Final competition
        print("Final competition started")
        start = datetime.datetime.now().strftime('%Y%m%d-%H%M')
        for p1 in self._players:
            for p2 in self._players:
                print(f'{p1.name} is playing agains {p2.name}')

                if p1.name != p2.name:
                    ps = [p1, p2]
                else:
                    ps = [p1, p1]

                start_lens_runner = []; start_lens_tagger = []; unique_players = list(set(ps))
                for p in unique_players:
                    start_lens_runner += [len(p._reward_store_runner)]
                    start_lens_tagger += [len(p._reward_store_tagger)]

                mdrtr = Moderator(ps, experiment=self._experiment)
                arn = Arena(mdrtr, training_phase="part2")
                arn.competition(100)

                results = {'player1': p1.name, 'player2': p2.name, 'start_time': start, 'end_time': datetime.datetime.now().strftime('%Y%m%d-%H%M')}
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
                df_run = pd.DataFrame(results, index=[0])
                if 'df' in locals():
                    df = pd.concat((df, df_run))
                else:
                    df = df_run
                df.to_csv(os.getcwd()+self._experiment+'/final-results/results.csv', index=False)

    def start_tensorboard(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self._experiment[1:]+'/logs', '--port', '6006'])
        url = tb.launch()
        print(f"TensorBoard is running at {url}")

    def show_all_tensorboards(self):
        folder = os.listdir(os.getcwd())
        experiments = [f for f in folder if "experiment-" in f]
        port = 6007
        for experiment in experiments:
            print(experiment+'/logs')
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', experiment+'/logs', '--port', str(port)])
            url = tb.launch()
            print(f"TensorBoard is running at {url}")
            port += 1