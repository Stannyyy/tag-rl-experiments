# Import packages
import time
import numpy as np
from config import Config
import pickle
import datetime

# Arena
class Arena(Config):
    def __init__(self, modertr, training_phase="default"):

        # Import config
        Config.__init__(self)
        
        # Arena variables
        self.cnt = 1
        self.stt = time.time()
        self.loss_check = True
        self.modertr = modertr
        self.training_phase = training_phase
        self.start_time = datetime.datetime.now()
        self.total_time = 0

    def play_and_learn(self, mode='sequential'):
        self.start_stopwatch()
        new_round = True
        # Loop for number of episodes
        if self.cnt < self.numEpisodes:
            if mode == 'sequential':
                while (self.cnt % self.numEpisodesPerRound != 1) | (new_round == True):
                    new_round = False
                    for player in self.modertr.players:
                        player.step = self.cnt
                        player._cnt = self.cnt

                    # Play episode!
                    self.modertr.play_one(False)
                    if len(self.modertr.players[0]._losses) > 0:
                        self.progress_bar(task='Playing episode: ' + str(self.cnt) + " with loss " + str(np.round(self.modertr.players[0]._losses[-1],2)))
                    self.cnt += 1

                    # Stop the stopwatch
                    self.stop_stopwatch()
            elif mode == 'parallel':
                self.modertr.play_many()
                self.stop_stopwatch()
                self.cnt += self.numEpisodesPerRound

            # Print progress
            print('\nRound', self.cnt-1, 'out of', self.numEpisodes, self.total_time, 'sec elapsed')

            unique_players = list(set(self.modertr.players))
            for player in unique_players:
                if player.isRandom:
                    # Print progress
                    av_rwd_tagger = np.array(player.reward_store_tagger[-100:]).mean().round(5)
                    av_rwd_runner = np.array(player.reward_store_tagger[-100:]).mean().round(5)
                    print(player.name + '; av reward tagger: ' + str(
                        av_rwd_tagger) + ', av reward runner: ' + str(av_rwd_runner))

                    # Print progress
                    av_loss = np.array(player.losses[-100:]).mean().round(5)
                    av_rwd_tagger = np.array(player.reward_store_tagger[-100:]).mean().round(5)
                    av_rwd_runner = np.array(player.reward_store_runner[-100:]).mean().round(5)
                    eps = round(player.eps, 2)
                    print(player.name + '; av loss: ' + str(av_loss) + ', eps: ' + str(eps) + ', av reward tagger: ' + str(
                        av_rwd_tagger) + ', av reward runner: ' + str(av_rwd_runner))

                    # Check if learning done
                    if av_loss < 0.0001:
                        self.cnt = self.numEpisodes # Call it a day

            # Show a couple of episodes
            for i in range(2):
                self.modertr.play_one(self.createVideo, learn=False)

            # Save models
            self.save_status()


    def save_status(self):
        unique_players = list(set(self.modertr.players))
        for p in unique_players:
            if p.isRandom == False:

                # Save model
                if p._test_mode == False:
                    p.save_checkpoint(self.cnt, p.name, self.training_phase)
                    if p._curiosity:
                        p.save_checkpoint_next_state(self.cnt, p.name, self.training_phase)
                p.write_summary_to_tensorboard()

                # Save status
                p._model = 0
                p._model_next_state = 0
                p._summary_writer = ''
                p._tboard_callback = ''
                self.modertr = ''
                self.step = p.step
                self._eps = p._eps
                self._cnt = p._cnt
                with open(p._state_path, "wb") as file_:
                    pickle.dump(self, file_, -1)
                p.reload()


    def competition(self):

        # Reload players
        for p in self.modertr.players:
            p.reload()

        # Loop for number of episodes
        cnt = 0
        while cnt < 10:
            cnt += 1
            print(f"Showing {cnt} out of {10} before starting large competition")
            self.modertr.play_one(save_game=True, learn=False)

        self.modertr.play_many(learn=False)

    def start_stopwatch(self):
        self.start_time = datetime.datetime.now()
        
    def stop_stopwatch(self):
        episode_time = (datetime.datetime.now() - self.start_time).seconds
        self.total_time += episode_time
        self.start_stopwatch()

    def progress_bar(self, task, based_on='episodes', i=100, total=None):
        if based_on == 'episodes':
            total = self.numEpisodesPerRound
            i = self.cnt
        elif based_on == 'i':
            if total is None:
                total = self.numEpisodesPerRound / 10
        percent = int(np.ceil((100 * (i % total / float(total)))))
        if percent == 0:
            percent = 100
        bar = '█' * percent + '-' * (100 - percent)
        print(f"\r|{bar}| {percent}%   {task}  ", end="")