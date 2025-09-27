# Import packages
import time
import numpy as np
from config import Config
import pickle
import datetime

# Arena
class Arena(Config):
    def __init__(self, training_phase="default", **kwargs):

        # Import config
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)
        
        # Arena variables
        self.cnt = 1
        self.stt = time.time()
        self.loss_check = True
        self.training_phase = training_phase
        self.start_time = datetime.datetime.now()
        self.total_time = 0

    def play_and_learn(self, mode='sequential'):
        self.start_stopwatch()
        new_round = True
        # Loop for number of episodes
        if self.cnt < self.num_episodes:
            if mode == 'sequential':
                while (self.cnt % self.num_episodes_per_round != 1) | (new_round == True):
                    new_round = False
                    for player in self.moderator.players:
                        player.step = self.cnt
                        player.cnt = self.cnt

                    # Play episode!
                    self.moderator.play_one(False)
                    if len(self.moderator.players[0].losses) > 0:
                        self.progress_bar(task='Playing episode: ' + str(self.cnt) + " with loss " + str(np.round(self.moderator.players[0].losses[-1],2)))
                    self.cnt += 1

                    # Stop the stopwatch
                    self.stop_stopwatch()
            elif mode == 'parallel':
                self.moderator.play_many()
                self.stop_stopwatch()
                self.cnt += self.num_episodes_per_round

            # Print progress
            print('\nRound', self.cnt-1, 'out of', self.num_episodes, self.total_time, 'sec elapsed')

            unique_players = list(set(self.moderator.players))
            for player in unique_players:
                if player.is_random:
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
                        self.cnt = self.num_episodes # Call it a day

            # Show a couple of episodes
            for i in range(2):
                self.moderator.play_one(self.create_video, learn=False)

            # Save models
            self.save_status()


    def save_status(self):
        unique_players = list(set(self.moderator.players))
        for p in unique_players:
            if p.is_random == False:

                # Save model
                if p.test_mode == False:
                    p.save_checkpoint(self.cnt, p.name, self.training_phase)
                    if p.curiosity:
                        p.save_checkpoint_next_state(self.cnt, p.name, self.training_phase)
                p.write_summary_to_tensorboard()

                # Save status
                p._model = 0
                p._model_next_state = 0
                p._summary_writer = ''
                p._tboard_callback = ''
                self.moderator = ''
                self.step = p.step
                self.eps = p._eps
                self.cnt = p.cnt
                with open(p._state_path, "wb") as file_:
                    pickle.dump(self, file_, -1)
                p.reload()


    def competition(self):

        # Reload players
        for p in self.moderator.players:
            p.reload()

        # Loop for number of episodes
        cnt = 0
        while cnt < 10:
            cnt += 1
            print(f"Showing {cnt} out of {10} before starting large competition")
            self.moderator.play_one(save_game=True, learn=False)

        self.moderator.play_many(learn=False)

    def start_stopwatch(self):
        self.start_time = datetime.datetime.now()
        
    def stop_stopwatch(self):
        episode_time = (datetime.datetime.now() - self.start_time).seconds
        self.total_time += episode_time
        self.start_stopwatch()

    def progress_bar(self, task, based_on='episodes', i=100, total=None):
        if based_on == 'episodes':
            total = self.num_episodes_per_round
            i = self.cnt
        elif based_on == 'i':
            if total is None:
                total = self.num_episodes_per_round / 10
        percent = int(np.ceil((100 * (i % total / float(total)))))
        if percent == 0:
            percent = 100
        bar = '█' * percent + '-' * (100 - percent)
        print(f"\r|{bar}| {percent}%   {task}  ", end="")