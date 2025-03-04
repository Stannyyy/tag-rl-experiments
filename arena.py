# Import packages
import time
import numpy as np
from config import Config
import os
import pickle
import tensorflow as tfbare

# Arena
class Arena(Config):
    def __init__(self, modertr, training_phase="default"):

        # Import config
        Config.__init__(self)
        
        # Arena variables
        self.cnt = 0
        self.stt = time.time()
        self.loss_check = True
        self.modertr = modertr
        self.training_phase = training_phase

    def play_and_learn(self):
        
        # Loop for number of episodes
        while self.cnt < self.numEpisodes:
            self.cnt += 1
            self.progress_bar(task='Playing episode: ' + str(self.cnt))
            if (self.cnt % self.numEpisodesBeforePrint == 0) & (self.cnt != 0):

                # Print progress
                self.end = time.time()
                print('Round', self.cnt, 'out of', self.numEpisodes-1, round(self.end - self.stt), 'sec elapsed')

                for player in self.modertr.players:
                    if player.isRandom:
                        # Print progress
                        av_rwd = np.array(player.reward_store[-100:]).mean().round(5)
                        print(player.name + ' = av reward: ' + str(av_rwd))

                    else:
                        # Print progress
                        av_loss = np.array(player.losses[-100:]).mean().round(5)
                        av_rwd = np.array(player.reward_store[-100:]).mean().round(5)
                        eps = round(player.eps, 2)
                        print(player.name + ' = av loss: ' + str(av_loss) + ', eps: ' + str(eps) + ', av reward: ' + str(
                            av_rwd))

                        # Check if learning done
                        if av_loss < 0.0001:
                            self.cnt = self.numEpisodes # Call it a day

                # Show a couple of episodes
                for i in range(2):
                    self.modertr.play(self.createVideo)

                # Save models
                for p in self.modertr.players:
                    if p.isRandom == False:
                        self.progress_bar(task='Saving checkpoint')
                        if p._test_mode == False:
                            p.save_checkpoint(p.model, self.cnt, p.name, self.training_phase)

                            # Save state
                            p._model = 0
                            p._summary_writer = ''
                            with open(p._state_path, "wb") as file_:
                                pickle.dump(self, file_, -1)
                            p.reload()

                # Start new timer
                self.stt = time.time()

            # Play episode!
            self.modertr.play(False)

        # End arena
        self.cnt += 1

    def progress_bar(self, task, based_on='episodes', i=100):
        if based_on == 'episodes':
            total = self.numEpisodesBeforePrint
            percent = round(100 * (self.cnt % total / float(total)))
        elif based_on == 'i':
            total = self.numEpisodesBeforePrint/10
            percent = round(100 * (i % total / float(total)))
        if percent ==   0:
            percent = 100
        bar = '█' * int(percent) + '-' * (100 - int(percent))
        print(f"\r|{bar}| {percent}%   {task}  ", end="")