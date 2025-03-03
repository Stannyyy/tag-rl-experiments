# Import packages
import time
import numpy as np
from config import Config
import os
import pickle

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


        # Initialize paths
        if not os.path.exists(os.getcwd() + modertr._experiment + f'/checkpoints/{training_phase}'):
            os.mkdir(os.getcwd() + modertr._experiment + f'/checkpoints/{training_phase}')
        for player in self.modertr.players:
            name = player.name
            if not os.path.exists(os.getcwd() + modertr._experiment + f'/checkpoints/{training_phase}/{name}'):
                os.mkdir(os.getcwd() + modertr._experiment + f'/checkpoints/{training_phase}/{name}')

    def play_and_learn(self):
        
        # Loop for number of episodes
        while self.cnt < self.numEpisodes:

            if ((self.cnt + 1) % self.numEpisodesBeforePrint == 0) & (self.cnt != 0):

                # Print progress
                self.end = time.time()
                print('Round', str(self.cnt + 1), 'out of', self.numEpisodes, round(self.end - self.stt), 'sec elapsed')

                # Print interim results
                self.progress_bar(task='Print interim results')
                for p in range(len(self.modertr.players)):
                    if self.modertr._randomPlayers[p]:
                        # Print progress
                        av_rwd = np.array(self.modertr.players[p].reward_store[-100:]).mean().round(5)
                        print(self.modertr.players[p].name + ' = av reward: ' + str(av_rwd))

                    else:
                        # Print progress
                        av_loss = np.array(self.modertr.players[p].losses[-100:]).mean().round(5)
                        av_rwd = np.array(self.modertr.players[p].reward_store[-100:]).mean().round(5)
                        eps = round(self.modertr.players[p].eps, 2)
                        print(self.modertr.players[p].name + ' = av loss: ' + str(av_loss) + ', eps: ' + str(eps) + ', av reward: ' + str(
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
                        p.save_checkpoint(p.model, self.cnt, p.name, self.training_phase)

                # Save state
                with open(os.getcwd() + self.modertr._experiment + "/state" + "/" + self.modertr.players[0]._name + "-" + self.training_phase + ".pickle", "wb") as file_:
                    pickle.dump(self, file_, -1)

                # Start new timer
                self.stt = time.time()

            # Play episode!
            self.progress_bar(task='Playing episode: '+str(self.cnt))
            self.modertr.play(False)
            self.cnt += 1

    def progress_bar(self, task, based_on='episodes', i=100):
        if based_on == 'episodes':
            total = self.numEpisodesBeforePrint
            percent = round(100 * (self.cnt % total / float(total)))
        elif based_on == 'i':
            total = self.numEpisodesBeforePrint/10
            percent = round(100 * (i % total / float(total)))
        if percent > 97:
            percent = 100
        bar = '█' * int(percent) + '-' * (100 - int(percent))
        print(f"\r|{bar}| {percent}%   {task}  ", end="")
