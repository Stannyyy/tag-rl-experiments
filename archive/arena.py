# Import packages
import time
import matplotlib.pyplot as plt
import scipy
import numpy as np
from config import retrieve_config

# Extract all variables from the config file
config = retrieve_config()

# Arena
class Arena:
    def __init__(self, modertr):
        self.cnt = 0
        self.stt = time.time()
        self.loss_check = True
        self.modertr = modertr
        self.NUM_EPISODES = config['NUM_EPISODES']
        self.PRINT_EVERY = config['PRINT_EVERY']
        self.IS_RANDOM = config['IS_RANDOM']
        self.RENDER = config['RENDER']
        self._round = 0

    def play_and_learn(self):

        # Loop for number of episodes
        while self.cnt < self.NUM_EPISODES:

            if self.cnt == self.NUM_EPISODES - 1:

                # Test
                av_loss_1 = np.array(self.modertr._players[0]._model._losses[-1 * self.PRINT_EVERY:]).mean().round(5)
                av_loss_2 = np.array(self.modertr._players[1]._model._losses[-1 * self.PRINT_EVERY:]).mean().round(5)
                eps_1 = round(self.modertr._players[0]._eps, 2)
                eps_2 = round(self.modertr._players[1]._eps, 2)
                av_rwd_1 = int(np.array(self.modertr._players[0]._reward_store[-1 * self.PRINT_EVERY:]).mean().round())
                av_rwd_2 = int(np.array(self.modertr._players[1]._reward_store[-1 * self.PRINT_EVERY:]).mean().round())

                # Print learning status
                self.end = time.time()
                print('Round',str(self.cnt+1), 'out of',self.NUM_EPISODES, round(self.end - self.stt), 'sec elapsed')
                print('Player 1 = av loss: ' + str(av_loss_1) + ', eps: ' + str(eps_1) + ', av reward: ' + str(av_rwd_1))
                print('Player 2 = av loss: ' + str(av_loss_2) + ', eps: ' + str(eps_2) + ', av reward: ' + str(av_rwd_2))
                self.stt = time.time()

                # Show one episode
                self.modertr.play(self.RENDER)
                self.cnt += 1

            # Play episode! & time it
            self.modertr.play(False)
            self.cnt += 1

        # Record two games to see progress, two to level out playing field and to see both players as both taggers and runners
        if self._round % 10 == 9:
            self.modertr.play(False, True, False)
            self.modertr.play(False, True, True)
        self._round += 1

        # After a game in the arena, print a for each player
        plt.plot(self.modertr._players[0]._model._losses, label = self.modertr._players[0]._name)
        plt.plot(self.modertr._players[1]._model._losses, label = self.modertr._players[1]._name)
        # plt.ylim([0,0.01])
        plt.legend(loc="upper left")
        plt.show()