# Import packages
import time
import plotly.graph_objects as go
import numpy as np
from config import Config

# Arena
class Arena(Config):
    def __init__(self, modertr, update_epsilon_after=1000, nr_learning_episodes=1000, training_phase="default"):

        # Import config
        Config.__init__(self)
        
        # Arena variables
        self.cnt = 0
        self.stt = time.time()
        self.loss_check = True
        self.modertr = modertr
        self.update_epsilon_after = update_epsilon_after
        self.nr_learning_episodes = nr_learning_episodes
        self.training_phase = training_phase

    def play_and_learn(self):
        
        # Loop for number of episodes
        while self.cnt < self.numEpisodes:

            if ((self.cnt + 1) % self.numEpisodesBeforePrint == 0) & (self.cnt != 0):

                # Print progress
                self.end = time.time()
                print('Round', str(self.cnt + 1), 'out of', self.numEpisodes, round(self.end - self.stt), 'sec elapsed')

                # Learning!
                for i in range(self.nr_learning_episodes):
                    if self.cnt > self.update_epsilon_after:
                        do_update_epsilon = True
                    else:
                        do_update_epsilon = False
                    for p in range(len(self.modertr.players)):
                        if self.modertr._randomPlayers[p] == False:
                            self.modertr.players[p].learn_by_replay(do_update_epsilon)
                    self.progress_bar(task='Learning: ' + str(i), based_on='i', i=i)

                # Initialize plot
                fig = go.Figure()
                fig.update_layout(title='Losses Over Time', xaxis_title='Episode', yaxis_title='Loss',
                                  legend=dict(x=0, y=1, traceorder='normal'))
                fig2 = go.Figure()
                fig2.update_layout(title='Rewards Over Time', xaxis_title='Episode', yaxis_title='Reward',
                                   legend=dict(x=0, y=1, traceorder='normal'))
                width = int(self.numEpisodesBeforePrint / 10)

                # Print interim results
                self.progress_bar(task='Print interim results')
                print("\n")
                for p in range(len(self.modertr.players)):
                    if self.modertr._randomPlayers[p]:
                        # Print progress
                        av_rwd = np.array(self.modertr.players[p].reward_store[-100:]).mean().round(5)
                        print(self.modertr.players[p].name + ' = av reward: ' + str(av_rwd))

                        # Add trace to plot
                        rs = np.convolve(self.modertr._players[p]._reward_store,
                                          np.ones(width) / width,
                                          mode='valid')
                        fig2.add_trace(go.Line(y=rs, mode='lines', name=self.modertr.players[p].name))
                    else:
                        # Set samples to 0
                        self.modertr.players[p].samples = []

                        # Print progress
                        av_loss = np.array(self.modertr.players[p].losses[-100:]).mean().round(5)
                        av_rwd = np.array(self.modertr.players[p].reward_store[-100:]).mean().round(5)
                        eps = round(self.modertr.players[p].eps, 2)
                        print(self.modertr.players[p].name + ' = av loss: ' + str(av_loss) + ', eps: ' + str(eps) + ', av reward: ' + str(
                            av_rwd))

                        # Add traces to plot
                        fig.add_trace(go.Scatter(y=self.modertr.players[p].losses, mode='lines', name=self.modertr.players[p].name + "1"))
                        rs = np.convolve(self.modertr._players[p]._reward_store,
                                          np.ones(width) / width,
                                          mode='valid')
                        fig2.add_trace(go.Line(y=rs, mode='lines', name=self.modertr.players[p].name))

                # Show plots
                print("\n")
                fig.show()
                fig2.show()

                # Show a couple of episodes
                for i in range(2):
                    self.modertr.play(self.createVideo)

                # Save models
                for p in self.modertr.players:
                    if p.isRandom == False:
                        self.progress_bar(task='Saving checkpoint')
                        p.save_checkpoint(p.model, self.cnt, p.name, self.training_phase)

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
        bar = '█' * int(percent) + '-' * (100 - int(percent))
        print(f"\r|{bar}| {percent}% {task}", end="")


