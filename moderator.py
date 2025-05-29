# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:14:39 2021

@author: StannyGoffin
"""

# Import packages
import random
from config import Config
from game import Game
import datetime
import tensorflow as tfbare

# Moderate game
class Moderator(Config):

    def __init__(self, players, experiment="defaultname"):

        # Import config
        Config.__init__(self)

        # Moderator variables
        self._experiment = experiment
        self._players = players
        self._randomPlayers = [p.isRandom for p in players]
        self._order_turns = random.sample(range(self.numPlayers), k=self.numPlayers)
        self._turn = self._order_turns[0]
        self._turn_count = 0
        self._game_continues = True
        self._game = Game(experiment = experiment)

    def get_players(self):
        return self._players
    players = property(get_players)

    def shuffle_players(self):
        random.shuffle(self._players)

    def next_turn(self):
        idx = self._order_turns.index(self._turn)
        idx += 1
        if idx == len(self._order_turns):
            idx = 0
        self._turn = self._order_turns[idx]
        
    def play(self, save_game = False):
        
        # Initialize game
        self._turn_count = 0
        self._game.init_random_game()

        # Reset players
        [p.new_game() for p in self._players]

        # Render
        if save_game:
            self._game.render()
            text = self.write_video_text()
            self._game.save(text, prefix=str(self._turn_count))

        # Start game
        while self._game._ended == False:

            # Determine next state
            self.next_turn()
            player = self._players[self._turn]

            # Get options for player
            options = self._game.what_options(self._turn)

            # Set new player state
            player.set_state(self._game._x_list,
                             self._game._y_list,
                             self._turn,
                             self._game._taggers[self._turn])

            # Append next state and its options to sample (idx 3 and 4 of sample)
            player.update_sample(options)
            player.add_sample()

            # Make a move!
            choice = player.choose_action(options, save_game)
            reward = self._game.move(self._turn, choice)
            self._turn_count += 1

            # Render
            if save_game:
                self._game.render()

            # Set new sample: state, choice, reward (idx 0, 1, 2 of sample)
            player._reward = reward
            player._tot_reward += reward
            player.set_sample(choice, reward)
            
            # Write if save video
            if save_game:
                text = self.write_video_text()
                self._game.save(text, prefix=str(self._turn_count+1))

            # Determine if game ended and determine next state
            if self._turn_count >= 50:
                self._game._ended = True

        # Create video
        if save_game:
            game_name = ' is playing against '.join([p._name for p in self._players]) + ' on ' + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            self._game.record(game_name)

        # All players learn!
        unique_players = list(set(self._players))
        for player in unique_players:

            # Add rewards to reward store
            player.update_reward_store()

            # Add rewards to tensorboard
            player.add_rewards_to_tensorboard(self._turn_count)

            # Append next state and its options to sample (idx 3 and 4 of sample)
            player.update_sample(None)

            # Finalize sample buffer
            player.add_sample()
            player.finalize_sample_buffer()

            # Learn!
            player.learn_by_replay(self.batchSize * (len(self._players)/len(unique_players)))

            # Reset player
            player._tot_reward = 0

        # Shuffle players for robustness
        self.shuffle_players()

    def write_video_text(self):
        text = 'Is tagger info: ' + str([i for i, x in enumerate(self._game._taggers) if x][0]) + '\n' + \
                'Reward player 0: ' + str(self._players[0]._reward) + '\n' + \
                'Reward player 1: ' + str(self._players[1]._reward) + '\n' + \
                'Total reward player 0: ' + str(self._players[0]._tot_reward) + '\n' + \
                'Total reward player 1: ' + str(self._players[1]._tot_reward) + '\n' + \
                'Turns: ' + str(self._turn_count)
        return text

    def describe_choice(self, choice):
        print({False:'o',True:'x'}.get(self._game._taggers[self._turn]))
        print({0:'up', 1:'down', 2:'left', 3:'right',
               4:'up left', 5:'up right', 6:'down left', 7:'down right'}.get(choice))

    def describe_sample(self, sample):
        print("Sample: ", sample)
        print("Role: ", {0: 'runner', 1: 'tagger'}.get(sample[0][5]))
        print("Turn: ", {0: 'left', 1: 'right'}.get(sample[0][4]))
        print("Left x,y: ", sample[0][0], sample[0][2])
        print("Right x,y: ", sample[0][1], sample[0][3])
        print("Choice: ", {0:'up', 1:'down', 2:'left', 3:'right',
               4:'up left', 5:'up right', 6:'down left', 7:'down right'}.get(sample[1]))
        print("Reward: ", sample[2])
