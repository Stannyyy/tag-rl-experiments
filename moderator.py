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

# Moderate game
class Moderator(Config):

    def __init__(self, players):

        # Import config
        Config.__init__(self)

        # Moderator variables
        self._players = players
        self._randomPlayers = [p.isRandom for p in players]
        self._order_turns = random.sample(range(self.numPlayers), k=self.numPlayers)
        self._turn = self._order_turns[0]
        self._turn_count = 0
        self._game_continues = True
        self._game = Game()

    def get_players(self):
        return self._players
    players = property(get_players)

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

            # Player who's turn it is, is random
            isNotRandom = self._randomPlayers[self._turn]==False

            # Get options for player
            options = self._game.what_options(self._turn)

            # Align game and player state
            if isNotRandom:
                self._players[self._turn].set_state(self._game._x_list + self._game._y_list +
                                                    [self._turn, int(self._game._taggers[self._turn])])

                # Append next state and its options to sample (idx 3 and 4 of sample)
                if len(self._players[self._turn]._sample) == 3:
                    self._players[self._turn]._sample += [self._players[self._turn]._state,
                                                          options]
                if len(self._players[self._turn]._sample) == 5:
                    self._players[self._turn].add_sample()

            # Make a move!
            choice = self._players[self._turn].choose_action(options, save_game)
            reward = self._game.move(self._turn, choice)
            self._turn_count += 1

            # Render
            if save_game:
                self._game.render()

            # Append to sample (idx 0, 1, 2 of sample)
            self._players[self._turn]._reward = reward
            self._players[self._turn]._tot_reward += reward
            if isNotRandom:
                self._players[self._turn]._sample = [self._players[self._turn]._state, choice, reward]
            
            # Write if save video
            if save_game:
                text = self.write_video_text()
                self._game.save(text, prefix=str(self._turn_count+1))

            # Determine if game ended and determine next state
            if self._turn_count >= 50:
                self._game._ended = True

            if self._game._ended == False:
                self.next_turn()

        # Create video
        if save_game:
            game_name = ' is playing against '.join([p._name for p in self._players]) + ' on ' + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            self._game.record(game_name)

        # All players learn!
        for p in range(self.numPlayers):

            # Is p random?
            isNotRandom = self._randomPlayers[p]==False

            # When game ended because of max of turns, reward is multiplied by 10
            if self._turn_count >= 50:
                self._players[p]._tot_reward = self._players[p]._tot_reward - self._players[p]._reward
                self._players[p]._reward = self._players[p]._reward * 10
                self._players[p]._tot_reward = self._players[p]._tot_reward + self._players[p]._reward

                if isNotRandom:
                    self._players[p]._sample[2] = self._players[p]._reward

            # Change the reward of the other player to be the negative of  the reward of the current player
            elif (p != self._turn):
                self._players[p]._tot_reward = self._players[p]._tot_reward - self._players[p]._reward
                self._players[p]._reward = self._players[self._turn]._reward * -10
                self._players[p]._tot_reward = self._players[p]._tot_reward + self._players[p]._reward
                if isNotRandom:
                    if len(self._players[p]._sample) >= 3:
                        self._players[p]._sample[2] = self._players[p]._reward

            # Append next state and its options to sample (idx 3 and 4 of sample)
            if isNotRandom:
                if len(self._players[p]._sample) == 3:
                    self._players[p]._sample += [None, None]
                if len(self._players[p]._sample) == 5:
                    self._players[p].add_sample()

            # Reset player
            self._players[p]._reward_store.append(float(self._players[p]._tot_reward))
            self._players[p]._tot_reward = 0

            # Duplicate last 10 samples
            for dup in range(1, min(11, self._turn_count)):
                self._players[p]._samples += self._players[p]._samples[(dup*-1):]

        # Now other team is the taggers
        self._game._taggers = [t == False for t in self._game._taggers]

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
