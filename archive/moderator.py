# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:14:39 2021

@author: StannyGoffin
"""

# Import packages
import random 
import time
import numpy as np
from config import retrieve_config
from watch_game import record_game

# Extract all variables from the config file
config = retrieve_config()

# Moderate game
class Moderator:
    def __init__(self, game, players):
        self._game         = game
        self._players      = players
        self._render       = config['RENDER']
        self._render_speed = config['RENDER_SPEED']
        self._n_players    = len(self._players)
        self._batch_size   = config['BATCH_SIZE']
    
    def next_turn(self,order_turns,turn):
        idx = order_turns.index(turn)
        idx += 1
        if idx == len(order_turns):
            idx = 0
        turn = order_turns[idx]
        return turn

    def previous_turn(self,order_turns,turn):
        idx = order_turns.index(turn)
        idx -= 1
        if idx == -1:
            idx = len(order_turns) - 1
        turn = order_turns[idx]
        return turn
    
    def nonrandom_players(self):
        nonrandom_players_ = []
        for p in range(self._n_players):
            if self._players[p]._random == False:
                nonrandom_players_ += [p]
        return nonrandom_players_
    
    def determine_and_assign_rewards(self):
        reward_taggers = 0
        reward_runners = 0
        for p in range(self._n_players):
            if self._game._taggers[p]:
                reward_taggers += self._players[p]._reward
            else:
                reward_runners += self._players[p]._reward

        if (reward_taggers == 0) | ((abs(reward_runners) > abs(reward_taggers)) & (abs(reward_runners) != abs(reward_taggers))):
            reward_taggers = reward_runners * -1
        if (reward_runners == 0) | ((abs(reward_runners) < abs(reward_taggers)) & (abs(reward_runners) != abs(reward_taggers))):
            reward_runners = reward_taggers * -1

        for p in range(self._n_players):
            if self._game._taggers[p]:
                self._players[p]._reward = reward_taggers
            else:
                self._players[p]._reward = reward_runners
            if len(self._players[p]._samples_this_game) > 0:
                self._players[p]._samples_this_game[len(self._players[p]._samples_this_game)-1][2] = self._players[p]._reward

    def add_reward_to_total(self):
        for p in range(self._n_players):
            self._players[p]._tot_reward += self._players[p]._reward
        
    def play(self,render,create_video = False, create_video_now = True):
        
        # Initialize game
        self._game._tot_turns = 0
        game_ended = False
        self._game.random_game()
        
        # New round of turns
        order_turns = random.sample(range(self._n_players),k=self._n_players)
        
        # For convenience, make list of non-random players
        nonrandom_players_ = self.nonrandom_players()

        # Announce that you are now recording a game
        if create_video:
            print('Now creating video')

        # Set rewards and samples to 0
        for p in range(self._n_players):
            self._players[p]._reward = 0
            self._players[p]._samples_this_game = []

        # Determine prefix if create video
        if create_video:
            if create_video_now:
                prefix = 'game2_'
            else:
                prefix = 'game1_'

        # New round of moves
        for p in range(self._n_players):
            self._players[p]._can_move = True

        # Start game
        moves = 0
        while game_ended == False:
            
            # Play turns
            for turn in order_turns:
    
                # Render
                if render:
                    self._game.render()
                    time.sleep(self._render_speed)
            
                # The state is composed of x coordinates, y coordinates and whether or not the current player is a tagger
                # Rewrite the state so players own coordinates are always written first and are normalized to 0-1
                x_list = [i / self._game._grid_size for i in self._game._x_list]
                y_list = [i / self._game._grid_size for i in self._game._y_list]
                state = np.array([x_list[turn]] + [x_list[o] for o in order_turns if o != turn] +
                                         [y_list[turn]] + [y_list[o] for o in order_turns if o != turn] +
                                         [self._game._taggers[turn]])
                self._players[turn]._env = state

                # If the runner has not already been caught
                if game_ended == False:
                    # Make a move!
                    move, reward = self._players[turn].choose_and_do_action(self._game,turn,create_video)
                    moves += 1
                    
                    # Save values for upcoming sample
                    self._players[turn]._env    = state
                    self._players[turn]._move   = move
                    self._players[turn]._reward = reward
                else:
                    self._players[turn]._can_move = False
                
                # Determine next turn
                next_turn_ = self.next_turn(order_turns, turn)

                if create_video:
                    debug_text = 'Is tagger info: ' + str(self._game._taggers) + '\n' + 'Reward player 0: ' + str(self._players[0]._reward) + '\n' + 'Reward player 1: ' + str(self._players[1]._reward) + '\n' + 'Total reward player 0: ' + str(self._players[0]._tot_reward) + '\n' + 'Total reward player 1: ' + str(self._players[1]._tot_reward) + '\n' + 'Turns: ' + str(self._game._tot_turns)
                    self._game.save(debug_text,prefix=prefix)
                    self._game._tot_turns += 1

                # Determine if game ended and determine next state
                if game_ended == False:
                    if (np.abs(reward)>1)|(moves>=150):
                        for p in range(self._n_players):
                            self._players[p]._next_state = None
                        game_ended = True
                    else:
                        # In the next state, it's the next players turn
                        x_list = [i / self._game._grid_size for i in self._game._x_list]
                        y_list = [i / self._game._grid_size for i in self._game._y_list]
                        next_state = np.array([x_list[turn]] + [x_list[o] for o in order_turns if o != turn] +
                                         [y_list[turn]] + [y_list[o] for o in order_turns if o != turn] +
                                         [self._game._taggers[turn]])
                        self._players[turn]._next_state = next_state

                    self._players[turn]._options = self._game.what_options(next_turn_)

            # Add samples to memory for the players that are non-random
            for p in nonrandom_players_:
                # Now add the new sample
                if self._players[p]._can_move:

                    sample = [self._players[p]._env, self._players[p]._move, self._players[p]._reward, self._players[p]._next_state, self._players[p]._options]
                    self._players[p]._samples_this_game += [sample]

                    if self._players[p]._next_state is None:
                        # Add last state more often into memory as a calibration point
                        for x in range(self._game._grid_size):
                            self._players[p]._samples_this_game += [sample]

            # Add total rewards
            self.add_reward_to_total()

            # Learn             
            if game_ended:

                # Add inverted rewards from the other team to each player
                self.determine_and_assign_rewards()

                # Create video
                if create_video:

                    if create_video_now:
                        debug_text = 'Is tagger info: ' + str(self._game._taggers) + '\n' + 'Reward player 0: ' + str(self._players[0]._reward) + '\n' + 'Reward player 1: ' + str(self._players[1]._reward) + '\n' + 'Total reward player 0: ' + str(self._players[0]._tot_reward) + '\n' + 'Total reward player 1: ' + str(self._players[1]._tot_reward) + '\n' + 'Turns: ' + str(self._game._tot_turns)
                        self._game.save(debug_text,prefix=prefix)
                        game_name = ' is playing against '.join([p._name for p in self._players])
                        record_game(game_name)
                    else:
                        debug_text = 'Is tagger info: ' + str(self._game._taggers) + '\n' + 'Reward player 0: ' + str(self._players[0]._reward) + '\n' + 'Reward player 1: ' + str(self._players[1]._reward) + '\n' + 'Total reward player 0: ' + str(self._players[0]._tot_reward) + '\n' + 'Total reward player 1: ' + str(self._players[1]._tot_reward) + '\n' + 'Turns: ' + str(self._game._tot_turns)
                        self._game.save(debug_text,prefix=prefix)

                # Render
                self._game._tot_turns = 0
                if render:
                    self._game.render()
                    time.sleep(self._render_speed)

                # All players learn!
                for p in nonrandom_players_:
                    
                    # Only start learning once memory has reached batch size
                    if len(self._players[p]._samples) > self._batch_size:
                        
                        # Learn!
                        self._players[p].learn_by_replay()
                    
                    # Reset player 
                    self._players[p]._reward_store.append(float(self._players[p]._tot_reward))
                    self._players[p]._tot_reward = 0

                # Now other team is the taggers
                self._game._taggers = [t == False for t in self._game._taggers]