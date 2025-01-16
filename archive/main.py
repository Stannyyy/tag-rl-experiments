# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 17:07:36 2021

@author: StannyGoffin
"""

# Import packages
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import tensorflow.keras as tf

# Import own packages
from config import retrieve_config
from tag import Tag
from model import Model
from player import Player
from competition import competition

# Extract all variables from the config file
config = retrieve_config()

# GET ALGORITHM GOING!
# Set up model
num_states  = 2 * config['NUM_PLAYERS'] + 1
num_actions = 8

# Set up competition

# Set up game
game = Tag(config['GRID_SIZE'], config['NUM_PLAYERS'], config['NUM_TAGGERS'])

# Load working network
# p1_new = tf.models.load_model(r'C:\Users\Stanny\OneDrive - Trifork B.V\Documents\Tag\og_grid4')
# p2_new = tf.models.load_model(r'C:\Users\Stanny\OneDrive - Trifork B.V\Documents\Tag\og_grid4')
# players = [Player('a',Model(num_states, num_actions, config['ALPHA'], model = p1_new)),
#            Player('b',Model(num_states, num_actions, config['ALPHA'], model = p2_new))]

# Create initial 10 players
players = [Player('a',Model(num_states, num_actions, config['ALPHA'])),
           Player('b',Model(num_states, num_actions, config['ALPHA'])),
           Player('c',Model(num_states, num_actions, config['ALPHA'])),
           Player('d',Model(num_states, num_actions, config['ALPHA'])),
           Player('e',Model(num_states, num_actions, config['ALPHA'])),
           Player('f', Model(num_states, num_actions, config['ALPHA']), np.array(game._x_list + game._y_list)),
           Player('g', Model(num_states, num_actions, config['ALPHA']), np.array(game._x_list + game._y_list)),
           Player('h', Model(num_states, num_actions, config['ALPHA']), np.array(game._x_list + game._y_list)),
           Player('i', Model(num_states, num_actions, config['ALPHA']), np.array(game._x_list + game._y_list)),
           Player('j', Model(num_states, num_actions, config['ALPHA']), np.array(game._x_list + game._y_list))
           ]

# Now train for 50 generations
scoreboard = pd.DataFrame()
for generation in range(20):

    # Announce generation
    print('Now training generation',generation)

    # Competition
    players, winning_players, ranked_players = competition(game,players)
    scoreboard = pd.concat((scoreboard,ranked_players.reset_index(drop=True)),ignore_index=True)

    # Set up new game
    game = Tag(config['GRID_SIZE'], config['NUM_PLAYERS'], config['NUM_TAGGERS'])

    # Mutate
    # Copy number one, three times into the new player list, number two, two times into the new player list
    new_alpha = players[winning_players[0]]._model._alpha * 0.9
    player1_model = players[winning_players[0]]._model; player1_model._alpha = new_alpha
    player2_model = Model(num_states, num_actions, new_alpha, player1_model.mutate())
    player3_model = Model(num_states, num_actions, new_alpha, player1_model.mutate())
    player4_model = players[winning_players[1]]._model; player4_model._alpha = new_alpha
    player5_model = Model(num_states, num_actions, new_alpha, player4_model.mutate())
    player6_model = Model(num_states, num_actions, new_alpha, player4_model.mutate())
    player7_model = players[winning_players[2]]._model; player7_model._alpha = new_alpha
    player8_model = Model(num_states, num_actions, new_alpha, player7_model.mutate())
    player9_model = players[winning_players[3]]._model; player9_model._alpha = new_alpha
    player0_model = Model(num_states, num_actions, new_alpha, player9_model.mutate())

    players = [Player('a',player1_model, just_like = players[winning_players[0]]),
               Player('b',player2_model, just_like = players[winning_players[0]]),
               Player('c',player3_model, just_like = players[winning_players[0]]),
               Player('d',player4_model, just_like = players[winning_players[1]]),
               Player('e',player5_model, just_like = players[winning_players[1]]),
               Player('f',player6_model, just_like = players[winning_players[1]]),
               Player('g',player7_model, just_like = players[winning_players[2]]),
               Player('h',player8_model, just_like = players[winning_players[2]]),
               Player('i',player9_model, just_like = players[winning_players[3]]),
               Player('j',player0_model, just_like = players[winning_players[3]])
               ]

print(scoreboard)