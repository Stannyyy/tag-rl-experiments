# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 17:07:36 2021

@author: StannyGoffin
"""

# SET VARIABLES
# Reinforcement learning model variables
MAX_EPSILON  = 0.99
MIN_EPSILON  = 0.01
LAMBDA       = 0.001
ALPHA        = 0.001
GAMMA        = 0.9
BATCH_SIZE   = 50
MAX_MEMORY   = 5000

# Game variables
GRID_SIZE    = 4

# Player variables
NUM_PLAYERS  = 2
IS_RANDOM    = [False,True] # List of len(NUM_PLAYERS) saying which are random
NUM_TAGGERS  = 1

# Do you want to render the game (I advice to do this only after the model had time to form)
RENDER       = False
RENDER_SPEED = 1 # Prints move every ~ seconds

# Experiment variables
NUM_EPISODES = 100001
PRINT_EVERY  = 50 # Shows a plot of rewards per player every ~ episodes


# Import packages
import numpy as np
import pandas as pd
import random 
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.keras as tf
import rendering
import scipy
import matplotlib.pyplot as plt
import time

# Import modules
from tag import Tag
from model import Model
from player import Player
from random_player import Random_Player
from moderator import Moderator

# GET ALGORITHM GOING!
# Set up model
num_states  = 2 * NUM_PLAYERS + 1
num_actions = 8

# Set up game
game = Tag(GRID_SIZE, NUM_PLAYERS, NUM_TAGGERS)

# Set up players
if type(IS_RANDOM) == bool:
    if IS_RANDOM:
        IS_RANDOM = [True]*NUM_PLAYERS
    else:
        IS_RANDOM = [False]*NUM_PLAYERS

players = []
for player in IS_RANDOM:
    
    # New model per player
    model = Model(num_states, num_actions,BATCH_SIZE,ALPHA)

    # Set up session
    if player: # True means random player
        players += [Random_Player(np.array(game._x_list + game._y_list))]
    else: # False means 
        players += [Player(model, np.array(game._x_list + game._y_list), 
                           MAX_MEMORY, MAX_EPSILON, MIN_EPSILON, GAMMA, 
                           LAMBDA)]

# Set up moderator
modertr = Moderator(game,players,BATCH_SIZE,RENDER,RENDER_SPEED)

# Play all the episodes by modertr.run() - Show every PRINT_EVERY episodes
# what the total reward per player is over time
cnt = 0

# You can always stop by Ctrl + c and run manually from here
stt = time.time()
while cnt < NUM_EPISODES:
   
    if (cnt % PRINT_EVERY == 0)&(cnt!=0):
        
        # Test
        plyrs   = []
        av_loss = []
        eps     = []
        for plyr, rndm in enumerate(IS_RANDOM):
            if plyr == False:
                # If another PRINT_EVERY episodes are played, show graph
                plt.plot(scipy.ndimage.filters.gaussian_filter1d(modertr._players[plyr]._model._losses, len(modertr._players[plyr]._model._losses) /10))
                av_loss = av_loss + [np.array(modertr._players[plyr]._model._losses[-1*PRINT_EVERY:]).mean().round(1)]
                eps     = eps + [round(modertr._players[plyr]._eps,2)]
                plyrs  += ['Player '+str(plyr+1)]
            
        plt.xlabel("# Episodes")
        plt.ylabel("Loss")
        # plt.xlim([0, NUM_EPISODES])
        # plt.ylim([0, 2])
        plt.legend(plyrs)
        time_now = '-'.join([('{0:0'+str(max(len(str(t)),2))+'d}').format(t) for i, t in enumerate(time.localtime()[0:5])])
        plt.savefig("Plot of tag game played "+time_now)
        plt.show()
        plt.close("all")
        end = time.time()
        print('This round, s elapsed: '+str(round(end-stt))+', av loss: '+str(av_loss)+', eps: '+str(eps))
        stt = time.time()
        
        # Show one episode
        modertr.play(RENDER)
        cnt += 1
    
    # Play episode! & time it
    modertr.play(False)
    cnt += 1

# See how agents are behaving - Run manually
NUM_EPISODES_SHOW = 100
for i in range(NUM_EPISODES_SHOW):
    modertr.play(RENDER)
game.shut_down_GUI()

# Remove all plot imgs
[os.remove(file) for file in os.listdir(os.getcwd()) if file.endswith('.png')]
