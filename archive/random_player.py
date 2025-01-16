# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:14:08 2021

@author: StannyGoffin
"""

# Play
class Random_Player:
    def __init__(self, env, render=True):
        self._env          = env
        self._steps        = 0
        self._reward_store = []
        self._turn_store   = []
        self._tot_reward   = 0
        self._reward       = 0
        self._random       = True
        self._can_move     = True
    
    def choose_and_do_action(self,game,turn):
        options = game.what_options(turn)
        choice, reward = game.random_move(turn,options)
        self._reward      += reward
        self._tot_reward  += reward
        return choice, reward