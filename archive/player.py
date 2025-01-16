# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:14:00 2021

@author: StannyGoffin
"""

# Import packages
import random
import numpy as np
import math
from config import retrieve_config
import time

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

# Extract all variables from the config file
config = retrieve_config()

# Player
class Player:
    def __init__(self, name, model, render=False, just_like = None):
        if just_like is None:
            self._name         = name
            self._env          = np.array([])
            self._model        = model
            self._render       = render
            self._max_eps      = config['MAX_EPSILON']
            self._min_eps      = config['MIN_EPSILON']
            self._eps          = self._max_eps
            self._gamma        = config['GAMMA']
            self._lambda       = config['LAMBDA']
            self._steps        = 0
            self._reward_store = []
            self._turn_store   = []
            self._diff_memory  = []
            self._max_memory   = config['MAX_MEMORY']
            self._samples      = []
            self._samples_this_game = []
            self._tot_reward   = 0
            self._random       = False
            self._reward       = 0
            self._move         = -1
            self._options      = []
            self._next_state   = []
            self._can_move     = True
            self._timeline     = []
        else:
            self._name         = just_like._name + name
            self._env          = np.array([])
            self._model        = model
            self._max_eps      = config['MAX_EPSILON']
            self._min_eps      = config['MIN_EPSILON']
            self._eps          = just_like._eps
            self._gamma        = config['GAMMA']
            self._lambda       = config['LAMBDA']
            self._steps        = just_like._steps
            self._reward_store = []
            self._turn_store   = []
            self._diff_memory  = []
            self._max_memory   = config['MAX_MEMORY']
            self._samples      = just_like._samples.copy()
            self._samples_this_game = []
            self._tot_reward   = 0
            self._random       = False
            self._reward       = 0
            self._move         = -1
            self._options      = []
            self._next_state   = []
            self._can_move     = True
            self._timeline     = []
        self._check = []
        
    def choose_and_do_action(self,game,turn,create_video):
        options = game.what_options(turn)
        if (random.random() < self._eps) & (create_video == False):
            choice, reward = game.random_move(turn,options)

        else:
            prediction1, prediction2 = self._model.predict_one(self._env)
            prediction = [prediction1[i] + prediction2[i] for i in range(len(prediction1))]
            for i in [0,1,2,3,4,5,6,7]:
                if i not in options:
                    prediction[i] = -np.inf
            choice  = np.argmax(prediction)
            reward  = game.move(turn,choice)
        
        self._reward     += reward
        self._timeline   += [time.time()]
        self._move = choice
        return choice, reward
    
    def add_sample(self,sample):
        self._samples += [sample]
        if len(self._samples) > self._max_memory:
            self._samples = self._samples[-self._max_memory:]
    
    def make_batch(self,batch_size):
        batch = random.sample(self._samples,k=batch_size)
        return batch
        
    def learn_by_replay(self):
        batch = self.make_batch(self._model._batch_size)
        states = np.array([val[0] for val in batch]) 
        next_states = np.array([(np.zeros(self._model._num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a1, q_s_a2 = self._model.predict_batch(states)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d1, q_s_a_d2 = self._model.predict_batch(next_states)

        # setup training arrays
        x = np.zeros((len(batch), self._model._num_states))
        y1 = np.zeros((len(batch), self._model._num_actions))
        y2 = np.zeros((len(batch), self._model._num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state, options = b[0], b[1], b[2], b[3], b[4]

            # get the current q values for all actions in state
            current_q1 = q_s_a1[i]
            current_q2 = q_s_a2[i]

            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q1[action] = reward
                current_q2[action] = reward
            else:
                # Clipped Double Q-learning
                current_q1[action] = reward + self._gamma * np.minimum(np.amax(q_s_a_d1[i][options]),
                                                                    np.amax(q_s_a_d2[i][options]))
                current_q2[action] = reward + self._gamma * np.minimum(np.amax(q_s_a_d1[i][options]),
                                                                    np.amax(q_s_a_d2[i][options]))

            x[i] = state
            y1[i] = current_q1
            y2[i] = current_q2


        self._model.train_batch(x, y1, y2)
        self.update_epsilon()
        
    def update_epsilon(self):
        self._eps = self._min_eps + (self._max_eps - self._min_eps) * math.exp(-self._lambda * self._steps)
        self._steps += 1