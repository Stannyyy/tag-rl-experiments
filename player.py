# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:14:00 2021

@author: StannyGoffin
"""

# Import packages
import random
import numpy as np
import math
from model import Model


# Player
class Player(Model):

    def __init__(self, name, bootstrapValueEpsilon = 0.001, discountFactor = 0.98,
        learningRate = 0.0001, layers = [50,50], render=False, just_like = None):

        # Import model
        Model.__init__(self, learningRate = learningRate, layers = layers)

        # Identifying variables
        self._name = name if just_like is None else just_like._name + name
        self.isRandom = False

        # Model variables
        self._eps = self.maxEpsilon if just_like is None else just_like._eps
        self._bootstrapValueEpsilon = bootstrapValueEpsilon  # formerly lambda
        self._discountFactor = discountFactor  # formerly gamma

        # Experience variables (carrying over using just_like)
        self._steps = 0 if just_like is None else just_like._steps
        self._samples = [] if just_like is None else just_like._samples.copy()

        # Render variables
        self._render = render

        # Sample variables
        self._state = np.array([])
        self._sample = []

        # Collection variables
        self._reward_store = []

        # State variables
        self._reward = 0
        self._tot_reward = 0

    def choose_action(self, options, save_game):

        # Use chance to see whether to explore or exploit
        chance_value = random.random()
        if (chance_value < self._eps) and (save_game == False):
            choice = random.sample(options, k=1)[0]
        else:
            prediction = self.predict_one(self._state)
            prediction = [p if i in options else -np.inf for i, p in enumerate(prediction)]
            choice = np.argmax(prediction)
        return choice

    def get_name(self):
        return self._name
    name = property(get_name)

    def get_model(self):
        return self._model
    model = property(get_model)

    def get_losses(self):
        return self._losses
    losses = property(get_losses)

    def get_reward_store(self):
        return self._reward_store
    reward_store = property(get_reward_store)

    def get_eps(self):
        return self._eps
    eps = property(get_eps)

    def get_samples(self):
        return self._samples
    def set_samples(self, samples):
        self._samples = samples
    samples = property(get_samples, set_samples)

    def set_state(self, state):
        self._state = state
    state = property(set_state)
    
    def add_sample(self):
        if len(self._sample) == 5 and len(self._sample[0]) == 6:
            self._samples += [self._sample]
            self._sample = []
        if len(self._samples) > self.maxMemory:
            self._samples = self._samples[-self.maxMemory:]

    def learn_by_replay(self, do_update_epsilon = True):

        # Make random batch
        batch = random.sample(self._samples, k=self.batchSize)

        # Predict Q(s,a) given the batch of states
        states = np.array([val[0] for val in batch])
        q_s_a = self.predict_batch(states)

        # Predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        next_states = np.array([(np.zeros(self.numStates) if val[3] is None else val[3]) for val in batch])
        q_s_a_d = self.predict_batch(next_states)

        # Set up training arrays
        x = np.zeros((len(batch), self.numStates))
        y = np.zeros((len(batch), self.numActions))

        # Now loop over batch
        for i, b in enumerate(batch):

            # Extract sample
            state, action, reward, next_state, options = b[0], b[1], b[2], b[3], b[4]

            # Get the corrected q values for all actions in state
            corrected_q = q_s_a[i]

            # Update the q value for action
            if next_state is None:
                corrected_q[action] = reward
            else:
                prediction_next_state = np.amin([np.amax(q_s_a_d[i][options])])

                corrected_q[action] = reward + self._discountFactor * prediction_next_state

            x[i] = state
            y[i] = corrected_q

        self.train_batch(x, y)
        if do_update_epsilon:
            self.update_epsilon()
        
    def update_epsilon(self):
        self._eps = self.minEpsilon + (self.maxEpsilon - self.minEpsilon) * math.exp(-self._bootstrapValueEpsilon * self._steps)
        self._steps += 1

    def new_game(self):
        self._tot_reward = 0


# Player
class RandomPlayer():

    def __init__(self, name):

        # Identifying variables
        self._name = name
        self.isRandom = True

        # Collection variables
        self._reward_store = []

        # State variables
        self._reward = 0
        self._tot_reward = 0

    def choose_action(self, options, save_game):
        return random.sample(options, k=1)[0]

    def get_name(self):
        return self._name
    name = property(get_name)

    def get_reward_store(self):
        return self._reward_store
    reward_store = property(get_reward_store)

    def new_game(self):
        self._tot_reward = 0

    def set_state(self, state):
        pass

    def update_epsilon(self):
        pass

    def add_sample(self):
        pass

    def learn_by_replay(self, do_update_epsilon):
        pass

