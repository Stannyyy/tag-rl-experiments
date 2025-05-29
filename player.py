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
import tensorflow as tfbare
import os

# Player
class Player(Model):

    def __init__(self, experiment, name, bootstrapValueEpsilon = 0.001, discountFactor = 0.99,
        learningRate = 0.001, layers = [100,100,100], render=False, justLike=None, test_mode=False):

        # Import model
        Model.__init__(self, experiment=experiment, learningRate=learningRate, layers=layers)

        # Identifying variables
        self._name = name if justLike is None else justLike._name + name
        self.isRandom = False
        self.isStill = False

        # Model variables
        self._eps = self.maxEpsilon if justLike is None else justLike._eps
        self._bootstrapValueEpsilon = bootstrapValueEpsilon  # formerly lambda
        self._discountFactor = discountFactor  # formerly gamma

        # Experience variables (carrying over using justLike)
        self._steps = 0 if justLike is None else justLike._steps
        self._samples = [] if justLike is None else justLike._samples.copy()
        self._samples_count = 0
        self._sample_buffer = []
        self._learningSteps = 0

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

        # Is the player learning? Of temporarily paused due to test mode?
        self._test_mode = test_mode

        # Save intermittant folders
        self._state_path = os.getcwd() + experiment + "/state/part1-" + self._name.replace(" ","") + ".pickle"
        self._checkpoint_path = os.getcwd() + experiment + "/checkpoints/" + self._name + "/part1/"
        self._log_path = os.getcwd() + experiment + "/logs/dql_" + self._name + "/"

        # Set up the tensorboard
        self._summary_writer = tfbare.summary.create_file_writer(self._log_path)
        self._summary_loss_step = 0
        self._summary_params_step = 0
        self._summary_reward_step = 0

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

    def set_eps(self, eps):
        self._eps = eps
    eps = property(get_eps, set_eps)

    def get_samples(self):
        return self._samples
    samples = property(get_eps)

    def set_samples(self, samples):
        self._samples = samples
    samples = property(get_samples, set_samples)

    def get_state(self):
        return self._state
    state = property(get_state)

    def set_state(self, game_x_list, game_y_list, turn, is_tagger):
        self._state = game_x_list + game_y_list + [turn, int(is_tagger)]
    state = property(set_state)

    def get_sample(self):
        return self._sample
    sample = property(get_sample)

    def set_sample(self, choice, reward):
        self._sample = [self._state, choice, reward]
    sample = property(get_sample, set_sample)

    def update_sample(self, options):
        if options is None:
            self._sample += [None, None]
        if len(self._sample) == 3:
            self._sample += [self._state, options]

    def add_sample(self):
        if len(self._sample) != 5:
            self.finalize_sample_buffer()

        if self._sample != []:
            if len(self._sample) < 4:
                self._sample = []
            elif self._sample[3] is not None:
                if self._sample[0][-2] != self._sample[3][-2]: # If role of current and next state are different
                    self._sample_buffer += [self._sample]      # due to player playing against itself: buffer to correct
                    self.correct_sample_buffer()

        if self._sample != []:
            self._samples_count += 1
            self._samples += [self._sample]
            self._sample = []
        if len(self._samples) > self.maxMemory:
            self._samples = self._samples[-self.maxMemory:]
    
    def correct_sample_buffer(self):
        sample_to_correct = self._sample_buffer[0]
        turn = sample_to_correct[0][-2]
        for _sample in self._sample_buffer:
            if _sample[3] is not None:
                if _sample[3][-2] == turn:
                    sample_to_correct[-2:] = _sample[-2:]
                    self._sample = sample_to_correct
                    self._sample_buffer = self._sample_buffer[1:]
                    break
        if sample_to_correct[3][-2] != turn:
            self._sample = []

    def finalize_sample_buffer(self):
        for _sample in self._sample_buffer:
            _sample[3] = None
            _sample[4] = None
            if abs(self._reward) > abs(_sample[2]):
                if (_sample[2] > 0) == (self._reward > 0):
                    _sample[2] = self._reward * -1
                else:
                    _sample[2] = self._reward
            self._sample = _sample
            self.add_sample()
        self._sample_buffer = []

    def learn_by_replay(self, batch_size):
        # Only learn once memory has reached batch size and not in test mode
        if (self._test_mode) | (len(self._samples) <= batch_size):
            return 0

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

        # Set up reward array
        z = np.zeros((len(batch), self.numActions))

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

                prediction_next_state = np.amax(q_s_a_d[i][options])

                corrected_q[action] = reward + self._discountFactor * prediction_next_state

            x[i] = state
            y[i] = corrected_q
            z[i] = reward

        # Add q to tensorboard
        with self._summary_writer.as_default():
            tfbare.summary.scalar('Q', np.mean(y), step=self._learningSteps)
            tfbare.summary.scalar('abs-Q-end-state', np.mean(np.abs(y[np.abs(z) > 50])), step=self._learningSteps)
            tfbare.summary.scalar('abs-Q-non-end-state', np.mean(np.abs(y[np.abs(z) < 50])), step=self._learningSteps)
            self._learningSteps += 1

        self.train_batch(x, y)
        self.update_epsilon()

    def update_reward_store(self):
        self._reward_store.append(float(self._tot_reward))
        
    def update_epsilon(self):
        self._eps = self.minEpsilon + (self.maxEpsilon - self.minEpsilon) * math.exp(-self._bootstrapValueEpsilon * self._steps)
        self._steps += 1

        # Add losses to tensorboard
        with self._summary_writer.as_default():
            tfbare.summary.scalar('Epsilon', self._eps, step = self._steps)

    def add_rewards_to_tensorboard(self, turn_count):
        with self._summary_writer.as_default():
            tfbare.summary.scalar('Rewards', float(self._tot_reward),
                                  step=self._summary_reward_step)
            tfbare.summary.scalar('TurnCount', turn_count,
                                  step=self._summary_reward_step)
            self._summary_reward_step += 1

    def new_game(self):
        self._tot_reward = 0

    def reload(self):
        self.define_model()
        checkpoints = os.listdir(self._checkpoint_path)
        checkpoints.sort()
        self.load_checkpoint(self._checkpoint_path + checkpoints[-1])
        self._summary_writer = tfbare.summary.create_file_writer(self._log_path)

    def new_part(self, current_part, new_part):
        self._state_path = self._state_path.replace(current_part, new_part)
        self._checkpoint_path = self._checkpoint_path.replace(current_part, new_part)


# Player
class RandomPlayer():

    def __init__(self, name, test_mode=True):

        # Identifying variables
        self._name = name
        self.isRandom = True
        self.isStill = False

        # Collection variables
        self._reward_store = []

        # State variables
        self._reward = 0
        self._tot_reward = 0

        # Is the player learning? Of temporarily paused due to test mode?
        self._test_mode = test_mode

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

    def set_state(self, game_x_list, game_y_list, turn, is_tagger):
        pass

    def update_reward_store(self):
        self._reward_store.append(float(self._tot_reward))

    def add_rewards_to_tensorboard(self, turn_count):
        pass

    def update_epsilon(self):
        pass

    def set_sample(self, choice, reward):
        pass

    def update_sample(self, options):
        pass

    def finalize_sample_buffer(self):
        pass

    def add_sample(self):
        pass

    def learn_by_replay(self, batch_size):
        pass

    def reload(self):
        pass


# Player
class StillPlayer():

    def __init__(self, name, test_mode=True):

        # Identifying variables
        self._name = name
        self.isRandom = True
        self.isStill = True

        # Collection variables
        self._reward_store = []

        # State variables
        self._reward = 0
        self._tot_reward = 0

        # Is the player learning? Of temporarily paused due to test mode?
        self._test_mode = test_mode

    def choose_action(self, options, save_game):
        return 8

    def get_name(self):
        return self._name
    name = property(get_name)

    def get_reward_store(self):
        return self._reward_store
    reward_store = property(get_reward_store)

    def new_game(self):
        self._tot_reward = 0

    def set_state(self, game_x_list, game_y_list, turn, is_tagger):
        pass

    def update_reward_store(self):
        self._reward_store.append(float(self._tot_reward))

    def add_rewards_to_tensorboard(self, turn_count):
        pass

    def update_epsilon(self):
        pass

    def set_sample(self, choice, reward):
        pass

    def update_sample(self, options):
        pass

    def finalize_sample_buffer(self):
        pass

    def add_sample(self):
        pass

    def learn_by_replay(self, batch_size):
        pass

    def reload(self):
        pass
