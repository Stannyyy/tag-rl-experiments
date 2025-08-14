# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:14:00 2021

@author: StannyGoffin
"""

# Import packages
import random
import numpy as np
import math
from model import Model, ModelNextState
import tensorflow as tfbare
import os

# Player
class Player(Model, ModelNextState):

    def __init__(self, experiment, name, bootstrapValueEpsilon = 0.001, discountFactor = 0.95,
                 learningRate = 0.001, layers = [100,100,100], addLSTM = False, sequenceLength = 1,
                 render=False, justLike=None, testMode=False, curiosity=False, curiosity_beta=0,
                 maxEpsilon=None):

        # Import models
        Model.__init__(self, experiment=experiment, learningRate=learningRate, layers=layers, addLSTM=addLSTM)
        if curiosity:
            ModelNextState.__init__(self)

        # Identifying variables
        self._name = name if justLike is None else justLike._name + name
        self.isRandom = False
        self.isStill = False

        # Model variables
        self._eps = self.maxEpsilon if justLike is None else justLike._eps
        self._bootstrapValueEpsilon = bootstrapValueEpsilon  # formerly lambda
        self._discountFactor = discountFactor  # formerly gamma
        self._sequence_length = sequenceLength

        # Experience variables (carrying over using justLike)
        self._step = 0 if justLike is None else justLike._step
        self._samples = [] if justLike is None else justLike._samples.copy()
        self._samples_count = 0
        self._sample_buffer = []

        # Curiosity variables
        self._curiosity = curiosity
        self._curiosity_beta = curiosity_beta
        if maxEpsilon is not None:
            self.maxEpsilon = maxEpsilon

        # Render variables
        self._render = render

        # Sample variables
        self._state = np.array([])
        self._sample = []

        # Collection variables
        self._reward_store_tagger = []
        self._reward_store_runner = []

        # State variables
        self._reward = 0
        self._tot_reward_tagger = 0
        self._tot_reward_runner = 0

        # Is the player learning? Of temporarily paused due to test mode?
        self._test_mode = testMode

        # Save intermittant folders
        self._state_path = os.getcwd() + experiment + "/state/part1-" + self._name.replace(" ","") + ".pickle"
        self._checkpoint_path = os.getcwd() + experiment + "/checkpoints/" + self._name + "/part1/"
        self._log_path = os.getcwd() + experiment + "/logs/dql_" + self._name + "/"

        # Set up the tensorboard
        self._summary_writer = tfbare.summary.create_file_writer(self._log_path)
        self._summary_writer_collection = []

    def choose_action(self, options, save_game, game=None):

        # Use chance to see whether to explore or exploit
        chance_value = random.random()
        if (chance_value < self._eps) and (save_game == False):
            choice = random.sample(options, k=1)[0]
        else:
            if self._add_LSTM:
                ix_sequence_start = self._sequence_length*-1+1
                if ix_sequence_start == 0:
                    last_x_minus_1_samples = []
                else:
                    last_x_minus_1_samples = self._samples[(self._sequence_length * -1 + 1):]
                prediction = self.predict_one([[s[0] for s in last_x_minus_1_samples] + [self._state]])
            else:
                prediction = self.predict_one([self._state])
            prediction = [p if i in options else -np.inf for i, p in enumerate(prediction)]
            choice = np.argmax(prediction)

        return choice

    def summary_writer(self):
        with self._summary_writer.as_default():
            for s in self._summary_writer_collection:
                tfbare.summary.scalar(s.get('name'), s.get('value'), step=s.get('step'))
        self._summary_writer_collection = []

    def get_name(self):
        return self._name
    name = property(get_name)

    def get_model(self):
        return self._model
    model = property(get_model)

    def get_model_next_state(self):
        return self._model_next_state
    model_next_state = property(get_model_next_state)

    def get_losses(self):
        return self._losses
    losses = property(get_losses)

    def get_reward_store_tagger(self):
        return self._reward_store_tagger
    reward_store_tagger = property(get_reward_store_tagger)

    def get_reward_store_runner(self):
        return self._reward_store_runner
    reward_store_runner = property(get_reward_store_runner)

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

    def set_state(self, game_x_list, game_y_list, turn, is_tagger, turn_count=None):
        self._state = game_x_list + game_y_list + [turn, int(is_tagger)]
    state = property(set_state)

    def get_step(self):
        return self._step
    step = property(get_step)

    def set_step(self, step):
        self._step = step
    step = property(get_step, set_step)

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

        # Make random batch, but always include some end states
        batch = self.create_batch()

        # Predict Q(s,a) given the batch of states
        states = np.array([val[0] for seq in batch for val in seq])
        if self._add_LSTM:
            states = states.reshape((self.batchSize, self._sequence_length, self.numStates))
        q_s_a = self.predict_batch(states)
        q_s_a_uncorrected = np.copy(q_s_a)

        # Predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        next_states = np.array([(np.zeros(self.numStates) if val[3] is None else val[3]) for seq in batch for val in seq])
        if self._add_LSTM:
            next_states = next_states.reshape((self.batchSize, self._sequence_length, self.numStates))
        q_s_a_d = self.predict_batch(next_states)

        # Extract slices from batch
        all_states = np.array([b[0][0] for b in batch])*1.0
        all_next_states = np.array([None if b[0][0] is None else np.array(b[0][0]).astype(float) for b in batch])
        if self._add_LSTM:
            all_states = all_states.reshape((self.batchSize, self._sequence_length, self.numStates))
            all_next_states = all_next_states.reshape((self.batchSize, self._sequence_length, self.numStates))
        all_rewards = np.array([b[0][2] for b in batch])*1.0

        # Set up training arrays
        corrected_qs = np.zeros((self.batchSize, self.numActions))

        # Bulk predict next state
        if self._curiosity:
            predicted_next_state = self.predict_batch_next_state(all_states)

        # Now loop over batch
        for i, b in enumerate(batch):

            # Extract sample
            state, action, reward, next_state, options = b[-1][0], b[-1][1], b[-1][2], b[-1][3], b[-1][4]

            # Get the corrected q values for all actions in state
            corrected_q = q_s_a[i]

            # Clip corrected_q
            corrected_q = [max(min(c*1.0, self.tagPoints*1.0), self.tagPoints*-1.0) for c in corrected_q]

            # Update the q value for action
            if next_state is None:
                corrected_q[action] = reward
            else:

                # Curiosity bonus
                if self._curiosity:
                    curiosity_bonus = np.mean((predicted_next_state[i] - next_state) ** 2)
                    reward += self._curiosity_beta * curiosity_bonus

                prediction_next_state = np.amax(q_s_a_d[i][options])
                corrected_q[action] = reward + self._discountFactor * prediction_next_state

            corrected_qs[i] = corrected_q

        summary_writer_collection_add = self.train_batch(all_states, corrected_qs, self._step)
        self._summary_writer_collection += [summary_writer_collection_add]
        if self._curiosity:
            summary_writer_collection_add = self.train_batch_next_state(all_states, all_next_states, self._step)
            self._summary_writer_collection += [summary_writer_collection_add]
        self.update_epsilon()

        # Add q to tensorboard
        end_state = np.abs(all_rewards) >= (self.tagPoints - self.stepPoints*2)
        self._summary_writer_collection += [
            {"name": 'Q/overall',
             "value": np.round(np.mean(np.abs(q_s_a_uncorrected)), 1),
             "step": self._step}
        ]
        if np.sum(end_state) > 0:
            uncorrected_end_qs = q_s_a_uncorrected[end_state]
            corrected_end_qs = corrected_qs[end_state]
            crucial_action = np.abs(corrected_end_qs) >= (self.tagPoints - self.stepPoints*2)
            q_crucial_action = uncorrected_end_qs[crucial_action]
            q_alternative_action = uncorrected_end_qs[crucial_action==False]
            self._summary_writer_collection += [
                {
                    "name": 'Q/tagged-state-of-crucial-action', 
                    "value": np.round(np.mean(np.abs(q_crucial_action)), 1),
                    "step": self._step
                 },
                {
                    "name": 'Q/tagged-state-of-alternative-action',
                    "value": np.round(np.mean(np.abs(q_alternative_action)), 1),
                    "step": self._step
                },
                {
                    "name": 'Q/diff-rel',
                    "value": np.round(np.mean(np.abs(q_crucial_action)) / np.mean(np.abs(q_alternative_action)), 1),
                    "step": self._step
                },
                {
                    "name": 'Q/diff-abs',
                    "value": np.round(np.mean(np.abs(q_crucial_action)) - np.mean(np.abs(q_alternative_action)), 1),
                    "step": self._step
                },
                {
                    "name": 'Q/tagged-state-of-crucial-action-norm',
                    "value": np.round(np.mean(np.abs(q_crucial_action)) / np.mean(np.abs(q_s_a_uncorrected)), 1),
                    "step": self._step
                },
                {
                    "name": 'Q/tagged-state-of-alternative-action-norm',
                    "value": np.round(np.mean(np.abs(q_alternative_action)) / np.mean(np.abs(q_s_a_uncorrected)), 1),
                    "step": self._step
                },
            ]

    def create_batch(self):
        selection = random.choices(range(len(self._samples)-self._sequence_length), k=self.batchSize)

        if self._add_LSTM:
            batch = []
            while len(batch) < self.batchSize:
                for i in selection:
                    samples_i = []
                    add_i = 0
                    compare_4 = self._samples[i][0][4]
                    compare_5 = self._samples[i][0][5]
                    while len(samples_i) < self._sequence_length:
                        if (compare_4 == self._samples[i+add_i][0][4]) & (compare_5 == self._samples[i+add_i][0][5]):
                            if i+add_i >= (len(self._samples)-1):
                                break
                            if self._samples[i+add_i][4] is None:
                                if len(samples_i) != (self._sequence_length - 1):
                                    break
                            samples_i += [self._samples[i+add_i]]
                        add_i +=1
                    if len(samples_i) == self._sequence_length:
                        batch += [samples_i]
                    else:
                        continue
                selection = random.choices(range(len(self._samples) - self._sequence_length), k=self.batchSize-len(batch))
        else:
            batch = [[self._samples[i]] for i in selection]
        return batch

    def show_q_in_state(self, game):
        if self._add_LSTM:
            ix_sequence_start = self._sequence_length * -1 + 1
            if ix_sequence_start == 0:
                last_x_minus_1_samples = []
            else:
                last_x_minus_1_samples = self._samples[(self._sequence_length * -1 + 1):]
            prediction = self.predict_one([[np.array(s[0]) for s in last_x_minus_1_samples] + [self._state]])
        else:
            prediction = self.predict_one([self._state])

        prediction = [np.round(p,1) for p in prediction]
        game.render(prediction, self._state)
        print('---')
        print(self._state)
        print('---')
        for row in game._rendered:
            print('|'.join(row))
        print('---')

    def update_reward_store(self):
        if self._tot_reward_tagger != 0:
            self._reward_store_tagger.append(float(self._tot_reward_tagger))
        if self._tot_reward_runner != 0:
            self._reward_store_runner.append(float(self._tot_reward_runner))
        
    def update_epsilon(self):
        # Add epsilon to tensorboard
        self._summary_writer_collection += [
            {
                "name": 'params/epsilon',
                "value": np.round(self._eps, 3),
                "step": self._step
             }
        ]

        # Update epsilon
        self._eps = self.minEpsilon + (self.maxEpsilon - self.minEpsilon) * math.exp(-self._bootstrapValueEpsilon * self._step)

    def add_rewards_to_tensorboard(self, turn_count):
        self._summary_writer_collection += [
            {
                "name": 'Rewards/turn_count',
                "value": turn_count,
                "step": self._step
             }
        ]
        if self._tot_reward_tagger != 0:
            self._summary_writer_collection += [
                {
                    "name": 'Rewards/tagger',
                    "value": float(self._tot_reward_tagger),
                    "step": self._step
                }
            ]
        if self._tot_reward_runner != 0:
            self._summary_writer_collection += [
                {
                    "name": 'Rewards/runner',
                    "value": float(self._tot_reward_runner),
                    "step": self._step
                }
            ]

    def new_game(self):
        self._tot_reward_tagger = 0
        self._tot_reward_runner = 0

    def reload(self):
        if self._add_LSTM:
            input_shape = (None, self._sequence_length, self.numStates)
        else:
            input_shape = (None, self.numStates)

        self.define_model()
        self.model.build(input_shape=input_shape)
        checkpoints = [p for p in os.listdir(self._checkpoint_path) if '-next-state' not in p]
        checkpoints.sort()
        self.load_checkpoint(self._checkpoint_path + checkpoints[-1])

        if self._curiosity:
            self.define_model_next_state()
            self.model_next_state.build(input_shape=input_shape)
            checkpoints_next_state = [p for p in os.listdir(self._checkpoint_path) if '-next-state' in p]
            checkpoints_next_state.sort()
            self.load_checkpoint_next_state(self._checkpoint_path + checkpoints_next_state[-1])
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
        self._step = 0
        self._reward_store_tagger = []
        self._reward_store_runner = []

        # State variables
        self._reward = 0
        self._tot_reward_tagger = 0
        self._tot_reward_runner = 0

        # Is the player learning? Of temporarily paused due to test mode?
        self._test_mode = test_mode

    def choose_action(self, options, save_game, game=None):
        return random.sample(options, k=1)[0]

    def get_name(self):
        return self._name
    name = property(get_name)

    def get_reward_store_tagger(self):
        return self._reward_store_tagger
    reward_store_tagger = property(get_reward_store_tagger)

    def get_reward_store_runner(self):
        return self._reward_store_runner
    reward_store_runner = property(get_reward_store_runner)

    def new_game(self):
        self._tot_reward_tagger = 0
        self._tot_reward_runner = 0

    def set_state(self, game_x_list, game_y_list, turn, is_tagger, turn_count=None):
        pass

    def update_reward_store(self):
        self._reward_store_tagger.append(float(self._tot_reward_tagger))
        self._reward_store_runner.append(float(self._tot_reward_runner))

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

    def get_step(self):
        return self._step
    step = property(get_step)

    def set_step(self, step):
        self._step = step
    step = property(get_step, set_step)

# Player
class StillPlayer():

    def __init__(self, name, test_mode=True):

        # Identifying variables
        self._name = name
        self.isRandom = True
        self.isStill = True

        # Collection variables
        self._reward_store_tagger = []
        self._reward_store_runner = []

        # State variables
        self._reward = 0
        self._step = 0
        self._tot_reward_tagger = 0
        self._tot_reward_runner = 0

        # Is the player learning? Of temporarily paused due to test mode?
        self._test_mode = test_mode

    def choose_action(self, options, save_game, game=None):
        return 8

    def get_name(self):
        return self._name
    name = property(get_name)

    def get_reward_store_tagger(self):
        return self._reward_store_tagger
    reward_store_tagger = property(get_reward_store_tagger)

    def get_reward_store_runner(self):
        return self._reward_store_runner
    reward_store_runner = property(get_reward_store_runner)

    def new_game(self):
        self._tot_reward_tagger = 0
        self._tot_reward_runner = 0

    def set_state(self, game_x_list, game_y_list, turn, is_tagger, turn_count=None):
        pass

    def update_reward_store(self):
        self._reward_store_tagger.append(float(self._tot_reward_tagger))
        self._reward_store_runner.append(float(self._tot_reward_runner))

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

    def get_step(self):
        return self._step
    step = property(get_step)

    def set_step(self, step):
        self._step = step
    step = property(get_step, set_step)
