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
from memory import Memory
import tensorflow as tf
import copy
import os
from config import Config

# Player
class Player(Config, Model, ModelNextState, Memory):

    def __init__(self, experiment, name, justLike=None, **kwargs):

        # Import model, memory and config
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)
        if self.curiosity:
            ModelNextState.__init__(self, experiment=experiment,
                                    add_lstm=self.add_lstm, sequence_length_lstm=self.sequence_length_lstm)

        # Identifying variables
        self.name = name if justLike is None else justLike.name + name
        self.is_random = False
        self.is_still = False

        # Model variables
        self.eps = self.max_epsilon if justLike is None else justLike.eps
        self.cnt = 0

        # Memory
        self.memory = Memory(self.max_memory)

        # State variables
        self._state = np.array([])
        self._state_many = []

        # Collection variables
        self._reward_store_tagger = []
        self._reward_store_runner = []

        # State variables
        self._reward = 0
        self._tot_reward_tagger = 0
        self._tot_reward_runner = 0
        self.options = []

        # Save intermittent folders
        self._state_path = os.path.join(os.getcwd(), experiment, "state",
                                        "part1-" + self.name.replace(" ", "") + ".pickle")
        self._checkpoint_path = os.path.join(os.getcwd(), experiment, "checkpoints", self.name, "part1")
        self._log_path = os.path.join(os.getcwd(), experiment, "logs", "dql_" + self.name)

        # Set up the tensorboard
        self._summary_writer = tf.summary.create_file_writer(self._log_path)
        self._summary_writer_collection = []
        self._tboard_callback_command = 'self._tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), "'+experiment+'", "logs"), profile_batch= 1)'
        exec(self._tboard_callback_command)


    def prediction_to_probabilities(self, prediction, options):
        """
        Convert q predictions to probabilities that sum to 1 over finite entries.
        -inf entries get probability 0.
        temperature > 1.0 => flatter distribution
        temperature < 1.0 => sharper distribution

        It is basically normalization then softmax with temperature.
        """

        temperature = self.eps * 100
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        # Filter infinite values (not an option)
        probs = np.asarray(prediction, dtype=float)
        mask = np.where(options)[0]

        # Normalize
        probs[mask] -= np.mean(probs[mask])
        if sum(mask) == 1:
            probs[mask] = 1
        elif sum(mask) == 0:
            raise ValueError("no finite values")
        else:
            probs[mask] /= np.std(probs[mask])

        # Stable softmax on finite entries
        probs[mask] = np.exp(probs[mask] / temperature)
        probs[mask] = probs[mask] / np.sum(probs[mask])

        # Zero on infinite entries
        probs[~mask] = 0.0
        return probs

    def prediction_to_probabilities_many(self, predictions, options):
        """
        Convert q predictions to probabilities that sum to 1 over finite entries.
        -inf entries get probability 0.
        temperature > 1.0 => flatter distribution
        temperature < 1.0 => sharper distribution

        It is basically normalization then softmax with temperature.
        """

        temperature = self.eps * 100
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        # Normalize
        probs = predictions.astype(float, copy=True)
        probs *= options
        row_sum = (probs * options.astype(float)).sum(axis=1, keepdims=True)
        row_mean = row_sum / options.sum(axis=1, keepdims=True)

        predictions -= row_mean
        # probs[mask] /= np.std(probs[mask])

        # Stable softmax on finite entries
        predictions = np.exp(predictions / temperature) * options
        predictions = predictions / np.sum(predictions * options, axis=1, keepdims=True)

        return predictions

    def choose_action(self, options, save_game):

        """
        Choose action: random based on chance value epsilon OR based on current policy for the given state
        """

        # Use chance to see whether to explore or exploit
        chance_value = random.random()
        if (chance_value < self.eps) and (save_game == False):
            choice = random.sample(options, k=1)[0]
            return choice
        else:
            if self._add_LSTM:
                ix_sequence_start = self._model._sequence_length_LSTM * -1 + 1
                if ix_sequence_start == 0:
                    last_x_minus_1_samples = []
                else:
                    last_x_minus_1_samples = self.memory._samples[(self._model._sequence_length_LSTM * -1 + 1):]
                prediction = self.predict_one(np.array([[s[0] for s in last_x_minus_1_samples] + [self._state]]))
            else:
                prediction = self.predict_one(np.array([self._state]))

        if self.use_probabilities and (save_game == False):
            probabilities = self.prediction_to_probabilities(prediction, options)
            choice = int(np.random.choice(self.num_actions, p=probabilities))
        else:
            prediction[~options] = -np.inf
            choice = int(np.argmax(prediction))

        return choice

    def options_to_elligible_idx(self, options):
        # Vectorized parsing of options to be able to choose an action randomly
        idx_options = options.nonzero()[1]
        option_counts = options.sum(axis=1)
        starts = np.cumsum(np.r_[0, option_counts[:-1]])

        return option_counts, idx_options, starts

    def choose_many_actions(self, options, slct):

        """
        Choose many action: random based on chance value epsilon OR based on current policy for the given state
        for all games in the batch in parallel
        """

        # Check requirements
        if self._add_LSTM:
            raise Exception("LSTM option is not yet suitable to use with parallel mode")

        # Use chance to see whether to explore or exploit
        chance_value = random.random()
        if chance_value < self.eps:

            # Dissect options
            option_counts, idx_options, starts = self.options_to_elligible_idx(options)

            # Random choice
            rng = np.random.default_rng()
            r = rng.integers(0, option_counts)[slct]

            # Vector of all choices
            picked_cols = idx_options[starts[slct] + r]
            all_choices = np.full(options.shape[0], 8, dtype=int)
            all_choices[slct] = picked_cols
            return all_choices
        else:
            if len(slct) > 1:
                predictions = self.predict_batch(np.array([state for ix, state in enumerate(self._state_many) if ix in slct]))
            elif len(slct) == 1:
                predictions = self.predict_batch(np.array([[self._state_many[slct[0]]]]))
                predictions = np.expand_dims(predictions, axis=0)
            else:
                predictions = []

            # Filter options
            predictions = predictions.astype(float, copy=False)

        if self.use_probabilities:
            probabilities = self.prediction_to_probabilities_many(predictions, options[slct])
            choices = np.array([np.random.choice(self.num_actions, p=row) for row in probabilities])

        else:
            predictions[~options] = -np.inf
            choices = np.argmax(predictions)

        all_choices = []
        for ep in range(self.num_episodes_per_round):
            if ep in slct:
                all_choices.append(choices[slct.index(ep)])
            else:
                all_choices.append(8)

        return all_choices

    def write_summary_to_tensorboard(self):

        """
        Write collection of logs to tensorboard
        """

        with self._summary_writer.as_default():
            for s in self._summary_writer_collection:
                tf.summary.scalar(s.get('name'), s.get('value'), step=s.get('step'))
        self._summary_writer_collection = []

    def getname(self):
        return self.name

    name = property(getname)

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
        return self.eps

    eps = property(get_eps)

    def set_eps(self, eps):
        self.eps = eps

    eps = property(get_eps, set_eps)

    def get_state(self):
        return self._state

    state = property(get_state)

    def set_state(self, values):
        game_x_list, game_y_list, turn, is_tagger = values
        self._state = game_x_list + game_y_list + [turn, int(is_tagger)]

    state = property(get_state, set_state)

    def get_state_many(self):
        return self._state_many

    state_many = property(get_state_many)

    def set_state_many(self, values):
        self._state_many = []
        for state in values:
            game_x_list, game_y_list, turn, is_tagger = state
            self._state_many += [game_x_list + game_y_list + [turn, int(is_tagger)]]

    state_many = property(get_state_many, set_state_many)

    def get_options(self):
        return self.options

    options = property(get_options)

    def set_options(self, options):
        self.options = options

    options = property(get_options, set_options)

    def learn_by_replay(self, batch_size = None, epochs = 1, verbose = False):

        """
        Learn by replay! The model gets trained by using the sample buffer. You take a random batch of the memory,
        shuffle them and do a training round using the deep Q (reinforcement) learning protocol.
        Save the logs for the tensorboard
        """

        # If batch_size undefined, fill with config batch size
        if batch_size is None:
            if self.preselect_batch:
                batch_size = int(self.batch_size * 4)
            else:
                batch_size = int(self.batch_size)

        # Only learn once memory has reached batch size and not in test mode
        if self.test_mode or (len(self.memory._samples) < batch_size):
            return 0

        # Make a random batch
        print("\rCreating batch")
        batch = self.create_batch(batch_size = batch_size)
        if not batch:
            return 0

        # Extract samples
        print("\rDissecting batch")
        states = np.array([b[-1][0] for b in batch], dtype=float)
        actions = np.array([b[-1][1] for b in batch], dtype=int)
        rewards = np.array([b[-1][2] for b in batch], dtype=float)
        next_states = np.array([(np.zeros(self.num_states)-1 if val[-2] is None else val[-2]) for seq in batch for val in seq])
        options = [b[-1][4] for b in batch]

        # Define end states
        is_terminal = (next_states == -1).all(axis=1)  # vectorized comparison for object array
        is_nonterminal = ~is_terminal

        # Predict Q(s,a) given the batch of states
        print("\rPredicting q's for all states")
        q_s_a = self.predict_batch(states)

        # Predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        print("\rPredicting q's for all next states")
        q_s_a_d = self.predict_batch(next_states)

        # Bulk predict next state
        if self.curiosity:
            print("\rPredicting next states for all states")
            predicted_next_state = self.predict_batch_next_state(states)

        # Clip corrected q
        print("\rClipping predicted qs")
        corrected_qs = np.clip(q_s_a, -self.tag_points, self.tag_points)

        # Add curiosity bonus
        if self.curiosity:
            print("\rAdding curiosity bonus")
            # Mean-squared error across state dims per row
            mse = ((predicted_next_state - np.vstack(next_states[is_nonterminal])) ** 2).mean(axis=1)
            curiosity_bonus = np.zeros(batch_size, dtype=float)
            curiosity_bonus[is_nonterminal] = self.curiosity_beta * mse
            rewards += self.curiosity_beta * curiosity_bonus

        # Non-terminal states: replace with reward+y*maxQ(s',a')-V(s, a)
        print("\rCorrecting q's of chosen actions of non-terminal states")
        q_next_states = copy.deepcopy(q_s_a_d)
        v_current_state = np.sum(q_next_states*np.array(options), axis = 1)/np.sum(options, axis = 1)
        q_next_states[~np.array(options)] = -np.inf
        q_next_states = q_next_states.max(axis=1)
        corrected_qs[np.arange(batch_size),actions] = rewards + self.discount_factor * q_next_states - v_current_state

        # Overwrite terminal states: replace with reward
        print("\rCorrecting q's of chosen actions of terminal states")
        corrected_qs[is_terminal, actions[is_terminal]] = rewards[is_terminal]

        # Filter batch
        if self.preselect_batch:
            print("\rFiltering batch")
            correction_diff = np.round(np.sum(np.abs(corrected_qs - q_s_a), axis=1),1)
            idx_selection = np.argsort(correction_diff)[::-1][:int(batch_size/4)]
            all_states_selection = states[idx_selection]
            corrected_qs_selection = corrected_qs[idx_selection]
        else:
            all_states_selection = states
            corrected_qs_selection = corrected_qs

        # Train batch
        print("\rNow training can start")
        summary_writer_collection_add = self.train_batch(all_states_selection, corrected_qs_selection, self.cnt,
                                                         self._log_path, epochs=epochs, verbose=verbose)
        self._summary_writer_collection += [summary_writer_collection_add]
        if self.curiosity:
            # Train batch next state
            summary_writer_collection_add = self.train_batch_next_state(states, next_states, self.cnt)
            self._summary_writer_collection += [summary_writer_collection_add]
        self.update_epsilon()

        # Add q to tensorboard
        end_state = np.abs(rewards) >= (self.tag_points - self.step_points * 2)
        self._summary_writer_collection += [
            {"name": 'Q/overall',
             "value": np.round(np.mean(np.abs(q_s_a)), 1),
             "step": self.cnt}
        ]
        if np.sum(end_state) > 0:
            uncorrected_end_qs = q_s_a[end_state]
            corrected_end_qs = corrected_qs[end_state]
            crucial_action = np.abs(corrected_end_qs) >= (self.tag_points - self.step_points * 2)
            q_crucial_action = uncorrected_end_qs[crucial_action]
            q_alternative_action = uncorrected_end_qs[crucial_action == False]
            self._summary_writer_collection += [
                {
                    "name": 'Q/tagged-state-of-crucial-action',
                    "value": np.round(np.mean(np.abs(q_crucial_action)), 1),
                    "step": self.cnt
                },
                {
                    "name": 'Q/tagged-state-of-alternative-action',
                    "value": np.round(np.mean(np.abs(q_alternative_action)), 1),
                    "step": self.cnt
                },
                {
                    "name": 'Q/diff-rel',
                    "value": np.round(np.mean(np.abs(q_crucial_action)) / np.mean(np.abs(q_alternative_action)), 1),
                    "step": self.cnt
                },
                {
                    "name": 'Q/diff-abs',
                    "value": np.round(np.mean(np.abs(q_crucial_action)) - np.mean(np.abs(q_alternative_action)), 1),
                    "step": self.cnt
                },
                {
                    "name": 'Q/tagged-state-of-crucial-action-norm',
                    "value": np.round(np.mean(np.abs(q_crucial_action)) / np.mean(np.abs(q_s_a)), 1),
                    "step": self.cnt
                },
                {
                    "name": 'Q/tagged-state-of-alternative-action-norm',
                    "value": np.round(np.mean(np.abs(q_alternative_action)) / np.mean(np.abs(q_s_a)), 1),
                    "step": self.cnt
                },
            ]
        return 1

    def create_batch(self, batch_size=None):

        # If batch_size undefined, fill with config batch size
        if batch_size is None:
            batch_size = int(self.batch_size)

        # Guard: not enough samples to form a sequence
        if self._add_LSTM:
            selection_pool_size = len(self.memory._samples) - self._model._sequence_length_LSTM
        else:
            selection_pool_size = len(self.memory._samples)

        if selection_pool_size < batch_size:
            return []

        selection = random.choices(range(selection_pool_size), k=batch_size)

        if self._add_LSTM:
            batch = []
            while len(batch) < batch_size:
                for i in selection:
                    samples_i = []
                    add_i = 0
                    compare_4 = self._samples[i][0][4]
                    compare_5 = self._samples[i][0][5]
                    while len(samples_i) < self._model._sequence_length_LSTM:
                        if (compare_4 == self._samples[i + add_i][0][4]) and (
                                compare_5 == self._samples[i + add_i][0][5]):
                            if i + add_i >= (len(self._samples) - 1):
                                break
                            if self._samples[i + add_i][4] is None:
                                if len(samples_i) != (self._model._sequence_length_LSTM - 1):
                                    break
                            samples_i += [self._samples[i + add_i]]
                        add_i += 1
                    if len(samples_i) == self._model._sequence_length_LSTM:
                        batch += [samples_i]
                    else:
                        continue
                remaining = batch_size - len(batch)
                if remaining <= 0:
                    break
                selection = random.choices(range(selection_pool_size), k=remaining)
        else:
            batch = [[self.memory._samples[i]] for i in selection]
        return batch

    def show_q_in_state(self, game):
        if self._add_LSTM:
            ix_sequence_start = self._model._sequence_length_LSTM * -1 + 1
            if ix_sequence_start == 0:
                last_x_minus_1_samples = []
            else:
                last_x_minus_1_samples = self._samples[(self._model._sequence_length_LSTM * -1 + 1):]
            prediction = self.predict_one([[np.array(s[0]) for s in last_x_minus_1_samples] + [self._state]])
        else:
            prediction = self.predict_one([self._state])

        prediction = [np.round(p, 1) for p in prediction]
        game.render(prediction, self._state)
        print('---')
        print(self._state)
        print('---')
        for row in game.rendered:
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
                "value": np.round(self.eps, 3),
                "step": self.cnt
            }
        ]

        # Update epsilon
        self.eps = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(
            -self.bootstrap_value_epsilon * self.cnt)

    def add_rewards_to_tensorboard(self, turn_count):
        self._summary_writer_collection += [
            {
                "name": 'Rewards/turn_count',
                "value": turn_count,
                "step": self.cnt
            }
        ]
        if self._tot_reward_tagger != 0:
            self._summary_writer_collection += [
                {
                    "name": 'Rewards/tagger',
                    "value": float(self._tot_reward_tagger),
                    "step": self.cnt
                }
            ]
        if self._tot_reward_runner != 0:
            self._summary_writer_collection += [
                {
                    "name": 'Rewards/runner',
                    "value": float(self._tot_reward_runner),
                    "step": self.cnt
                }
            ]

    def new_game(self):
        self._tot_reward_tagger = 0
        self._tot_reward_runner = 0

    def reload(self):
        if self._add_LSTM:
            input_shape = (None, self._model._sequence_length_LSTM, self.num_states)
        else:
            input_shape = (None, self.num_states)

        self.define_model()
        self.model.build(input_shape=input_shape)

        checkpoints = [p for p in os.listdir(self._checkpoint_path) if
                       '-next-state' not in p and p.endswith('.weights.h5')]
        checkpoints.sort()
        if checkpoints:
            self.load_checkpoint(os.path.join(self._checkpoint_path, checkpoints[-1]))

        if self.curiosity:
            self.define_model_next_state()
            self.model_next_state.build(input_shape=input_shape)
            checkpoints_next_state = [p for p in os.listdir(self._checkpoint_path) if
                                      '-next-state' in p and p.endswith('.weights.h5')]
            checkpoints_next_state.sort()
            if checkpoints_next_state:
                self.load_checkpoint_next_state(os.path.join(self._checkpoint_path, checkpoints_next_state[-1]))
        self._summary_writer = tf.summary.create_file_writer(self._log_path)
        exec(self._tboard_callback_command)

    def new_part(self, current_part, new_part):
        self._state_path = self._state_path.replace(current_part, new_part)
        self._checkpoint_path = self._checkpoint_path.replace(current_part, new_part)


# Player
class RandomPlayer():

    def __init__(self, name, test_mode=True):
        # Identifying variables
        self.name = name
        self.is_random = True
        self.is_still = False

        # Collection variables
        self._reward_store_tagger = []
        self._reward_store_runner = []

        # State variables
        self._reward = 0
        self._tot_reward_tagger = 0
        self._tot_reward_runner = 0

        # Is the player learning? Or temporarily paused due to test mode?
        self.test_mode = test_mode

    def choose_action(self, options, save_game, game=None):
        if not options:
            return 0
        return random.sample(options, k=1)[0]

    def getname(self):
        return self.name

    name = property(getname)

    def get_reward_store_tagger(self):
        return self._reward_store_tagger

    reward_store_tagger = property(get_reward_store_tagger)

    def get_reward_store_runner(self):
        return self._reward_store_runner

    reward_store_runner = property(get_reward_store_runner)

    def new_game(self):
        self._tot_reward_tagger = 0
        self._tot_reward_runner = 0

    def set_state(self, values):
        game_x_list, game_y_list, turn, is_tagger = values
        pass

    def set_state_many(self, values):
        pass

    def update_reward_store(self):
        self._reward_store_tagger.append(float(self._tot_reward_tagger))
        self._reward_store_runner.append(float(self._tot_reward_runner))

    def add_rewards_to_tensorboard(self, turn_count):
        pass

    def update_epsilon(self):
        pass

    def set_sample(self, values):
        pass

    def update_sample(self, options):
        pass

    def update_sample_many(self):
        pass

    def add_corrected_sample(self, tag_happened):
        pass

    def learn_by_replay(self, batch_size):
        pass

    def reload(self):
        pass


# Player
class StillPlayer():

    def __init__(self, name, test_mode=True):
        # Identifying variables
        self.name = name
        self.is_random = True
        self.is_still = True

        # Collection variables
        self._reward_store_tagger = []
        self._reward_store_runner = []

        # State variables
        self._reward = 0
        self._tot_reward_tagger = 0
        self._tot_reward_runner = 0

        # Is the player learning? Or temporarily paused due to test mode?
        self.test_mode = test_mode

    def choose_action(self, options, save_game, game=None):
        return 8

    def getname(self):
        return self.name

    name = property(getname)

    def get_reward_store_tagger(self):
        return self._reward_store_tagger

    reward_store_tagger = property(get_reward_store_tagger)

    def get_reward_store_runner(self):
        return self._reward_store_runner

    reward_store_runner = property(get_reward_store_runner)

    def new_game(self):
        self._tot_reward_tagger = 0
        self._tot_reward_runner = 0

    def set_state(self, values):
        game_x_list, game_y_list, turn, is_tagger = values
        pass

    def set_state_many(self, values):
        pass

    def update_reward_store(self):
        self._reward_store_tagger.append(float(self._tot_reward_tagger))
        self._reward_store_runner.append(float(self._tot_reward_runner))

    def add_rewards_to_tensorboard(self, turn_count):
        pass

    def update_epsilon(self):
        pass

    def set_sample(self, values):
        pass

    def update_sample(self, options):
        pass

    def update_sample_many(self):
        pass

    def add_corrected_sample(self, tag_happened):
        pass

    def learn_by_replay(self, batch_size):
        pass

    def reload(self):
        pass