# Import packages
import random
import numpy as np
import math
import tensorflow as tf
import copy
import os
from model import Model, ModelNextState
from memory import Memory

# Player
class Player:

    def __init__(self, config, path, name, model=None, model_next_state=None, memory=None, **kwargs):

        # Import config
        config = copy.deepcopy(config)
        for key, value in kwargs.items():
            setattr(config, key, value)

        self._config = config

        # Import models
        if model is None:
            self._model = Model(config)
        else:
            self._model = model

        if config.curiosity:
            if model_next_state is None:
                self._model_next_state = ModelNextState(config)
            else:
                self._model_next_state = model_next_state

        # Import memory
        if memory is None:
            self._memory = Memory(config)
        else:
            self._memory = memory

        # Initialize the linked arena, so the player can see the score
        self._arena = None

        # Identifying variables
        self._name = name
        self._is_random = False
        self._is_still = False

        # Model variables
        self._epsilon = config.maximum_epsilon

        # State variables
        self._state = np.array([])
        self._state_many = []

        # Collection variables
        self._reward_store_tagger = []
        self._reward_store_runner = []
        self._q_logs = []

        # State variables
        self._current_reward = 0
        self._total_reward_tagger = 0
        self._total_reward_runner = 0
        self._current_options = []

        # Save intermittent folders
        self._state_path = os.path.join(path, "state", f"{self._name.replace(' ', '')}.pickle")
        self._checkpoint_path = os.path.join(path, "checkpoints", self._name)
        self._log_path = os.path.join(path, "logs", f"dql_{self._name}")

        # Set up the tensorboard
        self._summary_writer = tf.summary.create_file_writer(self._log_path)
        self._summary_writer_collection = []
        # self._tboard_callback_command = f"self._tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(r'{path}', 'logs'), profile_batch= 1)"
        # exec(self._tboard_callback_command)

    def prediction_to_probabilities(self, predictions, options):
        """
        Convert q predictions to probabilities that sum to 1 over possible entries.
        Entries that are not in options get probability 0.
        temperature > 1.0 => flatter distribution
        temperature < 1.0 => sharper distribution

        Normalize beforehand

        It is basically softmax with temperature.
        """

        # Check if single or many
        input_single = False
        predictions = np.array(predictions)
        options = np.array(options)
        if predictions.ndim == 1:
            input_single = True
            predictions = np.expand_dims(predictions, axis=0)
            options = np.expand_dims(options, axis=0)

        # Take arrays with infinite values out of options
        options[np.isinf(predictions)] = False
        predictions[np.isinf(predictions)] = 0.0

        # Normalize (note you cannot do np.mean directly, as total n has to be corrected for options)
        predictions = predictions.astype(float, copy=True)
        row_sum = (predictions * options.astype(float)).sum(axis=1, keepdims=True)
        row_mean = row_sum / options.sum(axis=1, keepdims=True)
        predictions -= row_mean
        predictions *= options

        row_std = (predictions ** 2).sum(axis=1) / options.sum(axis=1)
        predictions[row_std!=0] /= np.expand_dims(np.sqrt(row_std[row_std!=0]), axis=1)

        # Determine temperature
        temperature = self._epsilon * 100
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        # Softmax with temperature
        probabilities = np.exp(predictions / temperature) * options
        probabilities = probabilities / np.sum(probabilities * options, axis=1, keepdims=True)

        return probabilities[0] if input_single else probabilities

    def state_to_prediction(self):
        if self._config.add_lstm:
            ix_sequence_start = self._config.sequence_length_lstm * -1 + 1
            if ix_sequence_start == 0:
                last_x_minus_1_experiences = []
            else:
                last_x_minus_1_experiences = self._memory.experiences[(self._config.sequence_length_lstm * -1 + 1):]
            prediction = self._model.predict_one(np.concatenate((last_x_minus_1_experiences, np.array([self._state])), axis=0))
        else:
            prediction = self._model.predict_one(np.array([self._state]))

        return prediction

    def choose_action(self, options, save_game):

        """
        Choose action: random based on chance value epsilon OR based on current policy for the given state
        """
        options = np.array(options)
        # if len(options.shape) == 1:
        #     options = np.expand_dims(options, axis=0)

        # Use chance to see whether to explore or exploit
        chance_value = random.random()
        if (chance_value < self._epsilon) and (save_game == False):
            choice = random.sample(np.where(options)[0].tolist(), k=1)[0]
            return choice
        else:
            prediction = self.state_to_prediction()

        if self._config.use_probabilities and (save_game == False):
            probabilities = self.prediction_to_probabilities(prediction, options)
            choice = int(np.random.choice(self._config.action_size, p=probabilities))
        else:
            prediction[~options] = -np.inf
            choice = int(np.argmax(prediction))

        return choice

    @staticmethod
    def options_to_eligible_idx(options):
        # Vectorized parsing of options to be able to choose an action randomly
        idx_options = options.nonzero()[1]
        option_counts = options.sum(axis=1)
        starts = np.cumsum(np.r_[0, option_counts[:-1]])

        return option_counts, idx_options, starts

    def choose_many_actions(self, options, selection, competition_game=False):

        """
        Choose many action: random based on chance value epsilon OR based on current policy for the given state
        for all games in the batch in parallel
        """

        options = np.array(options)

        # Check requirements
        if self._config.add_lstm:
            raise Exception("LSTM option is not yet suitable to use with parallel mode")

        # Use chance to see whether to explore or exploit
        chance_value = random.random()
        if (chance_value < self._epsilon) and (competition_game == False):

            # Dissect options
            option_counts, idx_options, starts = self.options_to_eligible_idx(options)

            # Random choice
            rng = np.random.default_rng()
            r = rng.integers(0, option_counts)[selection]

            # Vector of all choices
            picked_cols = idx_options[starts[selection] + r]
            all_choices = np.full(options.shape[0], 8, dtype=int)
            all_choices[selection] = picked_cols
            return all_choices
        else:
            if len(selection) > 1:
                predictions = self._model.predict_batch(np.array([state for ix, state in enumerate(self._state_many) if ix in selection]))
            elif len(selection) == 1:
                predictions = self._model.predict_batch(np.array([[self._state_many[selection[0]]]]))
                predictions = np.expand_dims(predictions, axis=0)
            else:
                predictions = []

            # Filter options
            predictions = predictions.astype(float, copy=False)

        if self._config.use_probabilities and (competition_game == False):
            probabilities = self.prediction_to_probabilities(predictions, options[selection])
            choices = np.array([np.random.choice(self._config.action_size, p=row) for row in probabilities])

        else:
            predictions[~options[selection]] = -np.inf
            choices = np.array([np.argmax(row) for row in predictions])

        all_choices = []
        for episode in range(self._config.number_of_episodes_per_round):
            if episode in selection:
                all_choices.append(int(choices[selection.index(episode)]))
            else:
                all_choices.append(8)

        return all_choices

    def write_summary_to_tensorboard(self):

        """
        Write a collection of logs to tensorboard
        """

        with self._summary_writer.as_default():
            for s in self._summary_writer_collection:
                tf.summary.scalar(s.get('name'), s.get('value'), step=s.get('step'))
        self._summary_writer_collection = []

    def learn_by_replay(self, batch_size = None, epochs = 1, verbose = False):

        """
        Learn by replay! The model gets trained by using the experience buffer. You take a random batch of the memory,
        shuffle them and do a training round using the deep Q (reinforcement) learning protocol.
        Save the logs for the tensorboard
        """

        # If batch_size undefined, fill with config batch size
        if batch_size is None:
            if self._config.preselect_batch:
                batch_size = int(self._config.batch_size * 4)
            else:
                batch_size = int(self._config.batch_size)

        # Only learn once memory has reached batch size and not in test mode
        if self._config.test_mode or (len(self._memory.experiences) < batch_size):
            return 0

        # Make a random batch
        print("\rCreating batch", end='')
        batch = self.create_batch(batch_size = batch_size)
        if not batch:
            return 0

        # Extract experiences
        print("\rDissecting batch", end='')
        states = np.array([b[-1][0] for b in batch], dtype=float)
        actions = np.array([b[-1][1] for b in batch], dtype=int)
        rewards = np.array([b[-1][2] for b in batch], dtype=float)
        next_states = np.array([(np.zeros(self._config.state_size)-1 if val[-2] is None else val[-2]) for seq in batch for val in seq])
        options = [b[-1][4] for b in batch]

        # Define end states
        is_terminal = (next_states == -1).all(axis=1)  # vectorized comparison for object array

        # Predict Q(s,a) given the batch of states
        print("\rPredicting q's for all states", end='')
        q_s_a = self._model.predict_batch(states)

        # Predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        print("\rPredicting q's for all next states", end='')
        q_s_a_d = self._model.predict_batch(next_states)

        # Clip corrected q
        print("\rClipping predicted qs", end='')
        corrected_qs = np.clip(q_s_a, -self._config.tag_points, self._config.tag_points)

        # Bulk predict next state
        if self._config.curiosity:
            print("\rPredicting next states for all states", end='')
            predicted_next_state = self._model.predict_batch_next_state(states)

            print("\rAdding curiosity bonus", end='')
            # Mean-squared error across state dimensions per row
            mse = ((predicted_next_state - np.vstack(next_states[~is_terminal])) ** 2).mean(axis=1)
            curiosity_bonus = np.zeros(batch_size, dtype=float)
            curiosity_bonus[~is_terminal] = self._config.curiosity_beta * mse
            rewards += self._config.curiosity_beta * curiosity_bonus

        # Non-terminal states: replace with reward+y*maxQ(s',a')-V(s, a)
        print("\rCorrecting q's of chosen actions of non-terminal states", end='')
        q_next_states = copy.deepcopy(q_s_a_d)
        v_current_state = np.sum(q_next_states*np.array(options), axis = 1)/np.sum(options, axis = 1)
        q_next_states[~np.array(options)] = -np.inf
        q_next_states = q_next_states.max(axis=1)
        corrected_qs[np.arange(batch_size),actions] = rewards + self._config.discount_factor * q_next_states - v_current_state

        # Overwrite terminal states: replace with reward
        print("\rCorrecting q's of chosen actions of terminal states", end='')
        corrected_qs[is_terminal, actions[is_terminal]] = rewards[is_terminal]

        # Filter batch
        if self._config.preselect_batch:
            print("\rFiltering batch", end='')
            correction_diff = np.round(np.sum(np.abs(corrected_qs - q_s_a), axis=1),1)
            idx_selection = np.argsort(correction_diff)[::-1][:int(batch_size/4)]
            all_states_selection = states[idx_selection]
            corrected_qs_selection = corrected_qs[idx_selection]
        else:
            all_states_selection = states
            corrected_qs_selection = corrected_qs

        # Train batch
        print("\rLearn!", end='')
        self._model.train_batch(all_states_selection, corrected_qs_selection, self._log_path,
                                             epochs=epochs, verbose=verbose)
        if self._config.curiosity:
            # Train batch next state
            self._model_next_state.train_batch_next_state(states, next_states)

        self.update_epsilon()

        # Add q to tensorboard
        end_state = np.abs(rewards) >= (self._config.tag_points - self._config.step_points * 2)
        q_logs = [np.sum(end_state), q_s_a]
        if np.sum(end_state) > 0:
            uncorrected_end_qs = q_s_a[end_state]
            corrected_end_qs = corrected_qs[end_state]
            crucial_action = np.abs(corrected_end_qs) >= (self._config.tag_points - self._config.step_points * 2)
            q_crucial_action = uncorrected_end_qs[crucial_action]
            q_alternative_action = uncorrected_end_qs[crucial_action == False]
            q_logs += [q_crucial_action, q_alternative_action]
        self._q_logs += [q_logs]

        return 1

    def add_logs_to_tensorboard(self):
        self.add_epsilon_to_tensorboard()
        self.add_episode_times_to_tensorboard()
        self.add_losses_to_tensorboard()
        self.add_qs_to_tensorboard()

    def add_qs_to_tensorboard(self):
        q_logs = self._q_logs[-1]
        found_end_states = q_logs[0] > 0
        if found_end_states:
            q_s_a, q_crucial_action, q_alternative_action = q_logs[1:]
        else:
            q_s_a = q_logs[1]

        self._summary_writer_collection += [
            {"name": 'Q/overall',
             "value": np.round(np.mean(np.abs(q_s_a)), 1),
             "step": self._arena.episode_count}
        ]

        if found_end_states:

            self._summary_writer_collection += [
                {
                    "name": 'Q/tagged-state-of-crucial-action',
                    "value": np.round(np.mean(np.abs(q_crucial_action)), 1),
                    "step": self._arena.episode_count
                },
                {
                    "name": 'Q/tagged-state-of-alternative-action',
                    "value": np.round(np.mean(np.abs(q_alternative_action)), 1),
                    "step": self._arena.episode_count
                }
            ]

    def add_losses_to_tensorboard(self):

        loss = self._model.losses[-1]
        self._summary_writer_collection += [{
            "name": 'learning/losses',
            "value": loss,
            "step": self._arena.episode_count
        }]

        if self._config.curiosity:
            loss_next_state = self._model_next_state.losses_next_state[-1]
            self._summary_writer_collection += [{
                "name": 'learning/losses-next-state',
                "value": loss_next_state,
                "step": self._arena.episode_count
            }]

    def add_epsilon_to_tensorboard(self):
        # Add epsilon to tensorboard
        self._summary_writer_collection += [
            {
                "name": 'episodes/epsilon',
                "value": np.round(self._epsilon, 3),
                "step": self._arena.episode_count
            }
        ]

    def add_episode_times_to_tensorboard(self):
        if len(self._arena.episode_times) > 0:
            self._summary_writer_collection += [
                {
                    "name": 'episodes/times',
                    "value": np.mean(self._arena.episode_times),
                    "step": self._arena.episode_count
                }
            ]
        self._arena.episode_times = []

    def add_rewards_to_tensorboard(self, turn_count):
        self._summary_writer_collection += [
            {
                "name": 'rewards/turn_count',
                "value": turn_count,
                "step": self._arena.episode_count
            }
        ]
        if self._total_reward_tagger != 0:
            self._summary_writer_collection += [
                {
                    "name": 'rewards/tagger',
                    "value": float(self._total_reward_tagger),
                    "step": self._arena.episode_count
                }
            ]
        if self._total_reward_runner != 0:
            self._summary_writer_collection += [
                {
                    "name": 'rewards/runner',
                    "value": float(self._total_reward_runner),
                    "step": self._arena.episode_count
                }
            ]

    def add_competition_to_tensorboard(self, episode_count, competitor):
        self._summary_writer_collection += [
            {
                "name": f'competition/{competitor}/tagger',
                "value": float(self._total_reward_tagger),
                "step": episode_count
            }
        ]
        self._summary_writer_collection += [
            {
                "name": f'competition/{competitor}/runner',
                "value": float(self._total_reward_runner),
                "step": episode_count
            }
        ]

    def create_batch(self, batch_size=None):

        # If batch_size undefined, fill with config batch size
        if batch_size is None:
            batch_size = int(self._config.batch_size)

        # Create a batch for the case of LSTM
        if self._config.add_lstm:

            selection_pool_size = len(self._memory.experiences) - self._model.sequence_length_lstm

            # Guard: not enough experiences to form a sequence
            if selection_pool_size < batch_size:
                return []

            selection = random.choices(range(selection_pool_size), k=batch_size)

            batch = []
            while len(batch) < batch_size:
                for i in selection:
                    experiences_i = []
                    add_i = 0
                    compare_4 = self._memory.experiences[i][0][4]
                    compare_5 = self._memory.experiences[i][0][5]
                    while len(experiences_i) < self._model.sequence_length_lstm:
                        if (compare_4 == self._memory.experiences[i + add_i][0][4]) and (
                                compare_5 == self._memory.experiences[i + add_i][0][5]):
                            if i + add_i >= (len(self._memory.experiences) - 1):
                                break
                            if self._memory.experiences[i + add_i][4] is None:
                                if len(experiences_i) != (self._config.sequence_length_lstm - 1):
                                    break
                            experiences_i += [self._memory.experiences[i + add_i]]
                        add_i += 1
                    if len(experiences_i) == self._config.sequence_length_lstm:
                        batch += [experiences_i]
                    else:
                        continue
                remaining = batch_size - len(batch)
                if remaining <= 0:
                    break
                selection = random.choices(range(selection_pool_size), k=remaining)

        # Create a batch for the case that it is not LSTM
        else:
            selection_pool_size = len(self._memory.experiences)

            if selection_pool_size < batch_size:
                return []

            selection = random.choices(range(selection_pool_size), k=batch_size)

            batch = [[self._memory.experiences[i]] for i in selection]

        return batch

    def show_q_in_state(self, game):

        prediction = [np.round(p, 1) for p in self.state_to_prediction()]
        game.render(prediction, self._state)
        to_print = '\n---'
        to_print += f'\n{self._state}'
        to_print += '\n---'
        for row in game.rendered:
            to_print += f"\n{'|'.join(row)}"
        to_print += '\n---'
        print(to_print)
        return to_print

    def update_reward_store(self):
        if self._total_reward_tagger != 0:
            self._reward_store_tagger.append(float(self._total_reward_tagger))
        if self._total_reward_runner != 0:
            self._reward_store_runner.append(float(self._total_reward_runner))

    def update_epsilon(self):

        # Update epsilon
        self._epsilon = (self._config.minimum_epsilon +
                     (self._config.maximum_epsilon - self._config.minimum_epsilon) *
                     math.exp(-self._config.bootstrap_value_epsilon * self._arena.episode_count))

    def new_game(self):
        self._total_reward_tagger = 0
        self._total_reward_runner = 0

    def reload(self, arena, checkpoint_path_overwrite = None):

        self._arena = arena

        if self._config.add_lstm:
            input_shape = (None, self._model.sequence_length_lstm, self._config.state_size)
        else:
            input_shape = (None, self._config.state_size)

        self._model.define_model()
        self._model.model.build(input_shape=input_shape)

        self.update_epsilon()

        if checkpoint_path_overwrite is not None:
            self._model.load_checkpoint(checkpoint_path_overwrite)
        else:
            checkpoints = [p for p in os.listdir(self._checkpoint_path) if
                           '-next-state' not in p and p.endswith('.keras')]
            checkpoints.sort()
            if checkpoints:
                self._model.load_checkpoint(os.path.join(self._checkpoint_path, checkpoints[-1]))

        if self._config.curiosity:
            self._model_next_state.define_model_next_state()
            self._model_next_state.build(input_shape=input_shape)
            if checkpoint_path_overwrite is not None:
                self._model_next_state.load_checkpoint(checkpoint_path_overwrite.replace('.keras', '-next-state.keras'))
            else:
                checkpoints_next_state = [p for p in os.listdir(self._checkpoint_path) if
                                          '-next-state' in p and p.endswith('.keras')]
                checkpoints_next_state.sort()
                if checkpoints_next_state:
                    self._model_next_state.load_checkpoint_next_state(os.path.join(self._checkpoint_path, checkpoints_next_state[-1]))
        self._summary_writer = tf.summary.create_file_writer(self._log_path)
        # exec(self._tboard_callback_command)

    @property
    def is_random(self):
        return self._is_random

    @property
    def name(self):
        return self._name

    @property
    def checkpoint_path(self):
        return self._checkpoint_path

    @property
    def reward_store_tagger(self):
        return self._reward_store_tagger

    @property
    def reward_store_runner(self):
        return self._reward_store_runner

    @property
    def epsilon(self):
        if self._epsilon < 0:
            raise ValueError('Epsilon is negative, not allowed. Fix.')
        return self._epsilon

    @property
    def current_reward(self):
        return self._current_reward

    @current_reward.setter
    def current_reward(self, current_reward):
        self._current_reward = current_reward

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, values):
        game_x_list, game_y_list, turn, is_tagger = values
        self._state = game_x_list + game_y_list + [turn, int(is_tagger)]

    @property
    def state_many(self):
        return self._state_many

    @state_many.setter
    def state_many(self, values):
        self._state_many = []
        for state in values:
            game_x_list, game_y_list, turn, is_tagger = state
            self._state_many += [game_x_list + game_y_list + [turn, int(is_tagger)]]

    @property
    def total_reward_tagger(self):
        return self._total_reward_tagger

    @total_reward_tagger.setter
    def total_reward_tagger(self, total_reward_tagger):
        self._total_reward_tagger = total_reward_tagger

    @property
    def total_reward_runner(self):
        return self._total_reward_runner

    @total_reward_runner.setter
    def total_reward_runner(self, total_reward_runner):
        self._total_reward_runner = total_reward_runner

    @property
    def current_options(self):
        return self._current_options

    @current_options.setter
    def current_options(self, current_options):
        self._current_options = current_options

    @property
    def state_path(self):
        return self._state_path

    @property
    def memory(self):
        return self._memory

    @property
    def model(self):
        return self._model

    @property
    def model_next_state(self):
        return self._model_next_state

    @property
    def arena(self):
        return self._arena

    @arena.setter
    def arena(self, arena):
        self._arena = arena

    @property
    def config(self):
        return self._config


# Player
class RandomPlayer:

    def __init__(self, name):
        # Identifying variables
        self._name = name
        self._is_random = True
        self._is_still = False

        # Collection variables
        self._reward_store_tagger = []
        self._reward_store_runner = []

        # State variables
        self._current_reward = 0
        self._total_reward_tagger = 0
        self._total_reward_runner = 0

    def choose_action(self, options, save_game, game=None):
        if not options:
            return 0
        return random.sample(options, k=1)[0]

    def new_game(self):
        self._total_reward_tagger = 0
        self._total_reward_runner = 0

    def update_reward_store(self):
        self._reward_store_tagger.append(float(self._total_reward_tagger))
        self._reward_store_runner.append(float(self._total_reward_runner))

    def add_rewards_to_tensorboard(self, turn_count):
        pass

    def update_epsilon(self):
        pass

    def learn_by_replay(self, batch_size):
        pass

    def reload(self, arena):
        pass


# Player
class StillPlayer:

    def __init__(self, name):
        # Identifying variables
        self._name = name
        self._is_random = True
        self._is_still = True

        # Collection variables
        self._reward_store_tagger = []
        self._reward_store_runner = []

        # State variables
        self._current_reward = 0
        self._total_reward_tagger = 0
        self._total_reward_runner = 0

    def choose_action(self, options, save_game, game=None):
        return 8

    def new_game(self):
        self._total_reward_tagger = 0
        self._total_reward_runner = 0

    def update_reward_store(self):
        self._reward_store_tagger.append(float(self._total_reward_tagger))
        self._reward_store_runner.append(float(self._total_reward_runner))

    def add_rewards_to_tensorboard(self, turn_count):
        pass

    def update_epsilon(self):
        pass

    def learn_by_replay(self, batch_size):
        pass

    def reload(self, arena):
        pass