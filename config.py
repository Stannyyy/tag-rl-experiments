import warnings

# SET VARIABLES
# Reinforcement learning model variables
class Config:
    def __init__(self):

        # Training regime
        self._maximum_epsilon=1
        self._minimum_epsilon=0
        self._number_of_episodes_total=5000000
        self._number_of_episodes_per_round=10000
        self._bootstrap_value_epsilon=0.00001 # formerly lambda
        self._discount_factor=0.999 # formerly gamma
        self._game_play_mode='parallel'
        self._redo_batch=True

        # Stochastic policy
        self._use_probabilities = True
        self._temperature = 5
        self._temperature_alpha = 0.0000001
        self._temperature_beta = 0.05
        self._average_end_probability = 90
        self._temperature_factor = None

        # Competition regime
        self._number_of_competition_episodes = 100

        # Model variables
        self._learning_rate=0.00001
        self._layers=[100, 100, 100]

        # Architecture options
        ## LSTM
        self._add_lstm=False
        self._sequence_length_lstm=1

        # Memory variables
        self._batch_size=100000
        self._memory_size=5000

        # Game variables
        self._grid_size=25
        self._number_of_players=2
        self._number_of_taggers=1
        self._maximum_steps=150
        self._step_points=1.0
        self._tag_points=51.0
        self._action_size=9

        # Render variables
        self._create_video=True

        # Test mode
        self._test_mode=False

        # If custom state_size/action_size (for testing)
        self._state_size_overwrite=None
        self._action_size_overwrite=None

        # Derived attributes
        self._state_size = 2 * self._number_of_players + 2

        # Calculate temperature factor
        self.temperature_factor_from_average_end_probability()

    def temperature_factor_from_average_end_probability(self):
        if self._action_size != 9:
            raise Exception("Action size must be 9 for this function to work.")
        end_factors = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                       0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5]
        end_probabilities = [1.0, 0.997, 0.995, 0.992, 0.989, 0.988, 0.984, 0.978, 0.976, 0.974, 0.971, 0.948, 0.934,
                             0.906, 0.888, 0.849, 0.801, 0.759, 0.705, 0.651, 0.599, 0.558, 0.502, 0.457, 0.41, 0.357,
                             0.315, 0.273, 0.244, 0.206, 0.179, 0.151, 0.111]
        closest_index = min(range(len(end_probabilities)), key=lambda i: abs(end_probabilities[i] - self._average_end_probability))
        self._temperature_factor = end_factors[closest_index]

    @property
    def maximum_epsilon(self):
        return self._maximum_epsilon

    @maximum_epsilon.setter
    def maximum_epsilon(self, value):
        self._maximum_epsilon = value

    @property
    def minimum_epsilon(self):
        return self._minimum_epsilon

    @minimum_epsilon.setter
    def minimum_epsilon(self, value):
        self._minimum_epsilon = value

    @property
    def number_of_episodes_total(self):
        return self._number_of_episodes_total

    @number_of_episodes_total.setter
    def number_of_episodes_total(self, value):
        self._number_of_episodes_total = value

    @property
    def number_of_episodes_per_round(self):
        return self._number_of_episodes_per_round

    @number_of_episodes_per_round.setter
    def number_of_episodes_per_round(self, value):
        self._number_of_episodes_per_round = value

    @property
    def bootstrap_value_epsilon(self):
        return self._bootstrap_value_epsilon

    @bootstrap_value_epsilon.setter
    def bootstrap_value_epsilon(self, value):
        self._bootstrap_value_epsilon = value

    @property
    def discount_factor(self):
        return self._discount_factor

    @discount_factor.setter
    def discount_factor(self, value):
        self._discount_factor = value

    @property
    def use_probabilities(self):
        return self._use_probabilities

    @use_probabilities.setter
    def use_probabilities(self, value):
        self._use_probabilities = bool(value)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    @property
    def temperature_alpha(self):
        return self._temperature_alpha

    @property
    def temperature_beta(self):
        return self._temperature_beta

    @property
    def temperature_factor(self):
        return self._temperature_factor

    @temperature_factor.setter
    def temperature_factor(self, value):
        raise Exception("Use factor_from_average_probability module to set temperature_factor")

    @property
    def game_play_mode(self):
        return self._game_play_mode

    @game_play_mode.setter
    def game_play_mode(self, value):
        if value not in ['parallel', 'sequential']:
            raise Exception("Game play mode must be either 'parallel' or 'sequential'.")
        self._game_play_mode = value
        
    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, value):
        self._layers = list(value)

    @property
    def add_lstm(self):
        return self._add_lstm

    @add_lstm.setter
    def add_lstm(self, value):
        self._add_lstm = bool(value)

    @property
    def sequence_length_lstm(self):
        return self._sequence_length_lstm

    @sequence_length_lstm.setter
    def sequence_length_lstm(self, value):
        self._sequence_length_lstm = int(value)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = int(value)

    @property
    def memory_size(self):
        return self._memory_size

    @memory_size.setter
    def memory_size(self, value):
        if self._game_play_mode == 'parallel':
            raise Exception("Batch size is only relevant in sequential mode.")
        self._memory_size = int(value)

    @property
    def grid_size(self):
        return self._grid_size

    @grid_size.setter
    def grid_size(self, value):
        warnings.warn("Also change tag_points.")
        self._grid_size = int(value)

    @property
    def number_of_players(self):
        return self._number_of_players

    @number_of_players.setter
    def number_of_players(self, value):
        self._number_of_players = int(value)
        # keep derived state size in sync if no overwrite is set
        if self._state_size_overwrite is None:
            self._state_size = 2 * self._number_of_players + 2

    @property
    def number_of_taggers(self):
        return self._number_of_taggers

    @number_of_taggers.setter
    def number_of_taggers(self, value):
        self._number_of_taggers = int(value)

    @property
    def maximum_steps(self):
        return self._maximum_steps

    @maximum_steps.setter
    def maximum_steps(self, value):
        warnings.warn("Consider changing tag_points to reflect this change.")
        self._maximum_steps = int(value)

    @property
    def step_points(self):
        return self._step_points

    @step_points.setter
    def step_points(self, value):
        self._step_points = float(value)

    @property
    def tag_points(self):
        return self._tag_points

    @tag_points.setter
    def tag_points(self, value):
        self._tag_points = float(value)

    @property
    def create_video(self):
        return self._create_video

    @create_video.setter
    def create_video(self, value):
        self._create_video = bool(value)

    @property
    def test_mode(self):
        return self._test_mode

    @test_mode.setter
    def test_mode(self, value):
        self._test_mode = bool(value)

    @property
    def state_size(self):
        # compute from overwrite when provided, else from current number_of_players
        return self._state_size_overwrite if self._state_size_overwrite is not None else (2 * self._number_of_players + 2)

    @state_size.setter
    def state_size(self, value):
        raise Exception("Code is not ready for state size changes, use state_size_overwrite for testing purposes.")

    @property
    def state_size_overwrite(self):
        return self._state_size_overwrite

    @state_size_overwrite.setter
    def state_size_overwrite(self, value):
        self._state_size_overwrite = None if value is None else int(value)
        # also keep _state_size aligned for any direct readers
        if self._state_size_overwrite is None:
            self._state_size = 2 * self._number_of_players + 2
        else:
            self._state_size = self._state_size_overwrite

    @property
    def action_size(self):
        return self._action_size_overwrite if self._action_size_overwrite is not None else self._action_size

    @action_size.setter
    def action_size(self, value):
        raise Exception("Code is not ready for action size changes, use action_size_overwrite for testing purposes.")

    @property
    def action_size_overwrite(self):
        return self._action_size_overwrite

    @action_size_overwrite.setter
    def action_size_overwrite(self, value):
        self._action_size_overwrite = None if value is None else int(value)

    @property
    def redo_batch(self):
        return self._redo_batch

    @redo_batch.setter
    def redo_batch(self, value):
        self._redo_batch = bool(value)

    @property
    def number_of_competition_episodes(self):
        return self._number_of_competition_episodes

    @number_of_competition_episodes.setter
    def number_of_competition_episodes(self, value):
        self._number_of_competition_episodes = value
