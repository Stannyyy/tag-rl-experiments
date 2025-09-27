# SET VARIABLES
# Reinforcement learning model variables
class Config:
    def __init__(self, **kwargs):
        # Default values
        defaults = dict(
            # Training regime
            max_epsilon=1,
            min_epsilon=0.01,
            num_episodes=1000000,
            num_episodes_per_round=10000,
            bootstrap_value_epsilon=0.001, # formerly lambda
            discount_factor=0.99, # formerly gamma
            preselect_batch=False,
            use_probabilities=False,

            # Model variables
            learning_rate=0.0001,
            layers=[100, 100, 100],

            # Architecture options
            ## LSTM
            add_lstm=False,
            sequence_length_lstm=1,

            ## Curiosity bonus
            curiosity=False,
            curiosity_beta=0,

            # Memory variables
            batch_size=100, # Note these are only relevant in sequential mode
            max_memory=5000, # Note these are only relevant in sequential mode

            # Game variables
            grid_size=10, # If this is changed, consider changing tag_points
            num_players=2,
            num_taggers=1,
            max_steps=50, # If this is changed, consider changing tag_points
            step_points=1.0,
            tag_points=51.0,
            # Note that code has to be changed to support more or less actions
            num_actions=9,

            # Render variables
            create_video=True,

            # Test mode
            test_mode=False,

            # If custom num_states/num_actions (for testing)
            num_states_overwrite=None,
            num_actions_overwrite=None
        )

        # Apply overrides from kwargs
        defaults.update(kwargs)

        # Set all attributes dynamically
        for key, value in defaults.items():
            setattr(self, key, value)

        # Derived attributes
        self.num_states = 2 * self.num_players + 2