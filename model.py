# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:13:26 2021

@author: StannyGoffin
"""
import os

# Import packages
import tensorflow as tf
import numpy as np
from config import Config

# Find device
def find_device():

    # If gpu available, take gpu, if not just take what is available
    devices = tf.config.list_physical_devices()
    devices = ([device.name.replace('physical_device:', '') for device in devices if
                'gpu' in device.device_type.lower()] +
               [device.name.replace('physical_device:', '') for device in devices])
    
    # If no devices, raise error
    if len(devices) == 0:
        raise Exception("No devices found by tensorflow")
    
    # Else, use the first device (GPU preferred)
    return tf.device(devices[0])

# Model game
class Model(Config):

    def __init__(self, model, **kwargs):

        # Import config
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

        # Define model
        self.model = model if model is not None else None

        # Initialize the loss history
        self.losses = []

        # Set up the models
        self.define_model()

    def define_model(self):
        """
        Define the neural network to predict Q values
        """
        with find_device():
            layers = []
            for layer_nr, layer in enumerate(self.layers):
                if self.add_lstm and (layer_nr == 0):
                    # LSTM as the first layer when add_lstm is enabled
                    layers += [tf.keras.layers.LSTM(units=layer)]
                else:
                    # Dense layers with number of units defined by self.layers
                    # PReLU is leaky relu of which alpha is learned
                    layers += [tf.keras.layers.Dense(layer),
                               tf.keras.layers.PReLU()]

            # Finalize with an output layer to predict the Q values for all the different actions
            layers += [tf.keras.layers.Dense(self.num_actions, activation='linear')]

            # Compile model
            model = tf.keras.models.Sequential(layers)
            model.compile(
                loss='mse',
                optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate)
            )
            self.model = model

    def predict_one(self, state):
        """
        Predict Q-values for a single state
        """

        # If add_lstm is enabled, reshape the state into one that can be used by the LSTM layer of the model
        if self.add_lstm:
            state = np.array(state).astype(float).reshape(1, self.sequence_length_lstm, self.num_states)

        # Predict Q values
        prediction = self.model.predict(state, verbose=0)

        # Remove unnecessary dimensions
        return np.squeeze(prediction)

    def predict_batch(self, states):
        """
        Predict Q-values for a batch of states
        Returns an array of shape (batch_size, num_actions)
        """

        # A batch of one is just one
        if len(states) == 1:
            return self.predict_one(states[0])

        # Prep for LSTM if necessary
        if self.add_lstm:
            states = states.astype(float).reshape((self.batch_size, self.sequence_length_lstm, self.num_states))

        # Predict Q values
        predictions = self.model.predict(states, verbose=0)

        return np.squeeze(predictions)

    def train_batch(self, x_batch, y_batch, cnt, log_dir, epochs=1, verbose=False):
        """
        Train one batch and log the training loss
        Returns a summary dict for the tensorboard
        """

        # Train batch
        # tf.profiler.experimental.start(log_dir)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        log = self.model.fit(x_batch, y_batch, epochs=epochs, verbose=verbose, callbacks=[callback])
        # tf.profiler.experimental.stop()

        # Add losses to log
        self.losses += log.history.get('loss')

        # Add losses to tensorboard
        return {
            "name": 'params/losses',
            "value": np.round(log.history.get('loss'),5)[0],
            "step": cnt
        }

    def save_checkpoint(self, cnt, name, training_phase):
        """
        Save model weights
        """

        # Save weights
        checkpoint_path = os.path.join(os.getcwd(), self.experiment, 'checkpoints', name, training_phase)
        self.model.save_weights(os.path.join(checkpoint_path, f'cp-{cnt:06d}.weights.h5'))

    def load_checkpoint(self, path):
        """
        Load model weights from path
        """

        # Load weights
        self.model.load_weights(path)

# When using a curiosity bonus, we need a model to predict the next state
class ModelNextState(Config):

    def __init__(self, experiment="defaultName", modelNextState=None, add_lstm=False, sequence_length_lstm=1):

        # Import config
        Config.__init__(self)

        # Experiment name
        self.experiment = experiment

        # Define model
        self.learning_rate_next_state = 0.001  # formerly alpha
        self.layers_next_state = [50,50]
        self.model_next_state = modelNextState if modelNextState is not None else None
        self.add_lstm = add_lstm
        self.sequence_length_lstm = sequence_length_lstm

        # Set up the models
        self.define_model_next_state()

        # Initialize the loss history
        self.losses_next_state = []

    def define_model_next_state(self):

        """
        Define the neural network to predict the next state
        """

        # If gpu available, take gpu, if not just take what is available
        with find_device():
            layers = []
            for layer_nr, layer in enumerate(self.layers_next_state):
                if self.add_lstm and (layer_nr == 0):
                    # LSTM as the first layer when add_lstm is enabled
                    layers += [tf.keras.layers.LSTM(units=layer)]
                else:
                    # Dense layers with number of units defined by self.layers
                    # PReLU is leaky relu of which alpha is learned
                    layers += [tf.keras.layers.Dense(layer),
                               tf.keras.layers.PReLU()]

            # Finalize with an output layer to predict the next state values
            layers += [tf.keras.layers.Dense(self.num_states, activation='linear')]

            # Compile model
            model = tf.keras.models.Sequential(layers)
            model.compile(
                loss='mse',
                optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate_next_state)
            )
            self.model_next_state = model

    def predict_one_next_state(self, state):
        """
        Predict the next state for a single state
        """

        # If add_lstm is enabled, reshape the state into one that can be used by the LSTM layer of the model
        if self.add_lstm:
            state = np.array(state).reshape(1, self.sequence_length_lstm, self.num_states)

        # Predict next state values
        prediction = self.model_next_state.predict(state, verbose=0)

        # Remove unnecessary dimensions
        return np.squeeze(prediction)

    def predict_batch_next_state(self, states):
        """
        Predict next state values for a batch of states
        Returns an array of shape (batch_size, num_states)
        """

        # A batch of one is just one
        if len(states) == 1:
            return self.predict_one_next_state(states[0])

        # Prep for LSTM if necessary
        if self.add_lstm:
            states = states.astype(float).reshape((self.batch_size, self.sequence_length_lstm, self.num_states))

        # Predict next state values
        predictions = self.model_next_state.predict(states, verbose=0)

        return np.squeeze(predictions)

    def train_batch_next_state(self, x_batch, y_batch, cnt):
        """
        Train one batch and log the training loss
        Returns a summary dict for the tensorboard
        """

        # Train batch
        log = self.model_next_state.fit(x_batch, y_batch, epochs=1, verbose=0)

        # Add losses to log
        self.losses_next_state += log.history.get('loss')

        # Add losses to tensorboard
        return {
                "name": 'params/losses-next-state',
                "value": np.round(log.history.get('loss'),5)[0],
                "step": cnt
             }

    def save_checkpoint_next_state(self, cnt, name, training_phase):
        """
        Save model weights
        """

        # Save weights
        checkpoint_path = os.path.join(os.getcwd(), self.experiment, 'checkpoints', name, training_phase)
        self.model_next_state.save_weights(os.path.join(checkpoint_path, f'cp-{cnt:06d}-next-state.weights.h5'))

    def load_checkpoint_next_state(self, path):
        """
        Load model weights from path
        """

        # Load weights
        self.model_next_state.load_weights(path)
