# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:13:26 2021

@author: StannyGoffin
"""
import os

# Import packages
import tensorflow as tfbare
import tensorflow.keras as tf
import numpy as np
from config import Config

# Model game
class Model(Config):
    def __init__(self, experiment="defaultname", model=None, learningRate = 0.0001, layers = [50,50], addLSTM = False):

        # Import config
        Config.__init__(self)

        # Experiment name
        self._experiment = experiment

        # Define model
        self._learningRate = learningRate  # formerly alpha
        self._layers = layers
        self._model = model if model is not None else None
        self._add_LSTM = addLSTM

        # Define the placeholders
        self._states = None
        self._actions = None

        # Define the output operations
        self._logits = None
        self._loss = None
        self._optimizer = None
        self._var_init = None

        # Initialize the loss history
        self._losses = []

        # Initialize the checkpoint callback
        self.cp_callback = None

        # Set up the models
        self.define_model()

    def define_model(self):
        with tfbare.device('/gpu:0'):
            layers = []
            for layer_nr, layer in enumerate(self._layers):
                if self._add_LSTM & (layer_nr == 0):
                    layers += [tf.layers.LSTM(units=layer)]
                else:
                    layers += [tf.layers.Dense(layer,
                                               activation=tf.layers.LeakyReLU(alpha=self._learningRate))]
            layers += [tf.layers.Dense(self.numActions, activation='linear')]
            model = tf.models.Sequential(layers)
            model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=self._learningRate))
        self._model = model

    def predict_one(self, state):
        if self._add_LSTM:
            state = np.array(state).reshape(1, self._sequence_length, self.numStates)
        prediction = np.squeeze(self._model.predict(state, verbose=0))
        return prediction

    def predict_batch(self, states):
        if self._add_LSTM:
            states = states.astype(float)
        return np.squeeze(self._model.predict(states, verbose=0))

    def train_batch(self, x_batch, y_batch):

        # Train batch
        log = self._model.fit(x_batch, y_batch, epochs=1, verbose=0)

        # Add losses to log
        self._losses += log.history.get('loss')

        # Add losses to tensorboard
        with self._summary_writer.as_default():
            tfbare.summary.scalar('Losses', log.history.get('loss')[0], step = self._summary_loss_step)
            self._summary_loss_step += 1
            if len(self._losses) % 100 == 0:
                for i, layer in enumerate(self._model.layers):
                    if self._add_LSTM:
                        if i == 0:
                            for it, ly in enumerate(layer.get_weights()):
                                tfbare.summary.histogram(f'LSTM_layer_{it}', ly, step = self._summary_params_step)

                            continue
                    weights, biases = layer.get_weights()
                    tfbare.summary.histogram(f'Layer_{i}_weights', weights, step = self._summary_params_step)
                    tfbare.summary.histogram(f'Layer_{i}_biases', biases, step = self._summary_params_step)
                self._summary_params_step += 1

    def save_checkpoint(self, model, cnt, name, training_phase):

        # Save weights
        model.save_weights(os.getcwd() + self._experiment + f'/checkpoints/{name}/{training_phase}/cp-{cnt:06d}.weights.h5')

    def load_checkpoint(self, path):

        # Load weights
        self._model.load_weights(path)

class ModelNextState():
    def __init__(self, modelNextState=None):

        # Define model
        self._learning_rate_next_state = 0.001  # formerly alpha
        self._layers_next_state = [50,50]
        self._model_next_state = modelNextState if modelNextState is not None else None

        # Set up the models
        self.define_model_next_state()

        # Initialize the loss history
        self._losses_next_state = []

    def define_model_next_state(self):
        with tfbare.device('/gpu:0'):
            layers = []
            for layer_nr, layer in enumerate(self._layers_next_state):
                layers += [tf.layers.Dense(layer, activation=tf.layers.LeakyReLU(alpha=self._learning_rate_next_state))]
            layers += [tf.layers.Dense(self.numStates, activation='linear')]
            model = tf.models.Sequential(layers)
            model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=self._learning_rate_next_state))
        self._model_next_state = model

    def predict_one_next_state(self, state):
        if self._add_LSTM:
            state = np.array(state).reshape(1, self._sequence_length, self.numStates)
        return np.squeeze(self._model_next_state.predict(state, verbose=0))

    def predict_batch_next_state(self, states):
        if self._add_LSTM:
            states = states.astype(float)
        return np.squeeze(self._model_next_state.predict(states, verbose=0))

    def train_batch_next_state(self, x_batch, y_batch):

        # Train batch
        log = self._model_next_state.fit(x_batch, y_batch, epochs=5, verbose=0)

        # Add losses to log
        self._losses_next_state += log.history.get('loss')

        # Add losses to tensorboard
        with self._summary_writer.as_default():
            tfbare.summary.scalar('Losses next state', log.history.get('loss')[0], step = self._summary_loss_step)
            if len(self._losses_next_state) % 1000 == 0:
                for i, layer in enumerate(self._model_next_state.layers):
                    weights, biases = layer.get_weights()
                    tfbare.summary.histogram(f'Layer_{i}_weights next state', weights, step = self._summary_params_step)
                    tfbare.summary.histogram(f'Layer_{i}_biases next state', biases, step = self._summary_params_step)

    def save_checkpoint_next_state(self, model, cnt, name, training_phase):

        # Save weights
        model.save_weights(
            os.getcwd() + self._experiment + f'/checkpoints/{name}/{training_phase}/cp-{cnt:06d}-next-state.weights.h5')

    def load_checkpoint_next_state(self, path):

        # Load weights
        self._model_next_state.load_weights(path)