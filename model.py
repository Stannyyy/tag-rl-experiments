# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:13:26 2021

@author: StannyGoffin
"""
import os

# Import packages
import tensorflow.keras as tf
import numpy as np
from config import Config

# Model game
class Model(Config):
    def __init__(self, model=None, learningRate = 0.0001, layers = [50,50]):

        # Import config
        Config.__init__(self)

        # Define model
        self._learningRate = learningRate  # formerly alpha
        self._layers = layers
        self._model = model if model is not None else None

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
        self._model = self.define_model()

    def define_model(self):
        layers = []
        for layer_nr in range(len(self._layers)):
            if layer_nr == 0:
                layers += [tf.layers.Dense(self._layers[layer_nr],
                                           activation=tf.layers.LeakyReLU(alpha=self._learningRate),
                                           input_shape=[self.numStates])]
            else:
                layers += [tf.layers.Dense(self._layers[layer_nr],
                                           activation=tf.layers.LeakyReLU(alpha=self._learningRate))]
        layers += [tf.layers.Dense(self.numActions, activation='linear')]
        model = tf.models.Sequential(layers)
        model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=self._learningRate))
        return model

    def predict_one(self, state):
        prediction = self._model.predict(np.array(state).reshape(1, self.numStates), verbose=0)[0]
        return prediction

    def predict_batch(self, states):
        return self._model.predict(states, verbose=0)

    def train_batch(self, x_batch, y_batch):

        # Train batch
        log = self._model.fit(x_batch, y_batch, epochs=1, verbose=0)

        # Add losses to log
        self._losses += log.history.get('loss')

    def mutate(self, model):

        # For each layer, copy model weights and mutate with chanceOfMutation
        new_layers = []
        for l in range(len(model.get_weights())):

            # Retrieve weights
            layer = model.get_weights()[l]

            # Mutate weights
            random_mutation_probs = np.random.rand(*layer.shape)
            random_mutation_probs = np.where(random_mutation_probs < self.chanceOfMutation,
                                             (np.random.rand() - 0.5) / 2, 0)
            new_layer = layer + random_mutation_probs
            new_layers += [new_layer]

        # Clone model 1 and set new, mutated weights
        copy = tf.models.clone_model(model)
        copy.set_weights(new_layers)

        return copy

    def save_checkpoint(self, model, cnt, name, training_phase):

        # Save weights
        model.save_weights(os.getcwd() + f'/checkpoints/{name}/{training_phase}/cp-{cnt:06d}.ckpt')