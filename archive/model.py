# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:13:26 2021

@author: StannyGoffin
"""

# Import packages
import tensorflow.keras as tf
from config import retrieve_config
import numpy as np

# Extract all variables from the config file
config = retrieve_config()

# Model game
class Model:
    def __init__(self, num_states, num_actions, alpha, model1 = None, model2 = None):
        self._num_states  = num_states
        self._num_actions = num_actions
        self._batch_size  = config['BATCH_SIZE']
        # define the placeholders
        self._states    = None
        self._actions   = None
        if model1 is not None:
            self._model1 = model1
        else:
            self._model1 = None
        if model2 is not None:
            self._model2 = model2
        else:
            self._model2 = None
        self._alpha = alpha
        # the output operations
        self._logits    = None
        self._loss      = None
        self._optimizer = None
        self._var_init  = None

        # losses
        self._losses  = []
        self._losses1 = []
        self._losses2 = []
        # now setup the model
        self.define_model()

    def define_model(self):
        
        # Define two models
        model1 = tf.models.Sequential()
        model1.add(tf.layers.Dense(50, activation=tf.layers.LeakyReLU(alpha=0.01), input_shape=[self._num_states]))
        model1.add(tf.layers.Dense(50, activation=tf.layers.LeakyReLU(alpha=0.01)))
        model1.add(tf.layers.Dense(self._num_actions,activation='linear'))
        model1.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=self._alpha))
        self._model1 = model1
        
        # Define two models
        model2 = tf.models.Sequential()
        model2.add(tf.layers.Dense(50, activation=tf.layers.LeakyReLU(alpha=0.01), input_shape=[self._num_states]))
        model2.add(tf.layers.Dense(50, activation=tf.layers.LeakyReLU(alpha=0.01)))
        model2.add(tf.layers.Dense(self._num_actions,activation='linear'))
        model2.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=self._alpha))
        self._model2 = model2
        
    def predict_one(self, state):
        prediction1 = self._model1.predict(state.reshape(1, self._num_states), verbose = 0)[0]
        prediction2 = self._model2.predict(state.reshape(1, self._num_states), verbose = 0)[0]
        return prediction1, prediction2
    
    def predict_batch(self, states):
        return self._model1.predict(states,verbose=0), self._model2.predict(states,verbose=0)
    
    def train_batch(self, x_batch, y1_batch, y2_batch):
        log1 = self._model1.fit(x_batch, y1_batch, verbose=0)
        self._losses1 += log1.history.get('loss')
        log2 = self._model2.fit(x_batch, y2_batch, verbose=0)
        self._losses2 += log2.history.get('loss')
        self._losses += log1.history.get('loss') + log2.history.get('loss')

    def mutate(self):
        copy1 = tf.models.clone_model(self._model1)
        new_layers = []
        for l in range(len(self._model1.get_weights())):
            layer = self._model1.get_weights()[l]
            if len(layer.shape) == 1:
                random_mutation_probs = np.random.rand(layer.shape[0])
            else:
                random_mutation_probs = np.random.rand(layer.shape[0],layer.shape[1])
            random_mutation_probs = np.where(random_mutation_probs < config['CHANCE_OF_MUTATION'], (np.random.rand()-0.5)/2,0)
            new_layer = layer + random_mutation_probs
            new_layers += [new_layer]
        copy1.set_weights(new_layers)

        copy2 = tf.models.clone_model(self._model2)
        new_layers = []
        for l in range(len(self._model2.get_weights())):
            layer = self._model2.get_weights()[l]
            if len(layer.shape) == 1:
                random_mutation_probs = np.random.rand(layer.shape[0])
            else:
                random_mutation_probs = np.random.rand(layer.shape[0],layer.shape[1])
            random_mutation_probs = np.where(random_mutation_probs < config['CHANCE_OF_MUTATION'], (np.random.rand()-0.5)/2,0)
            new_layer = layer + random_mutation_probs
            new_layers += [new_layer]
        copy2.set_weights(new_layers)

        return copy1, copy2

