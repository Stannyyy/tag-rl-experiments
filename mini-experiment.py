import tensorflow.keras as tf
import numpy as np

# with tf.device('/gpu:0'):
#     layers = []
#     for layer_nr, layer in enumerate(self._layers):
#         if self._add_LSTM & (layer_nr == 0):
#             layers += [tf.layers.LSTM(units=100)]
#         else:
#             layers += [tf.layers.Dense(layer,
#                                        activation=tf.layers.LeakyReLU())]
#     layers += [tf.layers.Dense(self.numActions, activation='linear')]
#     model = tf.models.Sequential(layers)
#     model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=self._learning_rate))
# self._model = model

x_batch = np.array([
    [[1,2,3,4,5,6],
     [1,2,4,5,6,7],
     [1,2,5,6,7,8],
     [0,1,2,3,4,5],
     [0,1,2,5,6,7]]
    ,
    [[1,3,3,4,9,6],
     [1,3,4,9,6,7],
     [1,3,9,6,7,8],
     [0,1,3,3,4,9],
     [0,1,3,9.0,6,7]]
])
y_batch = np.array([
    [1,1,1,1,1,1,1,1,10],
    [1,1,1,10,1,1,-1,-1,-1]
])
model = tf.Sequential([
    tf.layers.LSTM(units=100),
    tf.layers.Dense(100),
    tf.layers.Dense(9)
])
model.compile(loss='mse')
data = np.empty((0, 18))
for i in range(100):
    log = model.fit(x_batch, y_batch, epochs=500, verbose=0)
    if (i != 0):

        prediction = model.predict(x_batch)
        data = np.vstack([data, prediction.reshape(1,-2)])

x_batch = np.array([
     [0, 1, 2, 5, 6, 7]
    ,
     [0, 1, 3, 9.0, 6, 7]
])
model = tf.Sequential([
    tf.layers.Dense(100),
    tf.layers.Dense(100),
    tf.layers.Dense(9)
])
model.compile(loss='mse')
data2 = np.empty((0, 18))
for i in range(100):
    log = model.fit(x_batch, y_batch, epochs=500, verbose=0)
    if (i != 0):

        prediction = model.predict(x_batch)
        data2 = np.vstack([data2, prediction.reshape(1,-2)])

import matplotlib.pyplot as plt
# Plotting each of the 18 columns as a line diagram
figs = []
for i in range(18):
    plt.figure()
    plt.plot(range(99), data[:, i], label='data')
    plt.plot(range(99), data2[:, i], label='data2', linestyle='--')
    plt.xlabel('Row index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    print('done')
