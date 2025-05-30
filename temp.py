import numpy as np
import tensorflow.keras as tf
inputs = np.random.random((32, 10, 8))
lstm = tf.layers.LSTM(4)
output = lstm(inputs)
print(output.shape)

input = states.astype(float)
for layer in self._model.layers:
    print(layer)
    print(input.shape)
    output = layer(input)
    print(output.shape)
    input = output