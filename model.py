# Import packages
import tensorflow as tf
from tensorflow.keras import saving
import numpy as np

# Find device
def find_device():

    # If gpu available, take gpu. If not, just take what is available
    devices = tf.config.list_physical_devices()
    devices = ([device.name.replace('physical_device:', '') for device in devices if
                'gpu' in device.device_type.lower()] +
               [device.name.replace('physical_device:', '') for device in devices])

    # If no devices, raise an error
    if len(devices) == 0:
        raise Exception("No devices found by tensorflow")

    # Else, use the first device (GPU preferred)
    return tf.device(devices[0])

# Model game
class Model:

    def __init__(self, config, model=None):

        # Define model
        self._model = model if model is not None else None

        # Initialize config
        self._config = config

        # Initialize the loss history
        self._losses = []

        # Set up the models
        self.define_model()

    def define_model(self):
        """
        Define the neural network to predict Q values
        """
        with find_device():
            layers = []
            for layer_nr, layer in enumerate(self._config.layers):
                if self._config.add_lstm and (layer_nr == 0):
                    # LSTM as the first layer when add_lstm is enabled
                    layers += [tf.keras.layers.LSTM(units=layer)]
                else:
                    # Dense layers where the number of units is defined by self.layers
                    # PReLU is a leaky relu of which alpha is learned
                    layers += [tf.keras.layers.Dense(layer),
                               tf.keras.layers.PReLU()]

            # Finalize with an output layer to predict the Q values for all the different actions
            layers += [tf.keras.layers.Dense(self._config.action_size, activation='linear')]

            # Compile model
            model = tf.keras.models.Sequential(layers)
            model.compile(
                loss='mse',
                optimizer=tf.optimizers.Adam(learning_rate=self._config.learning_rate)
            )
            self._model = model

    def predict_one(self, state):
        """
        Predict Q-values for a single state
        """

        # If state is empty, return empty
        if len(state[0]) == 0:
            return np.array([])

        # If add_lstm is enabled, reshape the state into one that can be used by the LSTM layer of the model
        if self._config.add_lstm:
            state = np.array(state).astype(float).reshape(1, self._config.sequence_length_lstm, self._config.state_size)

        # Predict Q values
        prediction = self._model.predict(np.array(state), verbose=0)

        # Remove unnecessary dimensions
        return np.squeeze(prediction)

    def predict_batch(self, states):
        """
        Predict Q-values for a batch of states
        Returns an array of shape (batch_size, action_size)
        """

        # To numpy array
        states = np.array(states)

        # A batch of one is just one
        if len(states) == 1:
            return self.predict_one(states[0])

        # Prep for LSTM if necessary
        if self._config.add_lstm:
            states = states.astype(float).reshape(
                (self._config.batch_size, self._config.sequence_length_lstm, self._config.state_size))

        # Predict Q values
        predictions = self._model.predict(states, verbose=0)

        return np.squeeze(predictions)

    def train_batch(self, x_batch, y_batch, log_dir, epochs=1, verbose=False):
        """
        Train one batch and log the training loss
        Returns a summary dict for the tensorboard
        """

        # Train batch
        # tf.profiler.experimental.start(log_dir)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        log = self._model.fit(x_batch, y_batch, epochs=epochs, verbose=verbose, callbacks=[callback])
        # tf.profiler.experimental.stop()

        # Add losses to log
        self._losses += log.history.get('loss')

    def save_checkpoint(self, checkpoint_path_version):
        """
        Save model weights
        """

        # Save weights
        self._model.save(checkpoint_path_version + '.keras')

    def load_checkpoint(self, path):
        """
        Load model weights from path
        """

        # Load weights
        if '.weights.h5' in path:
            self._model.load_weights(path)
        else:
            self._model = saving.load_model(path, compile=True)

    @property
    def losses(self):
        return self._losses

    @property
    def model(self):
        return self._model