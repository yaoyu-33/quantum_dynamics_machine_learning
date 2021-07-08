"""Define the neural network architecture."""
import tensorflow as tf
from tensorflow.keras import layers


class RNNModel(tf.keras.Model):
    """Machine Learning Based Emulator (RNN Type)."""

    def __init__(self, config):
        """Initialization."""
        super().__init__()
        self.flatten = layers.Reshape(
            (config.input_frames, config.window_size*config.input_channels))
        self.dense1 = layers.Dense(
            config.hidden_size, activation="relu")
        self.gru = tf.keras.layers.GRU(
            config.hidden_size, return_state=True, activation='relu',
            dropout=config.dropout_rate)
        self.dense2 = layers.Dense(
            config.window_size * config.output_channels)
        self.reshape = layers.Reshape(
            (config.window_size, config.output_channels))

    def call(self, inputs, training=False, mask=None):
        """Calls the model on new inputs."""
        x = self.flatten(inputs)
        x = self.dense1(x)
        output, final_state = self.gru(
            x, initial_state=None, training=training)
        x = tf.concat((output, final_state), axis=-1)
        x = self.dense2(x)
        x = self.reshape(x)
        return x
