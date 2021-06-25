#Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class LinearModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.last = layers.Lambda(lambda x: x[:, -1, :])
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(config.window_size * config.output_channels)
        self.reshape = layers.Reshape((config.window_size, config.output_channels))

    def call(self, input_tensor, training=False):
        x = self.last(input_tensor)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.reshape(x)

        return x

class DenseModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.flatten = layers.Reshape((config.input_frames, config.window_size * config.input_channels))
        self.dense1 = layers.Dense(config.hidden_size, activation="relu")
        self.conv1d = layers.Conv1D(filters=config.hidden_size, kernel_size=(config.input_frames), activation='relu')
        self.dense2 = layers.Dense(config.window_size * config.output_channels)
        self.reshape = layers.Reshape((config.window_size, config.output_channels))

    def call(self, input_tensor, training=False):
        x = self.flatten(input_tensor)
        x = self.dense1(x)
        x = self.conv1d(x)
        x = self.dense2(x)
        x = self.reshape(x)
        return x

class CNNModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.flatten = layers.Flatten()
        self.conv1d =layers.Conv2D(filters=16, kernel_size=(config.input_frames, 1), activation='relu')
        self.dense1 = layers.Dense(config.hidden_size, activation="relu")
        self.dense2 = layers.Dense(config.window_size * config.output_channels)
        self.reshape = layers.Reshape((config.window_size, config.output_channels))

    def call(self, input_tensor, training=False):
        x = self.conv1d(input_tensor)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)
        return x


class RNNModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.flatten = layers.Reshape((config.input_frames, config.window_size * config.input_channels))
        self.dense1 = layers.Dense(config.hidden_size, activation="relu")
        self.gru = tf.keras.layers.GRU(config.hidden_size, return_state=True, activation='relu', dropout=config.dropout_rate)
        self.dense2 = layers.Dense(config.window_size * config.output_channels)
        self.reshape = layers.Reshape((config.window_size, config.output_channels))

    def call(self, input_tensor, training=False):
        x = self.flatten(input_tensor)
        x = self.dense1(x)
        output, final_state = self.gru(x, initial_state=None, training=training)
        x = tf.concat((output, final_state), axis=-1)
        x = self.dense2(x)
        x = self.reshape(x)
        return x
