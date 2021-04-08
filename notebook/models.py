# Import libraries
import os
import random
import math
import time
import datetime
import collections
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.animation as animation

from tensorflow import keras
from tensorflow.keras import layers
# from matplotlib.animation import FuncAnimation
# from optimization import create_optimizer
# from IPython.display import HTML
# from utils import *

class Config:
    def __init__(self, **kwargs):
        # read/write settings
        self.qdml_tfrecords = "../datasets/demo/tfrecords/*"
        self.model_name = "demo"
        self.models_dir = "../models/"
        self.log_name = None

        # training settings
        self.train_batch = 128
        self.input_frames = 4
        self.input_channels = 3  # 3 for full dataset
        self.output_channels = 2
        self.window_size = 23
        self.dropout_rate = 0.
        self.hidden_size = 69  # 69 for full dataset
        self.num_train_steps = 200000

        # optimization
        self.learning_rate = 1e-3
        self.lr_decay_power = 1.0
        self.weight_decay_rate = 0.01
        self.num_warmup_steps = 2000
        self.opt_beta_1 = 0.9
        self.opt_beta_2 = 0.999
        self.end_lr = 1e-6
        self.clip_norm = 1.0

        # misc
        self.num_threads = 4
        self.seed = 711
        self.logging_steps = 1000
        self.save_ckpt_steps = 1000000

        self.update(kwargs)        
        self.model_dir = os.path.join(self.models_dir, self.model_name)
        self.log_dir = os.path.join(self.model_dir, "logs")
        self.ckpts = os.path.join(self.model_dir, "ckpts")
    
    def update(self, kwargs):
        for k, v in kwargs.items():
            if v is not None:
                self.__dict__[k] = v


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
