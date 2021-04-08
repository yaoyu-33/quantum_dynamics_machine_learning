import os
import random
import math
import time
import datetime
import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from tensorflow import keras
from tensorflow.keras import layers
from matplotlib.animation import FuncAnimation

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def retrieve_data(file_name, pot_scalar):
    data = collections.defaultdict(list)
    f = open(file_name, "r")
    for line in f:
        for key in ["timestamp", "params", "psi_re", "psi_im", "pot"]:
            if line.startswith(key):
                data[key].append([float(x) for x in line.split()[1:]]
                    if key != "timestamp" else float(line.split()[1]))

    for key in ["timestamp", "params", "psi_re", "psi_im", "pot"]:
        data[key] = np.array(data[key])
    data["pot"] /= pot_scalar
    return data


def get_result(model, config, data, skip_frames=0, frames=400, normalization=False):
    features = tf.stack((tf.constant(data["psi_re"], dtype=tf.float32), 
                         tf.constant(data["psi_im"], dtype=tf.float32), 
                         tf.constant(data["pot"], dtype=tf.float32)), axis=-1)

    weights = gaussian(np.arange(config.window_size), config.window_size // 2, 2.8).reshape(config.window_size, 1, 1, 1)
    weights = tf.constant(weights, dtype=tf.float32)
    sum_weights = tf.reduce_sum(weights)
    feature = features[skip_frames:skip_frames+config.input_frames]
    pot = feature[:1, :, 2:]
    result = []
    while len(result) < frames:
        input = tf.stack([tf.roll(feature, shift=-shift, axis=1) for shift in
                          range(-(config.window_size // 2), config.window_size // 2 + 1)], axis=-2)
        input = tf.transpose(input, [1, 0, 2, 3])

        pred = tf.expand_dims(model(input[:, :, :, :config.input_channels]), axis=0)
        pred = [tf.roll(pred, shift=shift, axis=1)[:, :, config.window_size // 2 + shift] for shift in
                range(-(config.window_size // 2), config.window_size // 2 + 1)]
        pred = tf.reduce_sum(tf.stack(pred, axis=0) * weights, axis=0) / sum_weights

        if normalization:
            scalar = tf.math.sqrt(tf.math.reduce_sum(pred * pred, axis=[1, 2], keepdims=True) * 50 / 512)
            pred /= scalar
        
        result.append(pred)
        feature = tf.concat([feature[1:], tf.concat((pred, pot), axis=-1)], axis=0)
    result = tf.concat(result, axis=0)
    return result


def get_loss(config, data, result, skip_frames=0, frames=400):
    mae = tf.keras.losses.MeanAbsoluteError()
    target = np.stack([data["psi_re"], data["psi_im"]], axis=-1)
    loss = [mae(result[i], target[i + skip_frames + config.input_frames]).numpy() for i in range(frames)]
    return np.array(loss)


def build_anime(config, data, result, skip_frames=0, frames=400):
    plt.style.use('seaborn-pastel')
    fig = plt.figure()
#     ax = plt.axes(xlim=(0, data["params"][0][1]), ylim=(-2, 2))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    for ax in [ax1, ax2]:
        ax.set_xlim(0, data["params"][0][1])
        ax.set_ylim(-1, 1)
        
    lines1 = [ax1.plot([], [], lw=1, color="gray")[0], ax1.plot([], [], lw=2)[0], ax1.plot([], [], lw=2)[0]]
    lines2 = [ax2.plot([], [], lw=1, color="gray")[0], ax2.plot([], [], lw=2)[0], ax2.plot([], [], lw=2)[0]]
    time_text = ax1.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    def init():
        for line in lines1 + lines2:
            line.set_data([], [])
        time_text.set_text('time = %.3f' % 0)
        return lines1 + lines2 + [time_text]

    x = np.linspace(0, data["params"][0][1], int(data["params"][0][0]))
    pot = data["pot"][0]
    def animate(i):
        re_pred = result[i, :, 0].numpy()
        im_pred = result[i, :, 1].numpy()
        re_true = np.array(data["psi_re"][i + skip_frames + config.input_frames])
        im_true = np.array(data["psi_im"][i + skip_frames + config.input_frames])
        lines1[1].set_data(x, re_true)
        lines1[2].set_data(x, re_pred)
        lines1[0].set_data(x, pot)
        lines2[1].set_data(x, im_true)
        lines2[2].set_data(x, im_pred)
        lines2[0].set_data(x, pot)
        time_text.set_text('time = %.4f' % data["timestamp"][i + skip_frames + config.input_frames])
        return lines1 + lines2 + [time_text]

    return FuncAnimation(fig, animate, init_func=init,
                         frames=frames, interval=50, blit=True)