"""Evaluate neural network."""
import logging
import numpy as np
import tensorflow as tf

import emulator.utils


def evaluate(model, config, validation_data):
    """Evaluate the model.

    Args:
        model (tensorflow.keras.Model): neural network
        config
        validation_data

    Returns:
        (float): evaluation score
    """
    # TODO: Update Docstrings

    avg_correlations = []

    for i, data in enumerate(validation_data):
        logging.info('Evaluating dataset {}'.format(i))
        if i >= 1:  # TODO: Remove me!!!
            break
        result = get_result(model, config,data)
        avg_correlations.append(get_avg_corr(config, data, result))

    score = np.average(avg_correlations)
    return score


def get_result(model, config, data, skip_frames=0, frames=400, normalization=False):
    """Calculates resulting simulation from model"""
    # TODO: Update Docstrings
    # TODO: Clean the code. Flake8 should give zero warnings
    features = tf.stack((tf.constant(data["psi_re"], dtype=tf.float32),
                         tf.constant(data["psi_im"], dtype=tf.float32),
                         tf.constant(data["pot"], dtype=tf.float32)), axis=-1)

    weights = emulator.utils.gaussian(np.arange(config.window_size), config.window_size // 2, 2.8).reshape(config.window_size, 1, 1, 1)
    weights = tf.constant(weights, dtype=tf.float32)
    sum_weights = tf.reduce_sum(weights)
    feature = features[skip_frames:skip_frames + config.input_frames]
    pot = feature[:1, :, 2:]
    result = []

    # Predict target frame and concatenate it to use for next prediction
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


def get_avg_corr(config, data, result, skip_frames=0, frames=400):
    """Calculate Normalized Correlation between result and true simulation for every frame"""
    # TODO: Update Docstrings
    # TODO: Refactor the code. Flake8 should give zero warnings
    # TODO: Note that skip_frames is not used. Remove?
    target = data["psi_re"][config.input_frames:config.input_frames + frames] + 1j * data["psi_im"][
                                                                                     config.input_frames: config.input_frames + frames]
    result = result[:, :, 0].numpy() + 1j * result[:, :, 1].numpy()
    corr = [np.abs(np.correlate(target[i], result[i]) / (np.linalg.norm(target[i]) * np.linalg.norm(result[i]))) for i
            in range(len(target))]
    return np.average(np.array(corr))
