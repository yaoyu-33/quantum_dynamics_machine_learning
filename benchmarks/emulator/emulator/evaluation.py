"""Evaluate neural network."""
import logging
import numpy as np
import tensorflow as tf

import emulator.utils


def evaluate(model, config, validation_data):
    """Evaluate the model.

    Args:
        model (tensorflow.keras.Model): neural network
        config (global_config.Config): config object with training parameters
        validation_data (list): array containing validation data

    Returns:
        score (float): evaluation score
    """
    avg_correlations = []

    for i, data in enumerate(validation_data):
        logging.info('Evaluating dataset {}'.format(i))
        result = get_result(model, config, data)
        avg_correlations.append(get_avg_corr(config, data, result))

    score = np.average(avg_correlations)
    return score


def get_result(model, config, data, frames=400, normalization=False):
    """Calculate resulting simulation from model.

    Args:
        model (tensorflow.keras.Model): neural network
        config (global_config.Config): config object with training parameters
        data (defaultdict): dictionary containing simulation data
        frames (int): number of frames
        normalization (bool): boolean to normalize predictions
    Returns:
        result (tensor): tensor containing model prediction
    """
    features = tf.stack((tf.constant(data["psi_re"], dtype=tf.float32),
                         tf.constant(data["psi_im"], dtype=tf.float32),
                         tf.constant(data["pot"], dtype=tf.float32)), axis=-1)
    w_size = config.window_size
    weights = emulator.utils.gaussian(np.arange(w_size),
                                      w_size // 2, 2.8)
    weights = weights.reshape(w_size, 1, 1, 1)
    weights = tf.constant(weights, dtype=tf.float32)
    sum_weights = tf.reduce_sum(weights)
    feature = features[:config.input_frames]
    pot = feature[:1, :, 2:]
    result = []

    # Predict target frame and concatenate it to use for next prediction
    while len(result) < frames:
        shift_range = range(-(w_size // 2),
                            w_size // 2 + 1)
        input = tf.stack(
            [tf.roll(feature, shift=-shift, axis=1) for shift in shift_range],
            axis=-2
        )
        input = tf.transpose(input, [1, 0, 2, 3])
        pred = tf.expand_dims(model(input[:, :, :, :config.input_channels]),
                              axis=0)
        pred = [tf.roll(pred, shift=shift, axis=1)[:, :, w_size // 2 + shift]
                for shift in shift_range]
        pred = tf.reduce_sum(tf.stack(pred, axis=0) * weights, axis=0)
        pred = pred / sum_weights

        if normalization:
            scalar = tf.math.sqrt(tf.math.reduce_sum(pred * pred, axis=[1, 2],
                                                     keepdims=True) * 50 / 512)
            pred /= scalar

        result.append(pred)
        feature = tf.concat([feature[1:], tf.concat((pred, pot), axis=-1)],
                            axis=0)

    result = tf.concat(result, axis=0)
    return result


def get_avg_corr(config, data, result, frames=400):
    """Calculate Normalized Correlation.

    Args:
        config (global_config.Config): config object with training parameters
        data (defaultdict): dictionary containing simulation data
        result (tensor): tensor containing model prediction
        frames (int): number of frames
    Returns:
        avg_corr (float): average correlation over all frames
    """
    real = data["psi_re"][config.input_frames: config.input_frames + frames]
    imag = data["psi_im"][config.input_frames: config.input_frames + frames]
    target = real + 1j * imag
    result = result[:, :, 0].numpy() + 1j * result[:, :, 1].numpy()
    corr = [np.abs(
        np.correlate(target[i], result[i]) /
        (np.linalg.norm(target[i]) * np.linalg.norm(result[i]))
        ) for i in range(len(target))]
    avg_corr = np.average(np.array(corr))
    return avg_corr
