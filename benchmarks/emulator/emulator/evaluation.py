"""Evaluate neural network."""
import tensorflow as tf
import numpy as np
import emulator.utils


def evaluate(model, config, test_data):
    """Evaluate the model.

    Args:
        model (tensorflow.keras.Model): neural netowrk
        x (numpy.array): validation data
        y (numpy.array): validation labels

    Returns:
        (float): evaluation score
    """
    # TODO: Change into our procedure (generate 400 steps, check correlation)

    avg_correlations = []

    for data in test_data:
        result = emulator.utils.get_result(model,config,data)
        avg_correlations.append(emulator.utils.get_avg_corr(config, data, result))

    score = np.average(avg_correlations)
    return score
