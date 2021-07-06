"""Evaluate neural network."""
import tensorflow


def evaluate(model, x, y):
    """Evaluate the model.

    Args:
        model (tensorflow.keras.Model): neural netowrk
        x (numpy.array): validation data
        y (numpy.array): validation labels

    Returns:
        (float): evaluation score
    """
    # TODO: Change into our procedure (generate 400 steps, check correlation)

    score = model.evaluate(x, y)[1]
    return score
