"""Neural network training."""
import tensorflow


def train(model, x, y, epochs=10, batch_size=32):
    """Train neural network.

    Args:
        model (tensorflow.keras.Model): model to be trained
        x (numpy.array): training data
        y (numpy.array): training labels
        epochs (int): number of epochs
        batch_size (int): batch size

    Returns:
        (tensorflow.keras.Model): trained model
    """
    # TODO: Update the training procedure.

    model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
    )
