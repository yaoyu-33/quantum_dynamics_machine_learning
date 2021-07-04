"""Define the neural network architecture."""
import tensorflow


def build(hidden=256, dropout=0.2, lr=0.05, momentum=0.9):
    """Build Neural Network.

    Args:
        hidden (int): number of hidden neurons
        dropout (float): dropout ratio
        lr (float): learning ratio for the optimizer
        momentum (float): momentum parameter for the SGD optimizer

    Returns:
        (tensoflow.keras.Model): compiled neural network.
    """
    # TODO: Define here our recurrent neural network
    # TODO: Update the arguments (parameters that we want to optimize by ray)

    model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Flatten(input_shape=(28, 28)),
        tensorflow.keras.layers.Dense(hidden, activation="relu"),
        tensorflow.keras.layers.Dropout(dropout),
        tensorflow.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tensorflow.keras.optimizers.SGD(
            lr=lr, momentum=momentum),
        metrics=["accuracy"])

    return model
