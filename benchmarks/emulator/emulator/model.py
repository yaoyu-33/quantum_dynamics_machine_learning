"""Define the neural network architecture."""
import tensorflow


def build(hidden, dropout, lr, momentum):
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

    Conv2D = tensorflow.keras.layers.Conv2D
    MaxPooling2D = tensorflow.keras.layers.MaxPooling2D
    Flatten = tensorflow.keras.layers.Flatten
    Dense = tensorflow.keras.layers.Dense
    Dropout = tensorflow.keras.layers.Dropout
    BatchNormalization = tensorflow.keras.layers.BatchNormalization

    reg = tensorflow.keras.regularizers.L2
    init = tensorflow.keras.initializers.he_uniform()

    m = tensorflow.keras.models.Sequential()
    m.add(Conv2D(
        64, (3, 3), strides=1, padding="same", activation="elu",
        input_shape=(32, 32, 3),
        kernel_initializer=init, kernel_regularizer=reg(1e-4)))
    m.add(Conv2D(
        64, (3, 3), strides=1, padding="same", activation="elu",
        kernel_initializer=init, kernel_regularizer=reg(1e-4)))
    m.add(BatchNormalization())
    m.add(MaxPooling2D((2, 2)))
    m.add(Dropout(dropout))

    m.add(Conv2D(
        128, (3, 3), strides=1, padding="same", activation="elu",
        kernel_initializer=init, kernel_regularizer=reg(1e-4)))
    m.add(Conv2D(
        128, (3, 3), strides=1, padding="same", activation="elu",
        kernel_initializer=init, kernel_regularizer=reg(1e-4)))
    m.add(BatchNormalization())
    m.add(MaxPooling2D((2, 2)))
    m.add(Dropout(dropout))

    m.add(Conv2D(
        256, (3, 3), strides=1, padding="same", activation="elu",
        kernel_initializer=init, kernel_regularizer=reg(1e-4)))
    m.add(Conv2D(
        256, (3, 3), strides=1, padding="same", activation="elu",
        kernel_initializer=init, kernel_regularizer=reg(1e-4)))
    m.add(BatchNormalization())
    m.add(MaxPooling2D((2, 2)))
    m.add(Dropout(dropout))

    m.add(Conv2D(
        512, (3, 3), strides=1, padding="same", activation="elu",
        kernel_initializer=init, kernel_regularizer=reg(1e-4)))
    m.add(Conv2D(
        512, (3, 3), strides=1, padding="same", activation="elu",
        kernel_initializer=init, kernel_regularizer=reg(1e-4)))
    m.add(BatchNormalization())
    m.add(MaxPooling2D((2, 2)))
    m.add(Dropout(dropout))

    m.add(Flatten())
    m.add(Dense(hidden, activation="elu"))
    m.add(BatchNormalization())
    m.add(Dropout(0.5))
    m.add(Dense(10, activation="softmax"))

    m.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tensorflow.keras.optimizers.SGD(
            lr=lr, momentum=momentum),
        metrics=["accuracy"])

    return m
