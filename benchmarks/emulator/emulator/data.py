"""Download datasets."""
import os
import filelock
import tensorflow.keras.datasets


def get_train_data(data_path):
    """Get training dataset.

    Args:
        data_path (str): training datafile path

    Returns:
        None
    """
    # TODO: Load the emulator training data (e.g., from a file)
    # TODO: Return an exception if you can not find the file
    # TODO: Use the file lock (see the example below)

    with filelock.FileLock(os.path.expanduser("~/.emulator_data.lock")):
        (x, y), (_, __) = tensorflow.keras.datasets.mnist.load_data()

    # Normalize
    x = x.astype('float32')/255.0

    return x, y


def get_validation_data(data_path):
    """Get validation dataset.

    Args:
        data_path (str): training datafile path

    Returns:
        None
    """
    # TODO: Load the emulator training data (e.g., from a file)
    # TODO: Return an exception if you can not find the file
    # TODO: Use the file lock (see the example below)

    with filelock.FileLock(os.path.expanduser("~/.emulator_data.lock")):
        (_, __), (x, y) = tensorflow.keras.datasets.mnist.load_data()

    # Normalize
    x = x.astype('float32')/255.0

    return x, y
