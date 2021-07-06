"""Download datasets."""
import os
import filelock
import numpy


def get_train_data(data_path, labels_path):
    """Get training dataset.

    Args:
        data_path (str): training datafile path
        labels_path (str): training labels path

    Returns:
        None
    """
    # TODO: Load the emulator training data (e.g., from a file)
    # TODO: Return an exception if you can not find the file
    # TODO: Use the file lock (see the example below)

    with filelock.FileLock(os.path.expanduser("~/.emulator_data.lock")):
        x = numpy.load(data_path)
        y = numpy.load(labels_path)

    # Normalize
    x = x.astype('float32')/255.0

    return x, y


def get_validation_data(data_path, labels_path):
    """Get validation dataset.

    Args:
        data_path (str): validation datafile path
        labels_path (str): validation labels path

    Returns:
        None
    """
    # TODO: Load the emulator training data (e.g., from a file)
    # TODO: Return an exception if you can not find the file
    # TODO: Use the file lock (see the example below)

    with filelock.FileLock(os.path.expanduser("~/.emulator_data.lock")):
        x = numpy.load(data_path)
        y = numpy.load(labels_path)

    # Normalize
    x = x.astype('float32')/255.0

    return x, y
