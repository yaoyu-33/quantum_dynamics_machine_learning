"""Download datasets."""
import os
import numpy as np
import tensorflow as tf
import glob
import emulator.utils


def get_train_data(config):
    """Get training dataset.

    Args:
        data_path (str): training datafile path
        labels_path (str): training labels path

    Returns:
        None
    """
    try:
        dataset=iter(emulator.utils.get_dataset(config))
    except Exception as e: 
        print(e)

    return dataset


def get_test_data(config):
    """Get validation dataset.

    Args:
        data_path (str): validation datafile path
        labels_path (str): validation labels path

    Returns:
        None
    """
    test_data = []
    files = glob.glob("../../datasets/test/full_test_E0_0.0to9.0_BH_0.0to14.0/*.txt")
    for f in files:
        test_data.append(emulator.utils.retrieve_data(f))


    # # TODO: Load the emulator training data (e.g., from a file)
    # # TODO: Return an exception if you can not find the file
    # # TODO: Use the file lock (see the example below)

    # with filelock.FileLock(os.path.expanduser("~/.emulator_data.lock")):
    #     x = numpy.load(data_path)
    #     y = numpy.load(labels_path)

    # # Normalize
    # x = x.astype('float32')/255.0

    return test_data
