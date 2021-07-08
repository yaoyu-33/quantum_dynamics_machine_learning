"""Download datasets."""
import collections
import logging

import numpy as np
import tensorflow as tf
import glob


def get_training_dataset(config):
    """Get training dataset.

    Args:
        data_path (str): training datafile path
        labels_path (str): training labels path

    Returns:
        None
    """
    # TODO: Update docstrings
    # TODO: Why there is try/except? If you catch an exception
    #   the dataset will be undefined. And you will return it. This is not good.
    try:
        dataset = iter(_create_tensoflow_dataset(config))
    except Exception as e:
        print(e)

    return dataset


def get_validation_dataset(config):
    """Get validation dataset.

    Args:
        data_path (str): validation datafile path
        labels_path (str): validation labels path

    Returns:
        None
    """
    # TODO: Update docstrings
    test_data = []
    files = glob.glob(config.validation_files)
    for f in files:
        test_data.append(_retrieve_validation_data(f))

    return test_data


def _create_tensoflow_dataset(config):
    """Creates dataset from a config object"""
    # TODO: Update docstrings
    # TODO: Clean the code. Flake8 should give zero warnings
    seed = config.seed
    qdml_tfrecords = config.training_files
    batch_size = config.train_batch

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example and returns a training window with its label"""
        example = tf.io.parse_single_example(record, name_to_features)
        example = tf.reshape(example["feature"], (config.input_frames + 1, config.window_size, 3))
        return example[:-1, :, :config.input_channels], example[-1, :, :config.output_channels]

    name_to_features = {
        "feature": tf.io.FixedLenFeature([np.product((config.input_frames + 1, config.window_size, 3))], tf.float32),
    }

    # Read all file paths
    input_files = tf.io.gfile.glob(qdml_tfrecords)

    # Create and Process Dataset
    d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
    d = d.repeat()
    d = d.shuffle(buffer_size=len(input_files), seed=seed, reshuffle_each_iteration=True)
    d = d.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d = d.shuffle(buffer_size=5000, seed=seed, reshuffle_each_iteration=True)
    d = d.map(lambda record: _decode_record(record, name_to_features), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d = d.batch(batch_size)

    # Return dataset with batches of windows and labels
    return d


def _retrieve_validation_data(file_name, pot_scalar=10.0):
    """
    Input: Text file for raw simulation and potential scalar
    Output: Python DefaultDict containing the simulation data

    """
    # TODO: Update docstrings
    data = collections.defaultdict(list)
    f = open(file_name, "r")
    for line in f:
        for key in ["timestamp", "params", "psi_re", "psi_im", "pot"]:
            if line.startswith(key):
                data[key].append(
                    [float(x) for x in line.split()[1:]]
                    if key != "timestamp" else float(line.split()[1])
                )

    for key in ["timestamp", "params", "psi_re", "psi_im", "pot"]:
        data[key] = np.array(data[key])
    data["pot"] /= pot_scalar

    return data
