"""Download datasets."""
import collections

import numpy as np
import tensorflow as tf
import glob


def get_training_dataset(config):
    """Get training dataset.

    Args:
        config (global_config.Config): config object with training parameters

    Returns:
        dataset: tensorflow Dataset object for training data
    """
    try:
        dataset = iter(_create_tensorflow_dataset(config))
    except Exception:
        raise Exception('Dataset path is incorrect!')

    return dataset


def get_validation_dataset(config):
    """Get validation dataset.

    Args:
        config (global_config.Config): config object with training parameters

    Returns:
        test_data (list): array containing validation data
    """
    test_data = []
    files = glob.glob(config.validation_files)
    for f in files:
        test_data.append(_retrieve_validation_data(f))

    return test_data


def _create_tensorflow_dataset(config):
    """Create dataset from a config object.

    Args:
        config (global_config.Config): config object with training parameters

    Returns:
        dataset: Tensorflow dataset object
    """
    seed = config.seed
    qdml_tfrecords = config.training_files
    batch_size = config.train_batch

    def _decode_record(record, name_to_features):
        """Decode a record to a training example."""
        example = tf.io.parse_single_example(record,
                                             name_to_features)
        example = tf.reshape(example["feature"],
                             (config.input_frames + 1, config.window_size, 3))
        feature = example[:-1, :, :config.input_channels]
        label = example[-1, :, :config.output_channels]
        return feature, label

    name_to_features = {
        "feature": tf.io.FixedLenFeature(
            [np.product((config.input_frames + 1, config.window_size, 3))],
            tf.float32
        )
    }

    # Read all file paths
    input_files = tf.io.gfile.glob(qdml_tfrecords)

    # Create and Process Dataset
    d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
    d = d.repeat()
    d = d.shuffle(buffer_size=len(input_files),
                  seed=seed,
                  reshuffle_each_iteration=True)
    d = d.interleave(tf.data.TFRecordDataset,
                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d = d.shuffle(buffer_size=5000, seed=seed, reshuffle_each_iteration=True)
    d = d.map(lambda record: _decode_record(record, name_to_features),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d = d.batch(batch_size)
    dataset = d

    # Return dataset with batches of windows and labels
    return dataset


def _retrieve_validation_data(file_name, pot_scalar=10.0):
    """Create dictionary from raw simulation text file.

    Args:
        file_name (str): path to text file
        pot_scalar (float): variable to scale potential
    Returns:
        data (defaultdict): dictionary containing simulation data
    """
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
