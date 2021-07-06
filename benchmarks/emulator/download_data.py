"""Download data."""
import numpy
import tensorflow

import global_config


if __name__ == '__main__':
    """Download data."""
    (x, y), (xt, yt) = tensorflow.keras.datasets.cifar10.load_data()

    numpy.save(global_config.train_data_path, x)
    numpy.save(global_config.train_labels_path, y)
    numpy.save(global_config.valid_data_path, xt)
    numpy.save(global_config.valid_labels_path, yt)
