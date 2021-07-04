"""Load libraries and config variables."""
import collections
import numpy as np
import os
import logging
import subprocess
import tensorflow as tf
import time
from global_config import num_input_frames, window_size, temp_folder, save_folder
from global_config import temp_ratio, spat_ratio, b_weight


def retrieve_data(input_file, pot_scalar=10.0):
    """Parse simulation text file.

    Args:
        input_file: Text file for raw simulation
        pot_scalar: potential scalar

    Returns:
        Defaultdict: dictionary (simulation data)
    """
    data = collections.defaultdict(list)

    # Read from raw file and fill data
    with tf.io.gfile.GFile(input_file) as f:
        for line in f:
            for key in ["timestamp", "params", "psi_re", "psi_im", "pot"]:
                if line.startswith(key):
                    if key != 'timestamp':
                        data[key].append([float(x) for x in line.split()[1:]])
                    else:
                        data[key].append(float(line.split()[1]))

    # Convert each list to numpy array
    for key in ["timestamp", "params", "psi_re", "psi_im", "pot"]:
        data[key] = np.array(data[key])

    # Normalize Potential Values
    data["pot"] /= pot_scalar

    return data


def create_float_feature(values):
    """Convert list of float values to tf train Feature.

    Args:
        values: list of features
    Returns:
        tf.train.Feature: Tensorflow Feature
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def example_generator(data):
    """Create a training example from raw data.

    Args:
        data: parsed simulation dictionary
    Returns:
    """
    features = np.stack([data["psi_re"], data["psi_im"], data["pot"]], axis=-1)
    L = len(data["pot"][0])

    # Satisfying Boundary Condition
    features = np.hstack((features, features[:, :window_size, :]))

    # Temporal Sample Indices
    temp_sample = np.random.choice(
        features.shape[0] - num_input_frames - 1,
        int((features.shape[0] - num_input_frames - 1) * temp_ratio),
        replace=False
    )

    # Indices where potential exists
    v_tmp = np.where(data["pot"][0] != data["pot"][0][0])[0]
    v_start, v_end = v_tmp[[0, -1]] if len(v_tmp) > 1 else (-1, -1)

    # Spatial Sampling Probabilities
    sp_ratio = np.ones(L)
    if v_start > -1:
        sp_ratio[np.arange(v_start - window_size + 1, v_end + 1)] = b_weight
    sp_ratio /= sp_ratio.sum()

    # Create Windows from Temporal and Spatial Sampled Indices
    for i in temp_sample:
        spat_sample = np.random.choice(L,
                                       int(L * spat_ratio),
                                       p=sp_ratio,
                                       replace=False)

        for j in spat_sample:
            # Create tf examples from a set of windows
            r_range = i + num_input_frames + 1
            im_range = j + window_size
            tf_example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "feature": create_float_feature(
                            features[i: r_range, j: im_range].reshape(-1))
                    }
                )
            )
            yield tf_example


def simulate(params):
    """Run one simulation.

    Args:
        X0 (float): wave packet position (center)
        S0 (float): wave packet spread
        E0 (float): wave packet energy
        BH (float): barier hight
        BW (float): barier width

    Returns:
        array (numpy.array): 2D array (number_of_frames, spatial_width)
    """
    # Run Single Simulation
    start_time = time.time()
    X0, S0, E0 = params['X0'], params['S0'], params['E0']
    BH, BW = params['BH'], params['BW']
    filename = temp_folder + '_'.join([X0, S0, E0, BH, BW]) + '.txt'
    subprocess.call([
        './simulator.x',
        X0, S0, E0, BH, BW, '0.0',
        filename
        ], stdout=subprocess.PIPE)
    time_stamp1 = time.time()

    # Read the file
    data = retrieve_data(filename)
    time_stamp2 = time.time()

    # Create windows
    windows = list(example_generator(data))
    time_stamp3 = time.time()

    # Remove Intermediate File
    os.remove(filename)

    # Save generated windows
    save_path = os.path.join(
        save_folder, 'windows' + '_'.join([X0, S0, E0, BH, BW]) + '.npz')
    np.savez(save_path, windows)
    time_stamp4 = time.time()

    # Logs
    delta1 = time_stamp1 - start_time
    delta2 = time_stamp2 - time_stamp1
    delta3 = time_stamp3 - time_stamp2
    delta4 = time_stamp4 - time_stamp3
    logging.info(
        'simulation {:.3f}, reading {:.3f}, windowing {:.3f}, saving {:.3f}'
        .format(delta1, delta2, delta3, delta4))

    return {'status': 'DONE',
            'params': params,
            'path': '...',
            'win_len': len(windows)}
