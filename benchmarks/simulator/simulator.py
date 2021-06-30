import collections
import multiprocessing
import numpy as np
import subprocess
import tensorflow as tf
import time

from config import num_input_frames, window_size, seed
from config import temp_ratio, spat_ratio, barrier_sample_weight


def retrieve_data(input_file, pot_scalar=10.0):
    """
    Input: Text file for raw simulation and potential scalar
    Output: Python DefaultDict containing the simulation data

    """

    data = collections.defaultdict(list)

    #Read from raw file and fill data
    with tf.io.gfile.GFile(input_file) as f:
        for line in f:
            for key in ["timestamp", "params", "psi_re", "psi_im", "pot"]:
                if line.startswith(key):
                    data[key].append([float(x) for x in line.split()[1:]]
                                     if key != "timestamp" else float(line.split()[1]))

    #Convert each list to numpy array
    for key in ["timestamp", "params", "psi_re", "psi_im", "pot"]:
        data[key] = np.array(data[key])

    #Normalize Potential Values
    data["pot"] /= pot_scalar

    return data

def create_float_feature(values):
    """ Convert list of float values to tf train Feature"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def example_generator(data):
    """Takes in a raw file and converts it into tfrecords"""

    features = np.stack([data["psi_re"], data["psi_im"], data["pot"]], axis=-1)
    L = len(data["pot"][0])

    #Satisfying Boundary Condition
    features = np.hstack((features, features[:, :window_size, :]))

    #Temporal Sample Indices
    temp_sample = np.random.choice(features.shape[0] - num_input_frames - 1,
                                   int((features.shape[0] - num_input_frames - 1) * temp_ratio),
                                   replace=False
                                  )

    #Indices where potential exists
    v_tmp = np.where(data["pot"][0] != data["pot"][0][0])[0]
    v_start, v_end = v_tmp[[0, -1]] if len(v_tmp) > 1 else (-1, -1)


    #Spatial Sampling Probabilities
    spat_sample_ratio = np.ones(L)
    if v_start > -1:
        spat_sample_ratio[np.arange(v_start - window_size + 1, v_end + 1)] = barrier_sample_weight
    spat_sample_ratio /= spat_sample_ratio.sum()

    #Create Windows from Temporal and Spatial Sampled Indices
    for i in temp_sample:
        spat_sample = np.random.choice(L, int(L * spat_ratio), p=spat_sample_ratio, replace=False)
        for j in spat_sample:
            #Create tf examples from a set of windows
            tf_example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "feature":create_float_feature(features[i:i+num_input_frames+1, j:j+window_size].reshape(-1))
                    }
                )
            )
            yield tf_example
    return

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

    #Run Single Simulation
    start_time = time.time()
    subprocess.call(['./simulator.x',
                         params['X0'],
                         params['S0'],
                         params['E0'],
                         params['BH'],
                         params['BW'],
                         '0.0',
                         'tmp/' + '_'.join([params['X0'],params['S0'],params['E0'],params['BH'],params['BW']]) + '.txt'],
                         stdout=subprocess.PIPE)
    time_stamp1 = time.time()

    #Read the file
    data=retrieve_data('tmp/' + '_'.join([params['X0'],params['S0'],params['E0'],params['BH'],params['BW']]) + '.txt')
    time_stamp2 = time.time()

    #Create windows
    windows = list(example_generator(data))
    time_stamp3 = time.time()

    # single_simulation = run_single_simulation(params)
    # training_examples = chose_training_examples(single_simulation, params)
    # save('/tmp/...')  # You can save to the temp

    delta1 = time_stamp1 - start_time
    delta2 = time_stamp2 - time_stamp1
    delta3 = time_stamp3 - time_stamp2
    print('*** simulation {:.3f}, read {:.3f}, windowing {:.3f}'.format(delta1, delta2, delta3))

    return {'status': 'DONE', 'params': params, 'path': 'tmp/'}


def func(x):
    return x+1
