import os
import math
import time
import collections
import numpy as np
import tensorflow as tf

class Config:
    def __init__(self, **kwargs):
        """ Configuration Settings for Training and Models"""
        
        #Read/write Settings
        self.qdml_tfrecords = "../../datasets/demo/tfrecords/*"
        self.model_name = "demo"
        self.models_dir = "models/"
        self.test_dir = "../../datasets/test/full_test_E0_0.0to9.0_BH_0.0to14.0/"
        self.log_name = None

        #Training Settings
        self.train_batch = 128
        self.input_frames = 4
        self.input_channels = 3
        self.output_channels = 2
        self.window_size = 23
        self.dropout_rate = 0.
        self.hidden_size = 69
        self.num_train_steps = 900000 #5290000 for full dataset

        #Optimizer Settings
        self.learning_rate = 1e-3
        self.lr_decay_power = 1.0
        self.weight_decay_rate = 0.01
        self.num_warmup_steps = 9000 # 52900 for full dataset
        self.opt_beta_1 = 0.9
        self.opt_beta_2 = 0.999
        self.end_lr = 1e-6
        self.clip_norm = 1.0
        
        #Misc
        self.seed = 711
        self.logging_steps = 1000
        self.save_ckpt_steps = 1000000
        
        self.update(kwargs)        
        self.model_dir = os.path.join(self.models_dir, self.model_name)
        self.log_dir = os.path.join(self.model_dir, "logs")
        self.ckpts = os.path.join(self.model_dir, "ckpts")
    
    def update(self, kwargs):
        for k, v in kwargs.items():
            if v is not None:
                self.__dict__[k] = v

def gaussian(x, mu, sig):
    """ Gaussian Distribution """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def get_readable_time(elapsed):
    """ Convert time to a readable format """
    d, h, m, s = [int(x) for x in time.strftime("%d:%H:%M:%S", time.gmtime(elapsed)).split(':')]
    d -= 1
    return '{:2d}h{:2d}m{:2d}s'.format(24 * d + h, m, s)
 

def get_dataset(config):
    """Creates dataset from a config object"""
    
    seed = config.seed
    qdml_tfrecords = config.qdml_tfrecords
    batch_size = config.train_batch
    
    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example and returns a training window with its label"""
        example = tf.io.parse_single_example(record, name_to_features)
        example = tf.reshape(example["feature"], (config.input_frames + 1, config.window_size, 3))
        return example[:-1, :, :config.input_channels], example[-1, :, :config.output_channels]

    
    name_to_features = {
        "feature": tf.io.FixedLenFeature([np.product((config.input_frames + 1, config.window_size, 3))], tf.float32),
    }
    
    #Read all file paths
    input_files = tf.io.gfile.glob(qdml_tfrecords)
    
    #Create and Process Dataset
    d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
    d = d.repeat()
    d = d.shuffle(buffer_size=len(input_files), seed=seed, reshuffle_each_iteration=True)
    d = d.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d = d.shuffle(buffer_size=5000, seed=seed, reshuffle_each_iteration=True)
    d = d.map(lambda record: _decode_record(record, name_to_features), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d = d.batch(batch_size)
    
    #Return dataset with batches of windows and labels
    return d


def retrieve_data(file_name, pot_scalar = 10.0):
    """ 
    Input: Text file for raw simulation and potential scalar    
    Output: Python DefaultDict containing the simulation data

    """
    data = collections.defaultdict(list)
    f = open(file_name, "r")
    for line in f:
        for key in ["timestamp", "params", "psi_re", "psi_im", "pot"]:
            if line.startswith(key):
                data[key].append([float(x) for x in line.split()[1:]]
                    if key != "timestamp" else float(line.split()[1]))

    for key in ["timestamp", "params", "psi_re", "psi_im", "pot"]:
        data[key] = np.array(data[key])
    data["pot"] /= pot_scalar
    return data


def get_result(model, config, data, skip_frames=0, frames=400, normalization=False):
    """Calculates resulting simulation from model"""
    features = tf.stack((tf.constant(data["psi_re"], dtype=tf.float32), 
                         tf.constant(data["psi_im"], dtype=tf.float32), 
                         tf.constant(data["pot"], dtype=tf.float32)), axis=-1)

    
    weights = gaussian(np.arange(config.window_size), config.window_size // 2, 2.8).reshape(config.window_size, 1, 1, 1)
    weights = tf.constant(weights, dtype=tf.float32)
    sum_weights = tf.reduce_sum(weights)
    feature = features[skip_frames:skip_frames+config.input_frames]
    pot = feature[:1, :, 2:]
    result = []

    #Predict target frame and concatenate it to use for next prediction
    while len(result) < frames:
        input = tf.stack([tf.roll(feature, shift=-shift, axis=1) for shift in
                          range(-(config.window_size // 2), config.window_size // 2 + 1)], axis=-2)
        input = tf.transpose(input, [1, 0, 2, 3])
        pred = tf.expand_dims(model(input[:, :, :, :config.input_channels]), axis=0)
        pred = [tf.roll(pred, shift=shift, axis=1)[:, :, config.window_size // 2 + shift] for shift in
                range(-(config.window_size // 2), config.window_size // 2 + 1)]
        pred = tf.reduce_sum(tf.stack(pred, axis=0) * weights, axis=0) / sum_weights

        if normalization:
            scalar = tf.math.sqrt(tf.math.reduce_sum(pred * pred, axis=[1, 2], keepdims=True) * 50 / 512)
            pred /= scalar
        
        result.append(pred)
        feature = tf.concat([feature[1:], tf.concat((pred, pot), axis=-1)], axis=0)


    result = tf.concat(result, axis=0)
    return result

def get_avg_corr(config, data, result, skip_frames=0, frames=400):
    """Calculate Normalized Correlation between result and true simulation for every frame"""
    target = data["psi_re"][config.input_frames:config.input_frames+frames] + 1j * data["psi_im"][config.input_frames : config.input_frames + frames]
    result = result[:, :, 0].numpy() + 1j * result[:, :, 1].numpy()
    corr = [np.abs(np.correlate(target[i], result[i]) / (np.linalg.norm(target[i]) * np.linalg.norm(result[i]))) for i in range(len(target))]
    return np.average(np.array(corr))