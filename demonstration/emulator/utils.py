import os
import random
import math
import time
import datetime
import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from tensorflow import keras
from tensorflow.keras import layers
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class Config:
    def __init__(self, **kwargs):
        """ Configuration Settings for Training and Models"""
        
        #Read/write Settings
        self.qdml_tfrecords = "../datasets/demo/tfrecords/*"
        self.model_name = "demo"
        self.models_dir = "../models/"
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


def retrieve_data(file_name, pot_scalar):
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


def get_loss(config, data, result, skip_frames=0, frames=400):
    """Calculate Mean Absolute Error between result and true simulation for every frame"""
    mae = tf.keras.losses.MeanAbsoluteError()
    target = np.stack([data["psi_re"], data["psi_im"]], axis=-1)
    loss = [mae(result[i], target[i + skip_frames + config.input_frames]).numpy() for i in range(frames)]
    return np.array(loss)

def get_corr(config, data, result, skip_frames=0, frames=400):
    """Calculate Normalized Correlation between result and true simulation for every frame"""
    target = data["psi_re"][config.input_frames:config.input_frames+frames] + 1j * data["psi_im"][config.input_frames : config.input_frames + frames]
    result = result[:, :, 0].numpy() + 1j * result[:, :, 1].numpy()
    corr = [np.abs(np.correlate(target[i], result[i]) / (np.linalg.norm(target[i]) * np.linalg.norm(result[i]))) for i in range(len(target))]
    return np.array(corr)


def build_anime(config, data, result, pot_scalar, skip_frames=0, frames=400):
    """Build the animation"""
    plt.style.use('seaborn-pastel')
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title('Real(Psi)')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title('Imaginary(Psi)')
    plt.close()
    for ax in [ax1, ax2]:
        ax.set_xlim(0, data["params"][0][1])
        ax.set_ylim(-1, 1)
        ax.grid(False)
        
    lines1 = [ax1.plot([], [], lw=2, color="gray")[0], ax1.plot([], [], lw=3)[0], ax1.plot([], [], lw=3)[0]]
    lines2 = [ax2.plot([], [], lw=2, color="gray")[0], ax2.plot([], [], lw=3)[0], ax2.plot([], [], lw=3)[0]]
    time_text = ax1.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    def init():
        for line in lines1 + lines2:
            line.set_data([], [])
        time_text.set_text('time = %.3f' % 0)
        return lines1 + lines2 + [time_text]

    x = np.linspace(0, data["params"][0][1], int(data["params"][0][0]))
    pot = data["pot"][0]*pot_scalar/10
    def animate(i):
        re_pred = result[i, :, 0].numpy()
        im_pred = result[i, :, 1].numpy()
        re_true = np.array(data["psi_re"][i + skip_frames + config.input_frames])
        im_true = np.array(data["psi_im"][i + skip_frames + config.input_frames])
        lines1[1].set_data(x, re_true)
        lines1[2].set_data(x, re_pred)
        lines1[0].set_data(x, pot)
        lines2[1].set_data(x, im_true)
        lines2[2].set_data(x, im_pred)
        lines2[0].set_data(x, pot)
        time_text.set_text('time = %.4f' % data["timestamp"][i + skip_frames + config.input_frames])
        return lines1 + lines2 + [time_text]

    return FuncAnimation(fig, animate, init_func=init, frames=frames, interval=50, blit=True)

def show_animation(config, model, ckpt, test_file):
    """ Display Animation """
    pot_scalar = 10.
    if ckpt.split('/')[2][:4] == 'full':
        pot_scalar = 14.
    
    model = model(config)
    model.load_weights(ckpt)

    data = retrieve_data(test_file, pot_scalar)
    result = get_result(model, config, data)

    anim = build_anime(config, data, result, pot_scalar)

    return anim.to_html5_video()

def evaluate(config, ckpts_list, test_files, output, pot_scalar=10.0, total_frames=400):
    """Calculate Loss and Correlation with all test files for multiple models"""
    files = sorted(tf.io.gfile.glob(test_files))
    ckpt_names = [x[1].split('/')[-3] for x in ckpts_list]

    all_loss = np.zeros((len(ckpts_list), len(files), total_frames))
    all_corr = np.zeros((len(ckpts_list), len(files), total_frames))
    
    start_time = time.time()
    n_processed = 0
    for j, file in enumerate(files):
        data = retrieve_data(file, pot_scalar)
        for i, (model, ckpt) in enumerate(ckpts_list):
            model.load_weights(ckpt)
            result = get_result(model, config, data)
            loss = get_loss(config, data, result)
            corr = get_corr(config, data, result)
            all_loss[i, j, :] = loss
            all_corr[i, j, :] = corr.reshape(total_frames)
            
            elapsed = time.time() - start_time
            n_processed += 1
            print("Processed {:}/{:} ({:.1f}%), ELAPSED: {:}, ETA: {:}".format(
                n_processed, 
                len(ckpts_list) * len(files), 
                100.0 * n_processed / len(ckpts_list) / len(files), 
                get_readable_time(int(elapsed)),
                get_readable_time(int((len(ckpts_list) * len(files) - n_processed) / (n_processed / elapsed)))
                )
            )

    np.savez(output, files=files, ckpts=ckpt_names, all_loss=all_loss, all_corr=all_corr)




