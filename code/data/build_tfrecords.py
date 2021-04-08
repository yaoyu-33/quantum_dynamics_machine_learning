import argparse
import multiprocessing
import os
import random
import time
import tensorflow as tf
import numpy as np
from collections import defaultdict


def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def create_float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


class ExampleWriter(object):
    """Writes pre-training examples to disk."""

    def __init__(self, job_id, output_dir, num_jobs, args, num_out_files=100):
        self._writers = []
        for i in range(num_out_files):
            if i % num_jobs == job_id:
                output_fname = os.path.join(
                    output_dir, "qdml.tfrecord-{:}-of-{:}".format(
                        i, num_out_files))
                self._writers.append(tf.io.TFRecordWriter(output_fname))
        self.n_written = 0
        self.barrier_sample_weight = args.barrier_sample_weight
        self.args = args

    def example_generator(
            self, data, num_input_frames=4, window_size=23,
            temp_ratio=0.9, spat_ratio=0.1):
        features = np.stack([data["psi_re"], data["psi_im"], data["pot"]], axis=-1)
        features = np.hstack((features, features[:, :window_size, :]))
        # features = tf.stack((tf.constant(data.psi_re), tf.constant(data.psi_im), tf.constant(data["pot"])), axis=-1)
        # features = tf.concat((features, features[:, :window_size, :]), axis=-2)
        L = len(data["pot"][0])
        temp_sample = np.random.choice(
            features.shape[0] - num_input_frames - 1,
            int((features.shape[0] - num_input_frames - 1) * temp_ratio),
            replace=False)
        v_start, v_end = np.where(data["pot"][0] > 1e-10)[0][[0, -1]]
        spat_sample_ratio = np.ones(L)
        spat_sample_ratio[np.arange(v_start - window_size + 1, v_end + 1)] = self.barrier_sample_weight
        spat_sample_ratio /= spat_sample_ratio.sum()
        for i in temp_sample:
            spat_sample = np.random.choice(
                L, int(L * spat_ratio), p=spat_sample_ratio, replace=False)
            for j in spat_sample:
                tf_example = tf.train.Example(features=tf.train.Features(feature={
                    "feature": create_float_feature(
                        features[i:i + num_input_frames + 1, j:j + window_size].reshape(-1)),
                }))
                yield tf_example
        return

    def write_examples(self, input_file):
        """Writes out examples from the provided input file."""
        data = defaultdict(list)
        with tf.io.gfile.GFile(input_file) as f:
            for line in f:
                for key in ["timestamp", "params", "psi_re", "psi_im", "pot"]:
                    if line.startswith(key):
                        data[key].append([float(x) for x in line.split()[1:]]
                                         if key != "timestamp" else float(line.split()[1]))
        for key in ["timestamp", "params", "psi_re", "psi_im", "pot"]:
            data[key] = np.array(data[key])
        data["pot"] /= self.args.pot_scaler
        example_iterator = iter(self.example_generator(
            data, temp_ratio=self.args.temp_ratio, spat_ratio=self.args.spat_ratio))
        for example in example_iterator:
            self._writers[self.n_written % len(self._writers)].write(
                example.SerializeToString())
            self.n_written += 1

    def finish(self):
        for writer in self._writers:
            writer.close()


def write_examples(job_id, args):
    """A single process creating and writing out pre-processed examples."""

    def log(*args):
        msg = " ".join(map(str, args))
        print("Job {}:".format(job_id), msg, flush=True)

    log("Creating example writer")
    example_writer = ExampleWriter(
        job_id=job_id,
        output_dir=args.output_dir,
        num_jobs=args.num_processes,
        num_out_files=args.num_out_files,
        args=args
    )
    log("Writing tf examples")
    fnames = sorted(tf.io.gfile.listdir(args.input_dir))
    fnames = [f for (i, f) in enumerate(fnames)
              if i % args.num_processes == job_id]
    random.shuffle(fnames)
    start_time = time.time()
    for file_no, fname in enumerate(fnames):
        if file_no > 0:
            elapsed = time.time() - start_time
            log("processed {:}/{:} files ({:.1f}%), ELAPSED: {:}s, ETA: {:}s, "
                "{:} examples written".format(
                file_no, len(fnames), 100.0 * file_no / len(fnames), int(elapsed),
                int((len(fnames) - file_no) / (file_no / elapsed)),
                example_writer.n_written))
        example_writer.write_examples(os.path.join(args.input_dir, fname))
    example_writer.finish()
    log("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_processes", type=int, default=16)
    parser.add_argument("--num_out_files", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--temp_ratio", type=float, default=0.9)
    parser.add_argument("--spat_ratio", type=float, default=0.1)
    parser.add_argument("--barrier_sample_weight", type=float, default=48.0)
    parser.add_argument("--pot_scaler", type=float, default=10.)
    args = parser.parse_args()
    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.num_processes == 1:
        write_examples(0, args)
    else:
        jobs = []
        for i in range(args.num_processes):
            job = multiprocessing.Process(target=write_examples, args=(i, args))
            jobs.append(job)
            job.start()
        for job in jobs:
            job.join()


if __name__ == "__main__":
    main()
