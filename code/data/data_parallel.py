import argparse
import subprocess
import itertools
import multiprocessing
import os
import random
import time
import numpy as np
from numpy import prod
from collections import defaultdict


class Configs:
    def __init__(self):

        base_dir = "/home1/yaoyu/scratch/qdml/datasets/barrier_E0_1.0to9.0_BH_1.0to14_BW_7.0/"
        train_dir = os.path.join(base_dir, "train")
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        test_dir = os.path.join(base_dir, "test")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        self.num_processes = 16
        self.X0_list = [10.0, 40.0, 70.0]  # list(np.arange(10.0, 95.0, 10.0))
        self.S0_list = list(np.arange(1.0, 4.1, 0.5))
        self.E0_list = list(np.arange(1.0, 9.1, 1.0))
        self.BH_list = list(np.arange(1.0, 14 + 0.1, 1.0))
        self.BW_list = [7.0]
        self.EH_list = [0.0]
        self.total = prod(
            list(map(len, (self.X0_list, self.S0_list, self.E0_list, self.BH_list, self.BW_list, self.EH_list))))
        self.base_dir = base_dir
        self.train_dir = train_dir
        self.test_dir = test_dir


def write_examples(job_id, args):
    """A single process creating and writing out pre-processed examples."""

    def log(*args):
        msg = " ".join(map(str, args))
        print("Job {}:".format(job_id), msg)

    log("Writing qd simulation results...")
    start_time = time.time()
    total = np.sum(np.arange(args.total) % args.num_processes == job_id)
    n_written = 0
    for num, (X0, S0, E0, BH, BW, EH) in enumerate(
            itertools.product(args.X0_list, args.S0_list, args.E0_list, args.BH_list, args.BW_list, args.EH_list)):
        if num % args.num_processes == job_id:
            (X0, S0, E0, BH, BW, EH) = ("{:.1f}".format(x) for x in (X0, S0, E0, BH, BW, EH))
            result = subprocess.call(['/home1/yaoyu/yaoyu/qdml/data/qd1', X0, S0, E0, BH, BW, EH, args.train_dir + "/" +
                                      '_'.join([X0, S0, E0, BH, BW, EH]) + '.txt'], stdout=subprocess.PIPE)
            elapsed = time.time() - start_time
            n_written += 1
            log("processed {:}/{:} files ({:.1f}%), ELAPSED: {:}s, ETA: {:}s".format(
                n_written, total, 100.0 * n_written / total, int(elapsed),
                int((total - n_written) / (n_written / elapsed))))
    log("Done!")


def main():
    args = Configs()
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
