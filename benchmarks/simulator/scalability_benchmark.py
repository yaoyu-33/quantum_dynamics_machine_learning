"""Load libraries."""
import argparse
import logging
import multiprocessing
import numpy as np
import simulator
import sys


if __name__ == '__main__':
    """Run the benchmark."""
    # Parse the input.
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='number of processes', type=int)
    parser.add_argument('-n', help='number of examples', type=int)
    parser.add_argument(
        '-v', '--verbose', help='verbosity level', type=int, default=0)
    args = parser.parse_args()

    # Logging
    logging_level = logging.INFO if args.verbose >= 1 else logging.CRITICAL
    logging.basicConfig(stream=sys.stdout, level=logging_level)

    # Prepare a list of simulation parameters.
    parameters = [
        {
            'X0': str(np.random.uniform(10.0, 70.0)),
            'S0': str(np.random.uniform(1.0, 4.0)),
            'E0': str(np.random.uniform(1.0, 9.0)),
            'BH': str(np.random.uniform(1.0, 14.0)),
            'BW': str(np.random.uniform(7.0, 8.0))
        } for _ in range(args.n)
    ]

    # Run p processes
    pool = multiprocessing.Pool(processes=args.p)
    pool.map(simulator.simulate, parameters)
