"""Load libraries."""
import argparse
import multiprocessing
import numpy as np
import simulator


"""Prapare a list of simulation parameters."""
parameters = [
    {
        'X0': str(np.random.uniform(10.0, 70.0)),
        'S0': str(np.random.uniform(1.0, 4.0)),
        'E0': str(np.random.uniform(1.0, 9.0)),
        'BH': str(np.random.uniform(1.0, 14.0)),
        'BW': str(np.random.uniform(7.0, 8.0))
    } for _ in range(12)
]


if __name__ == '__main__':
    """Parse the input."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        help='number of processes',
        type=int
    )
    parser.add_argument(
        '-p',
        '--path',
        help='temp directory path',
        type=str, default='tmp/'
    )
    args = parser.parse_args()

    """Run the benchmark."""
    pool = multiprocessing.Pool(processes=args.n)
    pool.map(simulator.simulate, parameters)
