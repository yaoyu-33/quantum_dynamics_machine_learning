import time
import numpy as np


def gaussian(x, mu, sig):
    """ Gaussian Distribution """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def get_readable_time(elapsed):
    """ Convert time to a readable format """
    d, h, m, s = [int(x) for x in time.strftime("%d:%H:%M:%S", time.gmtime(elapsed)).split(':')]
    d -= 1
    return '{:2d}h{:2d}m{:2d}s'.format(24 * d + h, m, s)
 



