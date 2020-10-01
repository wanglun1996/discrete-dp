import random
import numpy as np
from scipy.stats import dlaplace, norm
from dis_gauss import discretegauss

def dis_gauss(sigma2=1.0, L=1.0, size=None):
    if size is None:
        return L * discretegauss.sample_dgauss(sigma2/L/L)
    else:
        return np.array([dis_gauss_slow(sigma2, L) for _ in range(size)])

def add_gauss(v, sigma2=1.0, L=1.0):
    """ add discrete Gaussian noise to a given vetor
        v: vector
        sigma2: noise parameter
        L: LZ
    """
    noise = dis_gauss(sigma2, L, size=len(v))
    return v + noise

def add_binom(v, m, p, L):
    """ add binomial noise to a given vector 
        v: vector
        m, p: noise parameters
        L: LZ
    """
    noise = np.random.binomial(m, p, size=len(v))
    return v + L * (noise - m * p)
