from math import floor
from mpmath import nsum, exp, inf, npdf
import numpy as np
from numpy.random import choice, uniform
from scipy.stats import dlaplace

# TODO: 1. check dnormal is correct when L is not 1
#       2. nsum: smaller tolerance

def dis_normal(mu=0., sigma=1., L=1., size=1):

    def pdf(x, mu, sigma, L):
        assert float(x * L).is_integer() and x >= 0, "pdf: point is not in lattice."
        denominator = L * nsum(lambda t: npdf(t, mu, sigma*L), [-inf, inf])
        if x == 0:
            return npdf(x, mu, sigma) / denominator
        return 2 * npdf(x, mu, sigma) / denominator

    def cdf(x):
        assert x >= 0, "cdf: point less than zero"
        return L * nsum(lambda t: pdf(t, mu, sigma, L), [0, floor(L * x)])

    def find_interval(u):
        if u <= cdf(0):
            return 0
        k = 1.
        while True:
            (left, right) = (cdf(k-1), cdf(k))
            if left < u <= right:
                return k
            k += 1.

    u = uniform(size=size)
    return choice([-1., 1.], size, [0.5, 0.5]) * np.array([find_interval(ui) for ui in u])

def dis_laplace(mu=0., b=1., L=1., size=1):
    """p=e^(-a)"""
    return dlaplace.rvs(L/b, loc=mu, size=size)

def ascii_histogram(hist):
    """A horizontal frequency-table/histogram plot."""
    for k in sorted(hist):
        print('{0:5d} {1}'.format(k, '+' * hist[k]))

if __name__ == '__main__':
    hist = {}
    for i in range(0, 100):
        print(i)
        idx = int(dis_normal())
        hist[idx] = hist.get(idx, 0) + 1
    ascii_histogram(hist)
     

