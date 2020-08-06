import random
import numpy as np
from scipy.stats import dlaplace, norm
from dis_gauss import discretegauss

# def dis_laplace(mu=0., b=1., L=1., size=1):
#     """p=e^(-a)"""
#     return dlaplace.rvs(L/b, loc=mu, size=size)

def dis_gauss_slow(sigma2=1.0, L=1.0, size=None):
    if size is None:
        return L * discretegauss.sample_dgauss(sigma2/L/L)
    else:
        return np.array([dis_gauss_slow(sigma2, L) for _ in range(size)])

def add_gauss_slow(v, sigma2=1.0, L=1.0):
    """ add discrete Gaussian noise to a given vetor
        v: vector
        sigma2: noise parameter
        L: LZ
    """
    noise = dis_gauss_slow(sigma2, L, size=len(v))
    return v + noise

def dis_gauss(sigma2=1.0, L=1.0, size=1):

    B = (100 * sigma2 // L) * L
    x = np.linspace(-B, B, num = int(2 * B / L) + 1)
    pdf = norm.pdf(x, scale=np.sqrt(sigma2))
    prob = pdf/np.sum(pdf)
    return np.random.choice(x, size, p=prob)

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

if __name__ == '__main__':
    # v = [1, 2, 3, 4]
    # print(add_binomial(v, 10, 0.5, 1))
    print(dis_gauss(size=10))
