import numpy as np
from scipy.linalg import hadamard

def clip_gradient(v, clip_bound=1.):
    norm = np.linalg.norm(v)
    if norm <= clip_bound:
        return v
    return v / norm

def quantize(v, k, B):
    """stochastic k-level quantization protocol
       v: the vector to be quantized
       k: the level of quantization
       B: the upper and lower bound
    """
    intervals = np.linspace(-B, B, num=k)
    step_size = 2 * B / (k-1)
    idx = [int((x+B) // step_size) for x in v]
    idx = np.array([[i, i+1] for i in idx])
    probs = np.array([[1-(x-intervals[idx_pair[0]])/step_size, (x-intervals[idx_pair[0]])/step_size] for x, idx_pair in list(zip(v, idx))])

    return np.array([intervals[np.random.choice(idx_pair, p=prob)] for idx_pair, prob in list(zip(idx, probs))])

# def reverse_quantize(v, k, B):
#     intervals = np.linspace(-B, B, num=k)
#     return np.array([intervals[int(idx)] for idx in v])

def random_diag(d):
    return np.random.choice([-1, 1], d)

# FIXME: add random matrix as an argument
def rotate(v, diag=None, reverse=False):
    """ random rotate the feature vector
        v: feature vector, the dimension must be a power of 2
    """
    d = len(v)
    if diag is None:
        diag = np.random.choice([-1, 1], d)

    def fwht(v):
        """In-place Fast Walsh–Hadamard Transform of array a."""
        h = 1
        while h < len(v):
            for i in range(0, len(v), h * 2):
                for j in range(i, i + h):
                    x = v[j]
                    y = v[j + h]
                    v[j] = x + y
                    v[j + h] = x - y
            h *= 2
        return v
    if reverse:
        rot_v = fwht(v) * diag / np.sqrt(d)
    else:
        rot_v = fwht(diag * v) / np.sqrt(d)

    return rot_v

def cylicRound(v, k, B):
    intervals = np.linspace(-B, B, num=k)
    step_size = 2 * B / (k-1)

    return np.array([intervals[int((x+B)/step_size)%int(2*B/step_size+1)] for x in v])

if __name__ == '__main__':
    v = [-11, -12, 11, 12]
    v = cylicRound(v, 21, 10)
    print(v)
