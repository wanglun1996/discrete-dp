import sys
sys.path.append('../infimnist_py')
import _infimnist as infimnist
import numpy as np

mnist = infimnist.InfimnistGenerator()
indexes = np.array([0, 10000, 70000], dtype=np.int64)
digits, labels = mnist.gen(indexes)

# example of preprocessing from [0, 255] to [0., 1.]
X = digits.astype(np.float32).reshape(indexes.shape[0], 28, 28)
X = X / 255

import matplotlib.pyplot as plt
plt.imshow(X[0])
plt.title('label: {}'.format(labels[0]))
