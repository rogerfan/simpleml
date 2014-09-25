'''
Example of how to use auto-encoders.
'''
import sys, os, time
import gzip, pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

%cd C:/Users/g1rxf01/Downloads/New folder/simpleml/examples
# %cd M:/Libraries/Documents/Code/Python/simpleml/examples
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from simpleml.neural import AutoEncoder


# Load data
with gzip.open('../data/mnist.gz', 'rb') as f:
    data = pickle.load(f)


# Setup autoencoder
num_comp = 25
ae = AutoEncoder(
    num_inputs=data['train'][0].shape[1],
    num_comp=num_comp,
    learn_rate=.5, momentum=.1, seed=23456
)

# Estimate autoencoder
start = time.perf_counter()
ae.fit(data['train'][0], epochnum=5, verbose=1)
print("Time: {:5.2f}".format(time.perf_counter() - start))

# Visualize the encodings
fig1 = plt.figure(figsize=(10, 10))
for i in range(num_comp):
    side = np.sqrt(num_comp)
    ax = fig1.add_subplot(side, side, i+1)
    ax.imshow(ae.layers[0].weights[1:, i].reshape([28, 28]), cmap=cm.Greys_r)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
fig1.savefig('ex_ae_mnist.pdf')

