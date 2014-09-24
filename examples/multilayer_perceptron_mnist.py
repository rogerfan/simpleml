'''
Example of how to use Multilayer Perceptrons.
'''
import sys, os, time
import gzip, pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# from matplotlib.backends.backend_pdf import PdfPages

# %cd C:/Users/g1rxf01/Downloads/New folder/simpleml/examples
# %cd M:/Libraries/Documents/Code/Python/simpleml/examples
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from simpleml.neural import MultilayerPerceptron
from simpleml.transform import to_dummies


# Load data
with gzip.open('../data/mnist.gz', 'rb') as f:
    data = pickle.load(f)

data = {key: (val[0], val[1], to_dummies(val[1])) for key, val in data.items()}


# Setup estimator
num_hidden_nodes = [101]
mlp = MultilayerPerceptron(
    num_inputs=data['train'][0].shape[1]+1,
    num_outputs=data['train'][2].shape[1],
    num_hidden_layers=len(num_hidden_nodes), num_hidden_nodes=num_hidden_nodes,
    learn_rate=.5, momentum=.1, seed=23456
)

# Estimate multilayer perceptron
start = time.perf_counter()
mlp.fit(data['train'][0], data['train'][2],
        epochnum=10, verbose=1)
pred = mlp.classify(data['test'][0], max_ind=True)
print("Time: {:5.2f}, Error: {:5.4f}".format(
    time.perf_counter() - start,
    1 - np.mean(pred == data['test'][1])
))


# Visualize first hidden layer
fig1 = plt.figure(figsize=(10, 10))
for i in range(num_hidden_nodes[0]-1):
    side = np.sqrt(num_hidden_nodes[0]-1)
    ax = fig1.add_subplot(side, side, i+1)
    ax.imshow(mlp.layers[0].weights[1:, i].reshape([28, 28]), cmap=cm.Greys_r)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
fig1.savefig('ex_mlp_mnist.pdf')

