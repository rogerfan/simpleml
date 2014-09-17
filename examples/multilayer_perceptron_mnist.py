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

from simpleml.perceptron import MultilayerPerceptron
from simpleml.transform import to_dummies


with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()

data = {}
names = ['train', 'valid', 'test']
for name, dataset in zip(names, (train_set, valid_set, test_set)):
    data[name] = (dataset[0], dataset[1], to_dummies(dataset[1]))


mlp = MultilayerPerceptron(
    num_inputs=data['train'][0].shape[1]+1,
    num_outputs=data['train'][2].shape[1],
    num_hidden_layers=2, num_hidden_nodes=[37, 17],
    learn_rate=.5, momentum=.1, seed=23456
)

start = time.perf_counter()
mlp.fit(data['train'][0], data['train'][2],
        epochnum=10, verbose=2)
pred = mlp.classify(data['test'][0], max_ind=True)
print("Time: {:5.2f}, Error: {:5.4f}".format(
    time.perf_counter() - start,
    1 - np.mean(pred == data['test'][1])
))


fig1 = plt.figure(figsize=(10, 10))
for i in range(36):
    ax = fig1.add_subplot(6, 6, i+1)
    ax.imshow(mlp.layers[0].weights[1:, i].reshape([28, 28]), cmap=cm.Greys_r)
fig1.savefig('ex_mlp_mnist.pdf')

