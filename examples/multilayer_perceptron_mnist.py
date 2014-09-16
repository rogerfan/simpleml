'''
Example of how to use Decision Trees and Random Forests.
'''
import sys, os, time
import gzip, pickle

import numpy as np

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
    data[name] = (dataset[0][:1000], dataset[1][:1000])



mlp = MultilayerPerceptron(
    num_inputs=data['train'][0].shape[1]+1,
    num_outputs=data['train'][1].shape[1],
    num_hidden_layers=2, num_hidden_nodes=10,
    learn_rate=.5, momentum=.1, seed=23456
)

start = time.perf_counter()
mlp.fit(data['train'][0], to_dummies(data['train'][1]),
        epochnum=10, verbose=True)
print("Time: {:6.2f}".format(time.perf_counter() - start))

