'''
Implementation of Boundary Forests from "The Boundary Forest Algorithm for
Online Supervised and Unsupervised Learning" by Charles Mathy et. al., 2015.
'''
import numpy as np

from .metrics import norm, cat_noteq

__all__ = ('BoundaryTree', 'BoundaryForest')


class BoundaryTree:
    '''
    Boundary Tree.

    Parameters
    ----------
    position : np.array
        Position vector of a point.
    label : np.array or float
        Label of a point.
    children : List of BoundaryTrees, optional
        List of child nodes (default []).
    parent : BoundaryTree, optional
        Parent node (default None).
    '''
    def __init__(self, position, label, children=None, parent=None):
        self.position = position
        self.label = label

        if children is None:
            self.children = []
            self.childnum = 0
        else:
            self.children = children
            self.childnum = len(children)
        self.parent = parent

    def __str__(self, level=0):
        ret = '| '*level + '{}: {}\n'.format(self.position, self.label)
        if self.children is not None:
            for child in self.children:
                ret += child.__str__(level+1)
        return ret

    def __eq__(self, other):
        posclose = np.allclose(self.position, other.position)
        labclose = np.isclose(self.label, other.label)
        return posclose and labclose

    def addchild(self, position, label):
        newnode = BoundaryTree(position, label, parent=self)
        self.children.append(newnode)
        self.childnum += 1

    def query(self, inp, maxchild=5, metric_pos=norm):
        A = self.children[:]

        if self.childnum < maxchild:
            A.append(self)

        node_positions = np.array([node.position for node in A])
        distances = metric_pos(node_positions, inp)
        min_ind = np.argmin(distances)
        min_node = A[min_ind]

        if min_node is self:
            return self
        else:
            return min_node.query(inp, maxchild=maxchild, metric_pos=metric_pos)

    def oldquery(self, inp, maxchild=5, metric_pos=norm):
        A = self.children[:]

        if self.childnum < maxchild:
            A.append(self)

        mindist = np.inf
        minnode = None
        for node in A:
            nodedist = metric_pos(inp, node.position)
            if nodedist < mindist:
                mindist = nodedist
                minnode = node

        if minnode == self:
            return self
        else:
            return minnode.query(inp, maxchild=maxchild, metric_pos=metric_pos)



    def train(self, y, label, labthresh=0.1, maxchild=5,
              metric_lab=cat_noteq, metric_pos=norm):
        vmin = self.query(y, maxchild=maxchild, metric_pos=metric_pos)

        if metric_lab(label, vmin.label) > labthresh:
            vmin.addchild(y, label)

        return self


class BoundaryForest:
    '''
    Boundary Forest.

    Parameters
    ----------
    metric_pos : function, optional
        Metric function for position vectors (default metrics.norm).
    metric_lab : function, optional
        Metric function for labels (default metrics.cat_noteq).
    labthresh : float, optional
        Threshold for determining if labels are close enough to combine nodes
        (default 0.1).
    maxchild : int, optional
        Maximum number of children per node (default 5).
    '''
    params_names = ('metric_pos', 'metric_lab', 'labthresh', 'maxchild',
                    'maxtrees')
    treeparams_names = ('metric_pos', 'metric_lab', 'labthresh', 'maxchild')

    def __init__(self, metric_pos=norm, metric_lab=cat_noteq, labthresh=0.1,
                 maxchild=5, maxtrees=np.inf):

        self.metric_pos = metric_pos
        self.metric_lab = metric_lab
        self.labthresh = labthresh
        self.maxchild = maxchild
        self.maxtrees = maxtrees

        self.trees = []
        self.numtrees = 0

    @property
    def params(self):
        result = {}
        for name in self.params_names:
            result[name] = getattr(self, name)
        return result

    @property
    def treeparams(self):
        result = {}
        for name in self.treeparams_names:
            result[name] = getattr(self, name)
        return result


    def train_addpoint(self, y, label):
        for bt in self.trees:
            bt.train(y, label, **self.treeparams)

        return self

    def train_adddata(self, data, labels):
        for pos, lab in zip(data, labels):
            for bt in self.trees:
                bt.train_point(pos, lab)

        return self

    def fit(self, data, labels):
        for i, (pos, lab) in enumerate(zip(data, labels)):
            if i == self.maxtrees:
                break
            self.trees.append(BoundaryTree(pos, lab))

        for i, tree in enumerate(self.trees):
            for j, (pos, lab) in enumerate(zip(data, labels)):
                if i != j:
                    tree.train(pos, lab, **self.treeparams)

        self.numtrees = len(self.trees)
        return self

    def predict_obs(self, pos):
        distances = np.full(self.numtrees, np.nan)
        labels = np.full(self.numtrees, np.nan)
        for i, tree in enumerate(self.trees):
            nearest_node = tree.query(pos, maxchild=self.maxchild,
                                      metric_pos=self.metric_pos)
            distances[i] = self.metric_pos(pos, nearest_node.position)
            labels[i] = nearest_node.label

        pred = np.sum(labels/distances) / np.sum(1/distances)
        return pred

    def predict(self, data):
        if len(data.shape) == 1:
            return self.predict_obs(data)
        else:
            return np.array([self.predict_obs(obs) for obs in data])
