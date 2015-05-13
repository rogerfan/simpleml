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

    def dist(self, y, metric_pos=norm):
        return metric_pos(y, self.position)

    def query(self, inp, maxchild=5, metric_pos=norm):
        A = self.children

        if self.childnum < maxchild:
            A.append(self)

        mindist = np.inf
        minnode = None
        for node in A:
            nodedist = node.dist(inp, metric_pos=metric_pos)
            if nodedist < mindist:
                mindist = nodedist
                minnode = node

        if minnode == self:
            return self
        else:
            return minnode.query(inp)

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
    params_names = ('metric_pos', 'metric_lab', 'labthresh', 'maxchild')

    def __init__(self, metric_pos=norm, metric_lab=cat_noteq, labthresh=0.1,
                 maxchild=5):

        self.metric_pos = metric_pos
        self.metric_lab = metric_lab
        self.labthresh = labthresh
        self.maxchild = maxchild

        self.trees = []

    @property
    def params(self):
        result = {}
        for name in self.params_names:
            result[name] = getattr(self, name)
        return result

    def train_addpoint(self, y, label):
        for bt in self.trees:
            bt.train(y, label, **self.params)

        return self

    def train_adddata(self, data, labels):
        for pos, lab in zip(data, labels):
            for bt in self.trees:
                bt.train_point(pos, lab)

        return self

    def train(self, data, labels):
        for pos, lab in zip(data, labels):
            self.trees.append(BoundaryTree(pos, lab))

        for i, tree in enumerate(self.trees):
            for j, (pos, lab) in enumerate(zip(data, labels)):
                if i != j:
                    tree.train(pos, lab, **self.params)

        return self

