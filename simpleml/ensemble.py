'''
Ensemble methods
'''
import numpy as np

from . import baseclasses


class EnsembleBinaryClassifier:
    def __init__(self):
        self.models = []
        self.n_models = 0
        self.weights = []

    def add_model(self, model, weight=1):
        if not isinstance(model, baseclasses.BinaryClassifier):
            typename = type(model).__name__
            raise TypeError("Model is '{}', which is not a "
                            "'BinaryClassifier.'".format(typename))
        self.models.append(model)
        self.n_models += 1
        self.weights.append(weight)

    def classify(self, new_data):
        n_obs = len(new_data)
        model_preds = np.zeros((n_obs, self.n_models))
        for i, model in enumerate(self.models):
            model_preds[:,i] = model.classify(new_data)
        results = np.average(model_preds, axis=1, weights=self.weights)
        return results >= .5








# class BaggingBinaryClassifier:
#     def __init__(self, binaryclassifier):
#         self.model =
