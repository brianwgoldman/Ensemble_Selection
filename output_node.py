import numpy as np
import utilities

class OutputNode(object):
    def set_parameters(self, feature_subset, *args, **kwargs):
        raise NotImplementedError()
    def fit(self, feature_subset, data, target):
        raise NotImplementedError()
    def predict(self, data):
        raise NotImplementedError()
    def score(self, feature_subset, data, target):
        raise NotImplementedError()

def get_output_node_class(config):
    lookup = utilities.all_subclasses(OutputNode)
    return lookup[config['output_node_type']]

class RandomWeightsNode(OutputNode):
    def __init__(self, config):
        self.feature_subset = []
        self.saved_weights = {}
        self.weights = []
        self.threshold = config['threshold']
        self.target_class = config['target_class']
        self.to_class = np.vectorize(lambda X: self.target_class if X else "")

    def set_parameters(self, feature_subset, *args, **kwargs):
        self.feature_subset = feature_subset
        weights = []
        for feature in self.feature_subset:
            try:
                weights.append(self.saved_weights[feature])
            except KeyError:
                # Generate weights only once per feature
                weight = np.random.random()
                self.saved_weigts[feature] = weight
                weights.append(weight)
        self.weights = np.array(weights)
    
    def fit(self, feature_subset, data, target):
        self.set_parameters(feature_subset)
    
    def predict(self, data):
        used = data[:, self.feature_subset]
        # multiply each column by its weight
        weighted = used * self.weights
        # sum across the column
        output_column = weighted.sum(axis=1)
        positives = output_column >= self.threshold
        return self.to_class(positives)
    
    def score(self, feature_subset, data, target):
        self.fit(feature_subset, data, target)
        estimates = self.predict(data)
        # extract only the used columns
        return sum(estimate == actual
                   for estimate, actual in zip(estimates, target))
