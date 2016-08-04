import numpy as np
import utilities
from sklearn import linear_model

class BaseNode(object):
    def set_parameters(self, feature_subset, *args, **kwargs):
        raise NotImplementedError()
    def fit(self, feature_subset, data, target):
        raise NotImplementedError()
    def predict(self, data):
        raise NotImplementedError()
    def score(self, feature_subset, data, target):
        raise NotImplementedError()

def get_output_node_class(config):
    lookup = utilities.all_subclasses(BaseNode)
    return lookup[config['output_node_type']]

class WeightedSum(BaseNode):
    def __init__(self, config):
        self.feature_subset = []
        self.saved_weights = {}
        self.weights = []
        self.threshold = config['threshold']
        assert(self.threshold is not None)
        self.target_class = config['target_class']
        assert(self.target_class is not None)
        self.to_class = np.vectorize(lambda X: self.target_class if X else "")

    def set_params(self, feature_subset, *args, **kwargs):
        self.feature_subset = feature_subset
        weights = []
        for feature in self.feature_subset:
            try:
                weights.append(self.saved_weights[feature])
            except KeyError:
                # Generate weights only once per feature
                weight = -np.log2(np.random.random())
                self.saved_weights[feature] = weight
                weights.append(weight)
        self.weights = np.array(weights)
        # Scale the actually used weights to sum to 1
        total = self.weights.sum()
        self.weights /= total

    def fit(self, feature_subset, data, target):
        self.set_params(feature_subset)
    
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

class SKLearn(BaseNode):
    def __init__(self, config):
        self.classifier_class = vars(linear_model)[config['linear_classifier']]
        self.stored_classifiers = {}
    def set_params(self, feature_subset, *args, **kwargs):
        as_tuple = tuple(sorted(feature_subset))
        self.feature_subset = as_tuple
        # get the existing classifier
        try:
            self.classifier = self.stored_classifiers[as_tuple]
        except KeyError:
            self.classifier = self.classifier_class()
            self.stored_classifiers[as_tuple] = self.classifier
        # TODO Consider setting the parameters of the classifier itself

    def fit(self, feature_subset, data, target):
        self.set_params(feature_subset)
        self.classifier.fit(data[:, self.feature_subset], target)

    def predict(self, data):
        return self.classifier.predict(data[:, self.feature_subset])

    def score(self, feature_subset, data, target):
        self.fit(feature_subset, data, target)
        return self.classifier.score(data[:, self.feature_subset], target)