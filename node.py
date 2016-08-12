import numpy as np
import utilities
from sklearn import linear_model, svm
from collections import defaultdict, Counter


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


class WeightedVote(BaseNode):

    def __init__(self, config):
        self.feature_subset = []
        self.saved_weights = {}
        self.weights = []

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
        frequencies = Counter(target)
        self.most_common = max(frequencies.keys(), key=frequencies.get)

    def predict(self, data):
        if len(self.feature_subset) == 0:
            result = np.array([self.most_common for _ in range(data.shape[0])])
            return result
        used = data[:, self.feature_subset]
        # TODO This probably needs optimization
        ballots = [defaultdict(float) for _ in range(used.shape[0])]
        for col in range(used.shape[1]):
            weight = self.weights[col]
            for row in range(used.shape[0]):
                ballots[row][used[row][col]] = weight
        predictions = [max(vote.keys(), key=vote.get) for vote in ballots]
        return np.array(predictions)

    def score(self, feature_subset, data, target):
        self.fit(feature_subset, data, target)
        estimates = self.predict(data)
        # extract only the used columns
        return sum(estimate == actual
                   for estimate, actual in zip(estimates, target)) / float(len(target))


class SKLearn(BaseNode):

    def __init__(self, config):
        # TODO Make configuration more flexible
        self.classifier_class = vars(linear_model)[config['linear_classifier']]
        #self.classifier_class = svm.SVC
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
        frequencies = Counter(target)
        self.most_common = max(frequencies.keys(), key=frequencies.get)
        if len(self.feature_subset) > 0:
            self.classifier.fit(data[:, self.feature_subset], target)

    def predict(self, data):
        if len(self.feature_subset) == 0:
            result = np.array([self.most_common for _ in range(data.shape[0])])
            return result
        return self.classifier.predict(data[:, self.feature_subset])
    
    def decision_function(self, data):
        return self.classifier.decision_function(data[:, self.feature_subset])

    def score(self, feature_subset, data, target):
        # TODO Handle 0 feature more gracefully
        if len(self.feature_subset) > 0:
            return self.classifier.score(data[:, self.feature_subset], target)
        else:
            predictions = self.predict(data)
            return sum(estimate == actual for estimate, actual in zip(predictions, target)) / target.shape[0]
