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
        self.saved_scores = {}

    def set_params(self, feature_subset, *args, **kwargs):
        self.feature_subset = tuple(feature_subset)
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
        try:
            self.cls_scores = self.saved_scores[self.feature_subset]
        except KeyError:
            self.cls_scores = None

    def fit(self, feature_subset, data, target):
        self.set_params(feature_subset)
        frequencies = Counter(target)
        self.most_common = max(frequencies.keys(), key=frequencies.get)
        self.classes_ = np.array(sorted(set(target)))
        self.cls_to_index = {v: i for i, v in enumerate(self.classes_)}
        predictions = self.predict(data)
        self.cls_scores = np.zeros(len(self.classes_))
        for predicted, actual in zip(predictions, target):
            index = self.cls_to_index[actual]
            if predicted == actual:
                self.cls_scores[index] += 1
            else:
                self.cls_scores[index] -= 1
        self.saved_scores[self.feature_subset] = self.cls_scores

    def decision_function(self, data):
        probs = np.zeros((data.shape[0], self.classes_.shape[0]))
        if len(self.feature_subset) == 0:
            # probs[:, self.cls_to_index[self.most_common]] = 1
            return probs
        for i, col in enumerate(self.feature_subset):
            weight = self.weights[i]
            for row in range(data.shape[0]):
                cls_index = self.cls_to_index[data[row, col]]
                probs[row, cls_index] += weight
        return probs

    def predict(self, data):
        selected = self.decision_function(data).argmax(axis=1)
        return self.classes_[selected]

    def score(self, feature_subset, data, target):
        self.set_params(feature_subset)
        return self.cls_scores.sum()

    def score_class(self, feature_subset, cls):
        self.set_params(feature_subset)
        return self.cls_scores[self.cls_to_index[cls]]

    def decision_function_class(self, data, cls):
        # TODO This needs to be faster
        probs = self.decision_function(data)
        return utilities.counts_to_probabilities(probs)[:, self.cls_to_index[cls]]


class WeightedDecisions(BaseNode):

    def __init__(self, config):
        self.feature_subset = []
        self.saved_weights = {}
        self.weights = None

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
        self.classes_ = np.array(sorted(set(target)))
        self.cls_to_index = {cls: np.where(cls == self.classes_)[0][0]
                             for cls in self.classes_}
        self.most_common_index = self.cls_to_index[self.most_common]

    def decision_function(self, data):
        if len(self.feature_subset) == 0:
            result = np.zeros((data.shape[0], self.classes_.shape[0]))
            result[:, self.most_common_index] = 1
            # TODO Set to most common
            return result
        used = data[:, self.feature_subset, :]
        probs = (used * self.weights[:, None]).sum(axis=1)
        assert(probs.shape == (data.shape[0], self.classes_.shape[0]))
        return probs

    def predict(self, data):
        selected = self.decision_function(data).argmax(axis=1)
        return self.classes_[selected]

    def score(self, feature_subset, data, target):
        self.fit(feature_subset, data, target)
        estimates = self.predict(data)
        # extract only the used columns
        return sum(estimate == actual
                   for estimate, actual in zip(estimates, target)) / float(len(target))


class SKLearn(BaseNode):
    options = {clf.__name__: clf for clf in [svm.LinearSVC,
                                             linear_model.LogisticRegression,
                                             linear_model.Perceptron]}

    def __init__(self, config):
        # TODO Make configuration more flexible
        self.classifier_class = self.options[config['linear_classifier']]
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
        self.classes_ = np.array(sorted(set(target)))
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


class RandomWeights(BaseNode):

    def __init__(self, config):
        self.feature_subset = []
        self.saved_weights = {}
        self.weights = {}
        # TODO Rename probabilities to "count", as that is what it stores
        self.probability_store = {}
        self.probabilities = []
        self.threshold_default = 0 if config['threshold'] == 'zero' else None
        self.threshold_store = defaultdict(lambda: self.threshold_default)

    def get_weight(self, feature):
        try:
            return self.weights[feature]
        except:
            w = np.random.uniform(-1, 1)
            self.weights[feature] = w
            return w

    def set_params(self, feature_subset, *args, **kwargs):
        as_tuple = tuple(sorted(feature_subset))
        self.feature_subset = as_tuple
        # get the existing classifier
        try:
            self.probabilities = self.probability_store[as_tuple]
            self.actual_probs = utilities.counts_to_probabilities(self.probabilities)
            self.threshold = self.threshold_store[as_tuple]
        except KeyError:
            self.probabilities = None
            self.actual_probs = None
            self.threshold = self.threshold_default

    def bin_data(self, data):
        if len(self.feature_subset) == 0:
            self.threshold = 0
            self.threshold_store[self.feature_subset] = 0
            return np.zeros(data.shape[0], dtype="int")
        used = data[:, self.feature_subset]
        selected_weights = np.empty(len(self.feature_subset))
        for i, feature in enumerate(self.feature_subset):
            selected_weights[i] = self.get_weight(feature)
        product = np.dot(used, selected_weights)
        assert(product.shape[0] == data.shape[0])
        if self.threshold is None:
            self.threshold = np.random.choice(product)
            self.threshold_store[self.feature_subset] = self.threshold
        return (product > self.threshold).astype('int')

    def fit(self, feature_subset, data, target):
        self.set_params(feature_subset)
        self.classes_ = np.array(sorted(set(target)))
        self.cls_to_index = {cls: np.where(cls == self.classes_)[0][0]
                             for cls in self.classes_}
        bin_guide = self.bin_data(data)
        counts = np.zeros((2, self.classes_.shape[0]))
        for i, bin_val in enumerate(bin_guide):
            counts[bin_val][self.cls_to_index[target[i]]] += 1
        self.probabilities = counts
        self.probability_store[self.feature_subset] = self.probabilities
        self.base_entropy = utilities.weighted_entropy([Counter(target).values()])
        self.actual_probs = utilities.counts_to_probabilities(self.probabilities)

    def decision_function(self, data):
        bin_guide = self.bin_data(data)
        return self.actual_probs[bin_guide]

    def decision_function_class(self, data, cls):
        bin_guide = self.bin_data(data)
        # TODO Consider removing normalization to save time
        cls_probs = self.actual_probs[:, self.cls_to_index[cls]]
        return cls_probs[bin_guide]

    def predict(self, data):
        selected = self.decision_function(data).argmax(axis=1)
        assert(selected.shape[0] == data.shape[0])
        return self.classes_[selected]

    def score(self, feature_subset, data, target):
        self.set_params(feature_subset)
        result = utilities.weighted_entropy(self.probabilities)
        return self.base_entropy - result

    def score_class(self, feature_subset, cls):
        self.set_params(feature_subset)
        new_bins = []
        index = self.cls_to_index[cls]
        for bin_prob in self.probabilities:
            # TODO If you go to real probabilities, you can remove the sum
            non_class = bin_prob.sum() - bin_prob[index]
            new_bins.append([non_class, bin_prob[index]])
        result = utilities.weighted_entropy(new_bins)
        joined = [[new_bins[0][0] + new_bins[1][0],
                  new_bins[0][1] + new_bins[1][1]]]
        base_entropy = utilities.weighted_entropy(joined)
        return base_entropy - result
