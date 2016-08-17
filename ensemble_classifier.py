import nk
import node
import numpy as np
from collections import defaultdict


class BaseClassifier(object):
    def __init__(self, config):
        raise NotImplementedError()

    def fit(self, data, target):
        raise NotImplementedError()

    def predict(self, data):
        raise NotImplementedError()

    def predict_using_numbers(self, data):
        raise NotImplementedError()


class EnsembleClassifier(BaseClassifier):

    def __init__(self, config):
        self.N = config['N']
        self.K = config['K']
        output_class = node.get_output_node_class(config)
        self.outputs = [output_class(config) for _ in range(self.N)]

    def build_nk_table(self, data, target):
        patterns = 2 << self.K
        self.nk_table = np.zeros((self.N, patterns), dtype="float")
        for i in range(self.N):
            print "Starting column", i, "of", self.N, "in the NK table"
            for pattern in range(patterns):
                relative_indexes = nk.int_to_set_bits(pattern)
                absolute_indexes = [(i + r) % self.N for r in relative_indexes]
                self.outputs[i].fit(absolute_indexes, data, target)
                quality = self.outputs[i].score(absolute_indexes, data, target)
                self.nk_table[i, pattern] = quality

    def optimize_nk(self):
        self.selected = nk.dynamic_programming(self.nk_table, self.K)

    def configure_outputs(self):
        self.output_scores = np.empty([self.N], dtype="float")
        circular = np.concatenate([self.selected, self.selected])
        for i in range(self.N):
            index = nk.list_to_int(circular[i:i + self.K + 1])
            self.output_scores[i] = self.nk_table[i][index]
            feature_subset = [(i + j) % self.N for j in
                              range(self.K + 1) if circular[i + j]]
            self.outputs[i].set_params(feature_subset)
        print "Best single output estimate:", max(self.output_scores), self.output_scores.mean()

    def fit(self, data, target):
        self.build_nk_table(data, target)
        self.optimize_nk()
        self.configure_outputs()

    def predict(self, data):
        votes = [defaultdict(float) for _ in range(data.shape[0])]
        for o in range(self.N):
            labels = self.outputs[o].predict(data)
            weight = self.output_scores[o]
            for r in range(data.shape[0]):
                votes[r][labels[r]] += weight
        result = [max(vote.keys(), key=vote.get) for vote in votes]
        return np.array(result)

    def decision_function(self, data):
        to_class = self.outputs[0].classes_
        probs = np.zeros([data.shape[0], to_class.shape[0]])
        for o in range(self.N):
            assert((to_class == self.outputs[o].classes_).all())
            output_probs = self.outputs[o].decision_function(data)
            weight = self.output_scores[o]
            probs += (output_probs * weight)
        return probs

    def predict_using_numbers(self, data):
        probs = self.decision_function(data)
        columns = np.argmax(probs, axis=1)
        return self.outputs[0].classes_[columns]


class MultiEnsembleClassifier(BaseClassifier):

    def __init__(self, config):
        self.N = config['N']
        self.K = config['K']
        output_class = node.get_output_node_class(config)
        self.outputs = [output_class(config) for _ in range(self.N)]
        self.patterns = 2 << self.K
        self.class_masks = {}
        self.output_scores = {}
        self.configure_patterns = {}

    def pattern_to_indexes(self, pattern, i):
        relative_indexes = nk.int_to_set_bits(pattern)
        return [(i + r) % self.N for r in relative_indexes]

    def fit(self, data, target):
        self.classes_ = np.array(sorted(set(target)))
        self.cls_to_index = {cls: np.where(cls == self.classes_)[0][0]
                             for cls in self.classes_}

        for i in range(self.N):
            print "Starting output", i, "of", self.N
            for pattern in range(self.patterns):
                absolute_indexes = self.pattern_to_indexes(pattern, i)
                self.outputs[i].fit(absolute_indexes, data, target)
        for i, cls in enumerate(self.classes_):
            print "Solving NK Table for class", i, "of", len(self.classes_)
            self.solve_nk_table(cls)

    def solve_nk_table(self, cls):
        nk_table = np.zeros((self.N, self.patterns), dtype="float")
        for i in range(self.N):
            for pattern in range(self.patterns):
                absolute_indexes = self.pattern_to_indexes(pattern, i)
                quality = self.outputs[i].score_class(absolute_indexes, cls)
                nk_table[i, pattern] = quality
        selected = nk.dynamic_programming(nk_table, self.K)
        output_scores = np.empty([self.N], dtype="float")
        configure_patterns = []
        circular = np.concatenate([selected, selected])
        for i in range(self.N):
            index = nk.list_to_int(circular[i:i + self.K + 1])
            output_scores[i] = nk_table[i][index]
            feature_subset = [(i + j) % self.N for j in
                              range(self.K + 1) if circular[i + j]]
            configure_patterns.append(feature_subset)
        self.class_masks[cls] = selected
        self.output_scores[cls] = output_scores
        self.configure_patterns[cls] = configure_patterns

    def get_class_probabilities(self, data, cls):
        column = self.cls_to_index[cls]
        total_weight = 0
        weights = self.output_scores[cls]
        patterns = self.configure_patterns[cls]
        probs = np.zeros(data.shape[0])
        zeros = 0
        for i in range(self.N):
            if weights[i] > 0:
                self.outputs[i].set_params(patterns[i])
                # Get only the target classes output probabilities
                output_probs = self.outputs[i].decision_function(data)[:, column]
                probs += (output_probs * weights[i])
                total_weight += weights[i]
            else:
                zeros += 1
        print "Expected percentage in this class:", (probs / total_weight).mean()
        print "Zeros:", zeros
        return probs / total_weight

    def predict_using_numbers(self, data):
        probs = np.empty((data.shape[0], self.classes_.shape[0]))
        for i, cls in enumerate(self.classes_):
            print "Predicting class", i, "of", len(self.classes_)
            probs[:, i] = self.get_class_probabilities(data, cls)
        columns = np.argmax(probs, axis=1)
        return self.classes_[columns]


class EntirelySeparate(BaseClassifier):

    def __init__(self, config):
        self.N = config['N']
        self.config = config

    def fit(self, data, target):
        self.classes_ = np.array(sorted(set(target)))
        self.cls_to_index = {cls: np.where(cls == self.classes_)[0][0]
                             for cls in self.classes_}
        # Build an entire classifier for each class
        self.classifiers = [EnsembleClassifier(self.config)
                            for _ in self.classes_]
        for clf in self.classifiers:
            clf.fit(data, target)
            # Save on memory
            clf.nk_table = None

    def predict_using_numbers(self, data):
        probs = np.zeros((data.shape[0], self.classes_.shape[0]))
        for clf in self.classifiers:
            probs += clf.decision_function(data)
        columns = np.argmax(probs, axis=1)
        return self.classes_[columns]
