import nk
import node
import numpy as np
from collections import defaultdict


class EnsembleClassifier(object):

    def __init__(self, config):
        self.N = config['N']
        self.K = config['K']
        output_class = node.get_output_node_class(config)
        self.outputs = [output_class(config) for _ in range(self.N)]

    def build_nk_table(self, data, target):
        patterns = 2 << self.K
        self.nk_table = np.zeros((self.N, patterns), dtype="float")
        for i in range(self.N):
            for pattern in range(patterns):
                relative_indexes = nk.int_to_set_bits(pattern)
                absolute_indexes = [(i + r) % self.N for r in relative_indexes]
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

    def predict(self, data):
        votes = [defaultdict(float) for _ in range(data.shape[0])]
        for o in range(self.N):
            labels = self.outputs[o].predict(data)
            weight = self.output_scores[o]
            for r in range(data.shape[0]):
                votes[r][labels[r]] += weight
        result = [max(vote.keys(), key=vote.get) for vote in votes]
        return np.array(result)
