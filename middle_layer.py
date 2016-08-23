import node
import math
import numpy as np
from utilities import show_completion, even_class_split_dataset


class BaseMiddleLayer(object):
    def __init__(self, config):
        raise NotImplementedError()

    def fit(self, data, target):
        raise NotImplementedError()

    def predict(self, data):
        raise NotImplementedError()


class NoAction(BaseMiddleLayer):
    def __init__(self, config):
        pass

    def fit(self, data, target):
        pass

    def predict(self, data):
        return data


class Nodes(BaseMiddleLayer):

    def __init__(self, config):
        self.N = config['N']
        self.sample_percentage = config['sample_percentage']
        # TODO Make configurable
        self.outputs = [node.SKLearn(config) for _ in range(config['N'])]

    def fit(self, data, target):
        sample_size = int(math.ceil(data.shape[1] * self.sample_percentage))
        for output in show_completion(self.outputs, self.N,
                                      "Building Middle layer"):
            feature_subset = np.random.choice(data.shape[1], sample_size,
                                              replace=False)
            train, _ = even_class_split_dataset(data, target, self.sample_percentage)
            output.fit(feature_subset, train[0], train[1])

    def predict(self, data):
        columns = [output.predict(data) for output in self.outputs]
        return np.vstack(columns).transpose()
        columns = [output.decision_function(data) for output in self.outputs]
        combined = np.swapaxes(np.array(columns), 0, 1)
        return combined



class Randomize(BaseMiddleLayer):

    def __init__(self, config):
        self.N = config['N']

    def fit(self, data, target):
        blocks = []
        size = 0
        step = data.shape[1]
        while size < self.N:
            blocks.append(np.random.permutation(step))
            size += step
        self.new_order = np.concatenate(blocks)[:self.N]

    def predict(self, data):
        return data[:, self.new_order]
