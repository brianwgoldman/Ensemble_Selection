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
        feature_subset = np.arange(data.shape[1])
        for output in show_completion(self.outputs, self.N,
                                      "Building Middle layer"):
            train, _ = even_class_split_dataset(data, target, self.sample_percentage)
            output.fit(feature_subset, train[0], train[1])

    def predict(self, data):
        columns = [output.predict(data) for output in
                   show_completion(self.outputs, self.N, "Transforming Data")]
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


class RandomProjections(BaseMiddleLayer):

    def __init__(self, config):
        self.N = config['N']

    def fit(self, data, target):
        shape = (data.shape[1], self.N)
        self.projection = np.random.uniform(-1, 1, shape)
        # return
        for col in range(shape[1]):
            number = np.random.choice(shape[0]) + 1
            select = np.random.choice(shape[0], number, replace=False)
            self.projection[select, col] = 0

    def predict(self, data):
        return np.dot(data, self.projection)
