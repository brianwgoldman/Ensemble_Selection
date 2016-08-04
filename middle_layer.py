import node
import math
import numpy as np


class MiddleLayer(object):
    def __init__(self, config):
        self.N = config['N']
        self.sample_percentage = config['sample_percentage']
        # TODO Make configurable
        self.outputs = [node.SKLearn(config) for _ in range(config['N'])]
    
    def fit(self, data, target):
        sample_size = int(math.ceil(data.shape[1] * self.sample_percentage))
        for output in self.outputs:
            feature_subset = np.random.choice(data.shape[1], sample_size, replace=False)
            output.fit(feature_subset, data, target)
    
    def predict(self, data):
        columns = [output.predict(data) for output in self.outputs]
        return np.vstack(columns).transpose()