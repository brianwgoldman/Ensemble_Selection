import numpy as np

def all_subclasses(cls):
    result = {subclass.__name__: subclass for subclass in cls.__subclasses__()}
    recurse = [all_subclasses(subclass) for subclass in result.values()]
    for r in recurse:
        result.update(r)
    return result

def split_dataset(data, target, percentage):
    assert(data.shape[0] == target.shape[0])
    permutation = np.random.permutation(data.shape[0])
    divider = int(percentage * data.shape[0])
    first_indexes = permutation[:divider]
    second_indexes = permutation[divider:]
    assert(first_indexes.shape[0] > 0)
    assert(second_indexes.shape[0] > 0)
    return ((data[first_indexes, :], target[first_indexes]),
            (data[second_indexes, :], target[second_indexes]))