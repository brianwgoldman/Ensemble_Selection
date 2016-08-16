import numpy as np
from collections import defaultdict
from math import log


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


def even_class_split_dataset(data, target, percentage):
    indexes_by_class = defaultdict(list)
    for i, cls in enumerate(target):
        indexes_by_class[cls].append(i)
    first_indexes, second_indexes = [], []
    for cls, indexes in indexes_by_class.items():
        divider = int(len(indexes) * percentage)
        assert(divider > 0)
        assert(divider + 1 < len(indexes))
        np.random.shuffle(indexes)
        first_indexes.extend(indexes[:divider])
        second_indexes.extend(indexes[divider:])
    return ((data[first_indexes, :], target[first_indexes]),
            (data[second_indexes, :], target[second_indexes]))


def entropy(X):
    total = float(sum(X))
    return -sum(x / total * log(x / total, 2) for x in X if x != 0)


def weighted_entropy(list_of_lists):
    grand_total = float(sum(sum(X) for X in list_of_lists))
    return sum(sum(X) / grand_total * entropy(X) for X in list_of_lists)

# TODO Do test case style main like in nk.py