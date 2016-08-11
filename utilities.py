import numpy as np
from collections import defaultdict, Counter


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
    '''
    original = Counter(target)
    first = Counter(target[first_indexes])
    second = Counter(target[second_indexes])
    print len(original.keys()), len(first.keys()), len(second.keys())
    assert(len(first.keys()) == len(second.keys()))
    assert(len(original.keys()) == len(first.keys()))
    #'''
    return ((data[first_indexes, :], target[first_indexes]),
            (data[second_indexes, :], target[second_indexes]))