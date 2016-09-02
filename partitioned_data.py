from os import path
import numpy as np
from collections import defaultdict
from glob import glob


def load_data(files):
    row_counts = []
    total_cols = None
    for filename in files:
        pieces = path.basename(filename).split('_')
        row_counts.append(int(pieces[2]))
        new_cols = int(pieces[3])
        assert(total_cols is None or total_cols == new_cols)
        total_cols = new_cols
    total_rows = sum(row_counts)
    print "Data size:", total_rows, total_cols
    data = np.empty((total_rows, total_cols), dtype="float64")
    used = 0
    for rows, filename in zip(row_counts, files):
        with open(filename, 'r') as f:
            data[used:used + rows, :] = np.load(f)
        used += rows
        print "Loaded", used, "rows"
    assert(used == total_rows)
    return data


def load_linear(files):
    target = []
    for filename in files:
        with open(filename, 'r') as f:
            target.append(np.load(f))
    return np.concatenate(target)


def group_files(files):
    grouped = defaultdict(dict)
    for filename in files:
        pieces = path.basename(filename)[:-4].split('_')
        partition_number = int(pieces[1])
        file_type = pieces[4]
        grouped[partition_number][file_type] = filename
    return grouped


def select_partitions(grouped, partition_numbers):
    selected = [grouped[number] for number in partition_numbers]
    keys = ['data', 'target', 'image']
    by_type = {}
    for key in keys:
        by_type[key] = [p[key] for p in selected]
    return by_type


def load_problem(folder, partition_style, index):
    files = glob(path.join(folder, '*.npy'))
    grouped = group_files(files)
    if partition_style == 'all':
        partition_numbers = grouped.keys()
    elif partition_style == 'single':
        assert(index in grouped.keys())
        partition_numbers = [index]
    elif partition_style == 'all_but_one':
        partition_numbers = [key for key in grouped.keys() if key != index]
        assert(len(partition_numbers) == len(grouped).keys() - 1)
    else:
        raise ValueError("Unknown partition style:" + partition_style)
    partition_numbers.sort()
    selected = select_partitions(grouped, partition_numbers)
    data = load_data(selected['data'])
    target = load_linear(selected['target'])
    return data, target

if __name__ == '__main__':
    from sklearn import linear_model, svm
    from sklearn.externals import joblib
    import argparse

    clf_lookup = {'Perceptron': linear_model.Perceptron,
                  'RBFSVM': svm.SVC,
                  'LinearSVM': svm.LinearSVC,
                  }
    clf_option_string = ', '.join(sorted(clf_lookup.keys()))

    description = 'Train classifiers using partitioned data'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('folder',
                        help='The folder containing the partitioned data')
    parser.add_argument('partition_style', help='How to combine partitions'
                        + ' (all, single, all_but_one)')
    parser.add_argument('classifier',
                        help='Which classifier to apply: ' + clf_option_string)
    parser.add_argument('output_file',
                        help='Where to save the classifier')
    parser.add_argument('-index', type=int, default=-1,
                        help='Control which partition is used/excluded')

    config = vars(parser.parse_args())
    clf = clf_lookup[config['classifier']]()
    data, target = load_problem(config['folder'], config['partition_style'], config['index'])
    clf.fit(data, target)
    joblib.dump(clf, config['output_file'])
