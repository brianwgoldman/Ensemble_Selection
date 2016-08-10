from __future__ import print_function
import numpy as np
from glob import glob
from os import path
from collections import defaultdict

def create_subproblem(folder, num_classes, output_filename):
    class_files = glob(path.join(folder, "*.npy"))
    subset_files = np.random.choice(class_files, num_classes, replace=False)
    with open(output_filename, 'w') as f:
        for filename in subset_files:
            print(filename, file=f)

def load_subproblem(filename):
    with open(filename, 'r') as f:
        subset_files = f.read().strip().split()
    target = []
    for filename in subset_files:
        with open(filename, "r") as f:
            class_data = np.load(f)
        class_label = path.splitext(path.basename(filename))[0]
        target.extend([class_label] * class_data.shape[0])
        try:
            data = np.vstack([data, class_data])
        except NameError:
            data = class_data

    target = np.array(target)
    return data, target

def create_class_files(groundtruth_file, input_folder, output_folder):
    with open(groundtruth_file, 'r') as f:
        lines = f.read().strip().split('\n')

    class_to_files = defaultdict(list)
    for line in lines:
        filename, cls = line.strip().split()
        class_to_files[cls].append(filename)

    count = 0
    for cls, files in class_to_files.items():
        data = []
        count += 1
        print("Starting class number", count)
        for filename in files:
            with open(path.join(input_folder, filename) + '.txt') as f:
                line = f.read().strip().split()
            data.append(line)
        data = np.array(data, dtype="float64")
        with open(path.join(output_folder, cls) + '.npy', 'w') as f:
            np.save(f, data)

if __name__ == '__main__':
    # TODO Probably should argparse this
    # TODO Allow command line switching between functions
    import sys
    folder = sys.argv[1]
    num_classes = int(sys.argv[2])
    output_filename = sys.argv[3]
    create_subproblem(folder, num_classes, output_filename)
