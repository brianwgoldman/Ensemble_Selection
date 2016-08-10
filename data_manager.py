from __future__ import print_function
import numpy as np
from glob import glob
from os import path, makedirs
from collections import defaultdict

def create_subproblem(input_folder, num_classes, output_filename, **_):
    class_files = glob(path.join(input_folder, "*.npy"))
    subset_files = np.random.choice(class_files, num_classes, replace=False)
    with open(output_filename, 'w') as f:
        for filename in subset_files:
            print(filename, file=f)

def load_problem(input_filename, **_):
    with open(input_filename, 'r') as f:
        subset_files = f.read().strip().split()
    target = []
    for filename in subset_files:
        with open(filename, "r") as f:
            class_data = np.load(f)
        class_label = path.splitext(path.basename(filename))[0]
        target.extend([class_label] * class_data.shape[0])
        # This try except block is used to create "data" initialized
        # to the very first file's "class_data"
        try:
            data = np.vstack([data, class_data])
        except NameError:
            data = class_data

    target = np.array(target)
    return data, target

def create_class_files(groundtruth_file, input_folder, output_folder, **_):
    makedirs(output_folder) # TODO Consider providing helpful error messages
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
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Munge data from Dhruva")
    parser.add_argument("tool")
    parser.add_argument('-input_folder', type=str)
    parser.add_argument('-num_classes', type=int)
    parser.add_argument('-output_filename', type=str)
    parser.add_argument('-input_filename', type=str)
    parser.add_argument('-groundtruth_file', type=str)
    parser.add_argument('-output_folder', type=str)
    command_line = vars(parser.parse_args())
    # TODO Make better error handling and help messages
    locals()[command_line['tool']](**command_line)
