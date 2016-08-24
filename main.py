import argparse
import sys
import numpy as np
import ensemble_classifier
import middle_layer
from utilities import even_class_split_dataset, all_subclasses
import data_manager
import json

# Set up argument parsing
description = 'Ensemble Selection using NK Landscapes'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('cfg_input_files', metavar='Configuration Files',
                    type=str, nargs='*', default=[],
                    help='Zero or more json formatted files containing' +
                    ' configuration information')

parser.add_argument('-cfg_out', type=str, nargs='?',
                    help='File name used to write out the final configuration')

parser.add_argument('-seed', type=int, nargs='?',
                    help='Use the specified random seed used')

parser.add_argument('-N', type=int,
                    help='The size of the NK landscape')


parser.add_argument('-compare', action="store_true",
                    help='Include this flag to compare with SKLearn classifiers')


parser.add_argument('-K', type=int,
                    help='The complexity of the NK landscape')

parser.add_argument('-problem_file', type=str,
                    help='What problem file to use, should contain a list of .npy classes')

parser.add_argument('-training_percentage', type=float, default=0.7,
                    help='Percentage of all data to use during training')

parser.add_argument('-middle_layer', type=str, default="Randomize",
                    help='Processing to do before attaching the ensemble method')

parser.add_argument('-output_node_type', type=str,
                    help='What type of output nodes to use')

parser.add_argument('-ensemble', type=str,
                    help='What type of ensemble to use')

parser.add_argument('-threshold', type=str, default="zero",
                    help='Cutoff used for single class classification')


parser.add_argument('-sample_percentage', type=float,
                    help='Percentage of actual features used by each middle layer node')

parser.add_argument('-target_class', type=str,
                    help='In single class classification, the target class')

parser.add_argument('-linear_classifier',
                    type=str, nargs='?',
                    help='Specify which linear classifier to use')

command_line = vars(parser.parse_args())
config = {}
for configfile in command_line['cfg_input_files']:
    print "Loading", configfile
    with open(configfile, "r") as f:
        config.update(json.load(f))
# This ensures the command line overwrites the config files
config.update(command_line)
try:
    seed = config['seed']
except KeyError:
    seed = np.random.randint(sys.maxint)
    config['seed'] = seed
np.random.seed(seed)

# Saves the configuration to a file so this experiment can be duplicated
if config['cfg_out'] != None and config['cfg_out'] != "none":
    with open(config['cfg_out'], 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

if config['problem_file'] is not None:
    data, target = data_manager.load_problem(config['problem_file'])
    '''
    from sklearn import datasets
    dataset = datasets.load_digits()
    data, target = dataset.data, dataset.target
    # '''

    train, test = even_class_split_dataset(data, target, config['training_percentage'])
    training_data, training_target = train
    testing_data, testing_target = test
else:
    with open("fixed/train_data.npy", "r") as f:
        training_data = np.load(f)
    with open("fixed/train_target.npy", "r") as f:
        training_target = np.load(f)

    with open("fixed/test_data.npy", "r") as f:
        testing_data = np.load(f)
    with open("fixed/test_target.npy", "r") as f:
        testing_target = np.load(f)

# Intermediate processing
middle = all_subclasses(middle_layer.BaseMiddleLayer)[config['middle_layer']](config)
middle.fit(training_data, training_target)
transformed_data = middle.predict(training_data)
print "Transformed Data", transformed_data.shape
classifier = all_subclasses(ensemble_classifier.BaseClassifier)[config['ensemble']](config)

classifier.fit(transformed_data, training_target)

testing_data_transformed = middle.predict(testing_data)
predictions = classifier.predict_using_numbers(testing_data_transformed)
print "Predicted test information"
# TODO Make this more general
from collections import defaultdict
confusion = defaultdict(int)
for prediction, actual in zip(predictions, testing_target):
    confusion[prediction, actual] += 1
print "Classes with at least one correct:", sum(p == a for p, a in confusion.keys())
correct = 0
for pair, count in confusion.items():
    if pair[0] == pair[1]:
        # print pair, count
        correct += count

print "Ensemble:", float(correct) / predictions.shape[0]
if config['compare']:
    from sklearn import linear_model, svm
    # clf = linear_model.LogisticRegression()
    # print "Logistic:", clf.fit(training_data, training_target).score(testing_data, testing_target)
    # clf = svm.LinearSVC()
    # print "BasicSVM:", clf.fit(training_data, training_target).score(testing_data, testing_target)
    clf = linear_model.Perceptron(shuffle=True)
    print "Perceptn:", clf.fit(training_data, training_target).score(testing_data, testing_target)
