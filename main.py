import nk
import argparse
import sys
import numpy as np
import sklearn
from ensemble_classifier import EnsembleClassifier
# Set up argument parsing
description = 'Ensemble Selection using NK Landscapes'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('configs', metavar='Configuration Files',
                    type=str, nargs='*',
                    help='Zero or more json formatted files containing' + ' configuration information')

parser.add_argument('-seed', dest='seed', type=int, nargs='?',
                    help='Use the specified random seed used')

parser.add_argument('-K', dest='K', type=int,
                    help='The complexity of the NK landscape')

parser.add_argument('-output_node_type', dest='output_node_type', type=str,
                    help='What type of output nodes to use')

parser.add_argument('-threshold', dest='threshold', type=float,
                    help='Cutoff used for single class classification')


parser.add_argument('-target_class', dest='target_class', type=str,
                    help='In single class classification, the target class')

parser.add_argument('-linear_classifier', dest='linear_classifier',
                    type=str, nargs='?',
                    help='Specify which linear classifier to use')

config = vars(parser.parse_args())
# TODO Read all config files into config

try:
    seed = config['seed']
except KeyError:
    seed = np.random.randint(sys.maxint)
    config['seed'] = seed
np.random.seed(seed)

# TODO Save config to a file for duplication purposes

# TODO Replace this with proper loading of data files
from sklearn import datasets
digits = datasets.load_digits()

training_data = digits.data
training_target = np.array(map(str, digits.target))

config['N'] = training_data.shape[1]

# TODO Here you should do intermediate processing

classifier = EnsembleClassifier(config)

# TODO Time between each of these
classifier.build_nk_table(training_data, training_target)
print "Table Built"
classifier.optimize_nk()
print "NK Optimized"
classifier.configure_outputs()
print "Configured ensemble"

# TODO Load test data for real
testing_data = training_data
testing_target = training_target

predictions = classifier.predict(testing_data)
print "Predicted test information"
# TODO Make this more general
from collections import defaultdict
confusion = defaultdict(int)
for prediction, actual in zip(predictions, testing_target):
    confusion[prediction, actual] += 1
correct = 0
for pair, count in confusion.items():
    print pair, count
    if pair[0] == pair[1]:
        correct += count

print float(correct) / predictions.shape[0]
