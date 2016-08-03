import nk
import output_node
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
parser.add_argument('-seed', dest='seed', type=int,
                    help='Use the specified random seed used')

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
training_target = digits.target

# TODO Here you should do intermediate processing

classifier = EnsembleClassifier(config)

# TODO Time between each of these
classifier.build_nk_table()
classifier.optimize_nk()
classifier.configure_outputs()

# TODO Load test data for real
testing_data = training_data
testing_target = training_target

predictions = classifier.predict(testing_data)

# TODO Make this more general
from collections import defaultdict
confusion = defaultdict(int)
for prediction, actual in zip(predictions, testing_target):
    confusion[prediction, actual] += 1
for pair, count in confusion.items():
    print pair, count
