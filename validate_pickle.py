from sklearn.externals import joblib
import sys
import numpy as np

pkl_file = sys.argv[1]
output_target_file = pkl_file[:-4] + '_target.npy'
print output_target_file
clf = joblib.load(pkl_file)

with open("fixed/train_data.npy", "r") as f:
    training_data = np.load(f)
with open("fixed/test_data.npy", "r") as f:
    testing_data = np.load(f)

data = np.concatenate([training_data, testing_data])
result = clf.predict(data)
with open(output_target_file, 'w') as f:
    np.save(f, result)
