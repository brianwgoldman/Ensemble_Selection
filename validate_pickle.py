from sklearn.externals import joblib
import sys
import numpy as np

if len(sys.argv) == 1:
    sys.exit()
if sys.argv[1][-4:] == 'pkl':
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
else:
    with open("fixed/train_target.npy", "r") as f:
        training_data = np.load(f)
    with open("fixed/test_target.npy", "r") as f:
        testing_data = np.load(f)
    actual = np.concatenate([training_data, testing_data])
    targets = []
    for target_file in sys.argv[1:]:
        with open(target_file, 'r') as f:
            target = np.load(f)
        print target_file, (target == actual).mean()
        targets.append(target)
    targets = np.array(targets)
    from scipy import stats
    winners, counts = stats.mode(targets)
    print winners
    print winners.shape, counts.mean()
    print (winners == actual).mean()
