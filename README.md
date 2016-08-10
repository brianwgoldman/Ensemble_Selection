# Ensemble_Selection


## Dependencies
Requires the `scikit learn` python module, which depends on `numpy` and `scipy`.

On the CS machines, you will need to edit your .bashrc to include the following
```
export PYTHONPATH="$PYTHONPATH:/usr/local/numpy-1.11.1/lib64/python2.7/site-packages"
export PYTHONPATH="$PYTHONPATH:/usr/local/scipy-0.17.1/lib64/python2.7/site-packages"
export PYTHONPATH="$PYTHONPATH:/usr/local/scikit-learn-0.17.1_fc24/lib64/python2.7/site-packages"
```

## Getting the data formatted
This will combine all of the images of the same class into a single binary file that numpy knows how to read.

```python data_manager.py create_class_files -input_folder validation/ -output_folder compressed -groundtruth_file validation/groundtruth.txt```

This will create a "problem" with 41 random classes (instead of the full 1000).

```python data_manager.py create_subproblem -input_folder compressed/ -num_classes 41 -output_file problem_with_41_classes.txt```

## Running the code

There are a lot of configuration options, a lot of which don't work together. But this should work:

```python main.py -N 50 -K 4 -problem_file problem_with_41_classes.txt -sample_percentage 0.3 -output_node_type WeightedVote -linear_classifier LogisticRegression```

This will load the problem_file=problem_with_41_classes.txt problem.
It will create a middle layer of N=50 linear_classifier=LogisticRegression classifiers, each using sample_percentage=0.3 percent
of the problem variables drawn at random. These will create an NK landscape with K=4, with each output node using an
output_node_type=WeightedVote to combine its inputs.
