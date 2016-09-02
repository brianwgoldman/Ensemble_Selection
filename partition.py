from glob import glob
from os import path
from collections import defaultdict
import numpy as np

pattern = path.join('train', 'train_set_*', '*.npy')
out_folder = 'cross_validation'
number_of_groups = 20
dimensions = 2048

files = glob(pattern)

cls_files = defaultdict(list)

def cls_img_from_filename(filename):
    base = path.basename(filename)[:-4]
    return base.split('_')

# hash files by their class
for filename in files:
    cls, _ = cls_img_from_filename(filename)
    cls_files[cls].append(filename)

def even_random_partition(data, number):
   # Break "data" into as equal of sized groups as possible
   full = len(data) / number
   remainder = len(data) % number
   sizes = np.array([full] * number)
   extras = np.random.choice(number, remainder, replace=False)
   sizes[extras] += 1
   assert(sizes.sum() == len(data))
   assert(sizes.max() - sizes.min() <= 1)
   np.random.shuffle(data)
   chunked = []
   used = 0
   for size in sizes:
       chunked.append(data[used: used + size])
       used += size
   return chunked

# groups[i] is the list of files going into parition i
groups = [[] for _ in range(number_of_groups)]
for files in cls_files.values():
    partitions = even_random_partition(files, number_of_groups)
    for i, partition in enumerate(partitions):
        assert(len(partition) > 0)
        groups[i].extend(partition)

for group_number, group in enumerate(groups):
    print "Starting group", group_number + 1, "of", number_of_groups
    np.random.shuffle(group)
    data = np.empty((len(group), dimensions))
    target = [None] * len(group)
    image = [None] * len(group)
    for file_number, filename in enumerate(group):
        cls, img = cls_img_from_filename(filename)
        target[file_number], image[file_number] = cls, img
        with open(filename, 'r') as f:
            data[file_number, :] = np.load(f)
    target = np.array(target)
    image = np.array(image)
    to_format = path.join(out_folder, 'partition_{0:02}_{1}_{2}_')
    base = to_format.format(group_number, len(group), dimensions)
    data_file = base + 'data.npy'
    target_file = base + 'target.npy'
    image_file = base + 'image.npy'
    print "Writing", data_file
    with open(data_file, 'w') as f:
        np.save(f, data)
    print "Writing", target_file
    with open(target_file, 'w') as f:
        np.save(f, target)
    print "Writing", image_file
    with open(image_file, 'w') as f:
        np.save(f, image)
