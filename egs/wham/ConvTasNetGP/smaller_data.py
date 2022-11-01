import os
import json
import sys

import numpy as np

random_seed = 111214
basedir = "data/wav16k"
setname = "min"
subsetnames = ["cv", "tr", "tt"]
filenames = ["mix_both", "mix_clean", "mix_single", "noise", "s1", "s2"]
data_percentages = [1, 5, 10, 20]

if not os.path.exists(os.path.join(basedir, setname)):
    sys.exit("{} is not found!".format(os.path.join(basedir, setname)))

for subsetname in subsetnames:
    for filename in filenames:
        json_filepath = os.path.join(
            basedir, setname, subsetname, filename + '.json')
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        # in-place sorting
        data.sort(key=lambda x: x[0])
        # in-place shuffling
        np.random.RandomState(random_seed).shuffle(data)

        for data_percentage in data_percentages:
            # get a smaller set
            _data = data[:int(len(data) * data_percentage / 100)]
            # in-place sorting
            _data.sort(key=lambda x: x[0])
            new_json_filepath = os.path.join(
                basedir, setname + "{:02d}pct".format(data_percentage),
                subsetname, filename + '.json')
            os.makedirs(os.path.dirname(new_json_filepath), exist_ok=True)
            with open(new_json_filepath, 'w') as f:
                json.dump(_data, f, indent=4)