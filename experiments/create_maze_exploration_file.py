"""
First, gets a file iterator.
Then, takes a pickle file and turns it into an "exploration amount" thing through
a hash table.
Finally, writes that somewhere.
"""

import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path
import os


def get_filename_iterator(read_dir):
    """Gets filename, and the data object, and return an iterator"""
    files = [f for f in Path(read_dir).iterdir() if f.is_file()]
    for file in files:
        name = file.name
        with file.open("rb") as f:
            data = pickle.load(f)

        yield name, data

def convert_data_to_exploration_amounts(data, box_size=0.25):
    states = data['states']
    counter = set([])
    exploration_tracker = []
    for ep in states:
        for state in ep:
            scaled_state = state / box_size
            scaled_tup = tuple(np.floor(scaled_state).astype('int'))
            counter.add(scaled_tup)
        exploration_tracker.append(len(counter))
    # print('how did we do')
    # __import__('ipdb').set_trace()
    return exploration_tracker


def create_grid_exploration_files(experiment_name, run_title):
    read_dir = os.path.join(experiment_name, run_title, "stored_states")
    write_dir = os.path.join(experiment_name, run_title, "grid_exploration_amounts")

    os.makedirs(write_dir, exist_ok=True)
    for name, data in get_filename_iterator(read_dir):
        print(name)
        exploration_tracker = convert_data_to_exploration_amounts(data)
        write_filename = os.path.join(write_dir, name)
        print(write_filename)
        with open(os.path.join(write_dir, name), "wb") as f:
            pickle.dump(exploration_tracker, f)



def all_run_titles(experiment_name):
    parent = Path(experiment_name)
    run_titles = [d.name for d in parent.iterdir()]
    print(run_titles)
    return run_titles


if __name__ == '__main__':
    experiment_name = "./remote_plots/slurm_plots/point_maze/rnd_sweep"
    run_titles = all_run_titles(experiment_name)
    for rt in run_titles:
        create_grid_exploration_files(experiment_name, rt)
