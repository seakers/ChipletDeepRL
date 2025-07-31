import numpy as np

def design_to_chiplet_values(partitions):
    partitions = np.sort(np.array(partitions, dtype=int))
    chiplet_vals = np.concatenate((partitions, [12])) - np.concatenate(([0], partitions))
    return chiplet_vals