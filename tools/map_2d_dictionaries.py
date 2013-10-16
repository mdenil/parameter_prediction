import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from parameter_prediction.datasets import dictionary

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def map_2d_dictionary(ax, dictionary):
    # cannonical order for dictionary elements is row major
    atoms = [
        np.reshape(dictionary.get_atom(i), dictionary.extent)
        for i in xrange(dictionary.size)]

    rows = grouper(atoms, int(dictionary.extent[0]))
    rows = [np.concatenate(row, axis=1) for row in rows]
    full = np.concatenate(rows, axis=0)

    ax.pcolor(full, cmap=plt.cm.gray)
    ax.set_xlim([0, full.shape[0]])
    ax.set_ylim([0, full.shape[1]])

    return ax

if __name__ == "__main__":
    output_dir = "scratch"

    extent = [8,8]

    dictionaries = [
        dictionary.DCTDictionary(extent),
        dictionary.GaussianKernelDictionary(extent, 1.0),
        ]

    for d in dictionaries:
        fig = plt.figure()
        map_2d_dictionary(fig.gca(), d)
        fig.savefig(os.path.join(output_dir, d.__class__.__name__ + ".png"))

