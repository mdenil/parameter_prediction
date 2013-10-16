import numpy as np
import itertools

"""
Dictionaries are (either implicitly or explicitly) a matrix where the rows are
dictionary atoms.

All dictionaries must implement the following interface:

@property input_dim:
    the dimension of each atom
@property size:
    the number of atoms in the dictionary

get_subdictionary(indices):
    indices is a sequence of integers which index elements of the dictionary.
    This method must return an explicit representation of the subdictionary
    formed by extracting the atoms at the corresponding indices.

    The explicit representation must be a numpy array with dimensions
    len(indices) x input_dim.

"""

class ExplicitDictionary(object):
    def __init__(self, W):
        """
        W should be a numpy array.
        The dictionary is composed of rows of W.
        """
        self.W = W

    def get_subdictionary(self, indices):
        return self.W[indices]

    @property
    def input_dim(self):
        return self.W.shape[1]

    @property
    def size(self):
        return self.W.shape[0]



def enumerate_space(extent):
    """Enumerate all the points in a space with the given extent.  Use this
    function to ensure spaces are always enumerated in the same order.
    """
    return itertools.imap(np.asarray, itertools.product(*map(xrange, extent)))


class GaussianKernelDictionary(object):
    def __init__(self, extent, scale):
        try:
            iter(scale)
            scale_iterable = True
        except TypeError:
            scale_iterable = False

        if scale_iterable:
            assert len(extent) == len(scale)
        else:
            scale = [scale]*len(extent)

        self.scale = np.asarray(scale).reshape((-1,1))
        self.extent = np.asarray(extent)
        self.__points = np.vstack(enumerate_space(self.extent)).T

    def get_subdictionary(self, indices):
        return np.vstack([self.get_atom(index) for index in indices])

    def get_atom(self, index):
        point = np.atleast_2d(self.__points[:,index]).T
        E = np.sum((self.__points - point)**2 / self.scale**2, axis=0)
        atom = np.exp(-0.5 * E)
        return atom 

    @property
    def input_dim(self):
        return np.prod(self.extent)

    @property
    def size(self):
        return np.prod(self.extent)

