
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


