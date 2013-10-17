import numpy as np
import itertools
import string

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

class IndexSpaceDictionary(object):
    def __init__(self, extent):
        self.extent = np.asarray(extent).reshape((-1,1))

    def get_subdictionary(self, indices):
        return np.vstack([self.get_atom(index) for index in indices])

    def get_atom(self, index):
        raise NotImplementedError

    @property
    def input_dim(self):
        return np.prod(self.extent)

    @property
    def size(self):
        return np.prod(self.extent)


class GaussianKernelDictionary(IndexSpaceDictionary):
    def __init__(self, extent, scale):
        super(GaussianKernelDictionary, self).__init__(extent)

        # _points is every coordinate in the (discrete) space indexed in
        # cannonical order.  Each column gives the coordinates of a different
        # point.
        self._points = np.vstack(enumerate_space(self.extent)).T

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


    def get_atom(self, index):
        point = np.atleast_2d(self._points[:,index]).T
        E = np.sum((self._points - point)**2 / self.scale**2, axis=0)
        atom = np.exp(-0.5 * E)
        return atom 

class DCTDictionary(IndexSpaceDictionary):
    def __init__(self, spatial_extent, frequency_extent):
        self.spatial_extent = np.atleast_2d(spatial_extent).T
        self.frequency_extent = np.atleast_2d(frequency_extent).T

        # _points is every coordinate in (discrete) frequency space indexed in
        # cannonical order.  Each column gives the coordinates of a different
        # point.
        self._points = np.vstack(enumerate_space(self.frequency_extent)).T

    def get_atom(self, index):
        # http://en.wikipedia.org/wiki/Discrete_cosine_transform#Multidimensional_DCTs

        point = self._points[:,index]

        dims = []
        for i,e in enumerate(self.spatial_extent):
            n = np.atleast_2d(np.arange(e))
            dims.append(np.cos(np.pi / self.spatial_extent[i,0] * (n+0.5) * point[i]).ravel())

        # set up einsum for an outer product
        #
        # this will fail with more than len(alphabet) dimensions (lol wtf are you doing?)
        from_idx = ",".join(string.ascii_lowercase[:len(dims)])
        to_idx = string.ascii_lowercase[:len(dims)]

        atom_topo = np.einsum(from_idx + "->" + to_idx, *dims)

        return atom_topo.ravel()

    @property
    def input_dim(self):
        return np.prod(self.spatial_extent)

    @property
    def size(self):
        return np.prod(self.frequency_extent)

