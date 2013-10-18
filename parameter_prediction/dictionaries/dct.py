import numpy as np
import string
from .utils import enumerate_space

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

