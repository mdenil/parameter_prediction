import numpy as np
from .utils import enumerate_space

class GaussianKernelDictionary(object):
    def __init__(self, extent, scale):
        self.extent = np.asarray(extent).reshape((-1,1))

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

    def get_subdictionary(self, indices):
        return np.vstack([self.get_atom(index) for index in indices])

    def get_atom(self, index):
        point = np.atleast_2d(self._points[:,index]).T
        E = np.sum((self._points - point)**2 / self.scale**2, axis=0)
        atom = np.exp(-0.5 * E)
        return atom 

    @property
    def input_dim(self):
        return np.prod(self.extent)

    @property
    def size(self):
        return np.prod(self.extent)

