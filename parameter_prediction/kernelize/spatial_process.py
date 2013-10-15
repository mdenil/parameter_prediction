import numpy as np
import operator
import itertools
import random


def enumerate_space(shape):
    """Enumerate all the points in a space with the given shape.  Use this
    function to ensure spaces are always enumerated in the same order.
    """
    return itertools.imap(np.asarray, itertools.product(*map(xrange, shape)))


# shape is always the shape of the domain to sample over

class RandomUniformSpatialProcess(object):
    def __init__(self, shape, n_points):
        self.shape = shape
        self.n_points = n_points
        self._points = list(itertools.product(*map(xrange, shape)))

    def generate(self):
        return map(np.asarray, random.sample(self._points, self.n_points))

    @property
    def full_shape(self):
        return self.shape

class RandomUniformChannelwiseSpatialProcess(object):
    def __init__(self, shape, n_points, n_channels):
        assert len(shape) == 2
        self._shape = shape
        self.n_points = n_points
        self.n_channels = n_channels

        self._points = list(itertools.product(*map(xrange, shape)))

    def generate(self):
        # Generates coordinates in order 01c.  Deal with it.
        points = []
        spatial_points = random.sample(self._points, self.n_points)
        for channel in xrange(self.n_channels):
            points.extend([np.hstack([p, channel]) for p in spatial_points])
        return points

    @property
    def full_shape(self):
        shape = list(self._shape)
        shape.append(self.n_channels)
        return shape

###########
# fancier spatial processes

class ChannelwiseSpatialProcess(object):
    def __init__(self, n_channels):
        self.n_channels = n_channels

    def generate(self):
        # Generates coordinates in order 01c.  Deal with it.
        points = []
        spatial_points = random.sample(self._points, self.n_points)
        for channel in xrange(self.n_channels):
            points.extend([np.hstack([p, channel]) for p in spatial_points])
        return points

    @property
    def full_shape(self):
        shape = list(self.shape)
        shape.append(self.n_channels)
        return shape


class SparseBox2DPointSpatialProcess(ChannelwiseSpatialProcess):
    def __init__(self, shape, window_shape, at, n_points, n_channels):
        """at == upper left corner"""
        super(SparseBox2DPointSpatialProcess, self).__init__(n_channels)
        self.shape = shape
        self.window_shape = np.asarray(window_shape)
        self.at = np.asarray(at)
        self.n_points = n_points
        
        self._points = [self.at + np.asarray(p) for p in enumerate_space(self.window_shape)]


class FixedLocationClusteredGaussianSpatialProcess(ChannelwiseSpatialProcess):
    def __init__(self, shape, stddev, n_points, n_channels, at):
        """at == centre"""
        super(FixedLocationClusteredGaussianSpatialProcess, self).__init__(n_channels)

        
        self.shape = shape
        self.stddev = stddev
        self.at = np.asarray(at)
        self.n_points = n_points
        self.n_points = int(self.n_points / self.n_channels)
        
        self._points = [
            self.at + stddev * np.random.standard_normal(size=len(shape))
            for p in xrange(self.n_points)
            ]

        self._points = [
            p for p in self._points
            if np.all((0 <= p) & (p < np.asarray(shape)))
            ]

    def generate(self):
        # Generates coordinates in order 01c.  Deal with it.
        points = []
        for channel in xrange(self.n_channels):
            points.extend([np.hstack([p, channel]) for p in self._points])
        return points

class ClusteredGaussianSpatialProcess(ChannelwiseSpatialProcess):
    def __init__(self, shape, stddev, n_points, n_channels):
        """at == centre"""
        super(ClusteredGaussianSpatialProcess, self).__init__(n_channels)
        
        self.shape = shape
        self.stddev = stddev
        self.n_points = n_points

        self.n_points = int(self.n_points / self.n_channels)
        
    def generate(self):
        # Generates coordinates in order 01c.  Deal with it.
        at = np.asarray(map(np.random.randint, self.shape))

        self._points = [
            self.stddev * np.random.standard_normal(size=len(self.shape))
            for p in xrange(self.n_points)
            ]

        _points = [
            (at + p).astype(np.int).astype(np.float) for p in self._points
            ]
        # remove out of bounds points
        _points = [p for p in _points if np.all((0 <= p) & (p < np.asarray(self.shape)))]
        # remove duplicates
        _points = map(np.asarray, set(map(tuple, _points)))

        points = []
        for channel in xrange(self.n_channels):
            points.extend([np.hstack([p, channel]) for p in _points])
        return points


class StridedBoxSpatialProcess(ChannelwiseSpatialProcess):
    def __init__(self, shape, window_shape, at, stride, n_channels):
        """at == upper left corner"""
        super(StridedBoxSpatialProcess, self).__init__(n_channels)
        self.shape = shape
        self.window_shape = np.asarray(window_shape)
        self.at = np.asarray(at)
        self.stride = stride
        self.n_channels = n_channels

        ranges = []
        for sh,st in zip(window_shape, stride):
            ranges.append([0, sh+1, st])

        self._points = [self.at + np.asarray(p)
                        for p in itertools.product(*[xrange(*r) for r in ranges])]


    def generate(self):
        # Generates coordinates in order 01c.  Deal with it.
        points = []
        for channel in xrange(self.n_channels):
            points.extend([np.hstack([p, channel]) for p in self._points])
        return points
