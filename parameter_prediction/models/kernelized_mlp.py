"""
Kernelized layers for MLPs
"""
__authors__ = "Misha Denli"
__credits__ = ["Misha Denil"]
__license__ = "3-clause BSD"

from collections import OrderedDict
import numpy as np
import random
import sys
import operator
import warnings
import itertools
import copy

from theano import config
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T

from pylearn2.costs.mlp import Default
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear import conv2d
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace
from pylearn2.utils import function
from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_union
from pylearn2.utils import safe_zip
from pylearn2.utils import sharedX

from pylearn2.models import mlp

from parameter_prediction.kernelize.kernel_expander import KernelExpander, ExpanderParameterizationSequence
from parameter_prediction.kernelize.expander_ops import KernelizedMatrixMul, KernelizedConv2D
import parameter_prediction.kernelize.kernel_expander

class LayerFromBlock(mlp.Layer):
    def __init__(self, block, nhid, layer_name):
        print """





        HELLO USER!


        YOU ARE USING THE WRONG LayerFromBlock CLASS!

        PLEASE STOP THIS MADNESS!


        YOU SHOULD USE THE ONE IN parameter_prediction.models.util INSTEAD!


        THANK YOU FOR YOUR COOPERATION!





        """
        self.block = block
        self.output_space = VectorSpace(nhid)
        self.layer_name = layer_name

    def get_params(self):
        return self.block._params

    def set_input_space(self, space):
        self.input_space = space
        self.desired_space = VectorSpace(self.input_space.get_total_dimension())

    def fprop(self, X):
        X = self.input_space.format_as(X, self.desired_space)
        return self.block(X)


    

class KernelizedLinear(mlp.Linear):
    def __init__(self, kernel_expanders, **kwargs):
        if not isinstance(kernel_expanders, ExpanderParameterizationSequence):
            kernel_expanders = kernel_expanders.generate()
        dim = sum(kep.n_replicas * kep.n_filters_per_replica
                  for kep in kernel_expanders.parameterizations)

        super(KernelizedLinear, self).__init__(dim=dim, **kwargs)
        self.kernel_expanders = kernel_expanders
    
    def set_input_space(self, space):

        self.input_space = space

        
        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.dim + self.copy_input * self.input_dim)

        W_expanders, vs = self.kernel_expanders.generate(
            layer_name=self.layer_name,
            rng=self.mlp.rng)

        self.transformer = KernelizedMatrixMul(W_expanders, vs)
        W, = self.transformer.get_params()
        W.name = "W"


    def get_params(self):
        return self.transformer.get_real_params() + [self.b]

    def get_weights(self):
        return np.concatenate(
                [np.dot(W_expander.get_value(borrow=True), v.get_value(borrow=True))
                    for W_expander, v in zip(self.transformer.W_expanders, self.transformer.vs)],
                axis=1)


class KernelizedSigmoid(mlp.BasicSigmoid, KernelizedLinear):
    def __init__(self, **kwargs):
        super(KernelizedSigmoid, self).__init__(**kwargs)


class KernelizedRectifiedLinear(KernelizedLinear, mlp.BasicRectifiedLinear):
    def __init__(self, init_bias=0, **kwargs):
        super(KernelizedRectifiedLinear, self).__init__(**kwargs)

        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((self.dim,)) + init_bias, name = self.layer_name + '_b')


########################################


class KernelizedConvRectifiedLinear(mlp.ConvRectifiedLinear):
    def __init__(self, kernel_expanders, **kwargs):
        if not isinstance(kernel_expanders, ExpanderParameterizationSequence):
            kernel_expanders = kernel_expanders.generate()
        n_channels = sum(kep.n_replicas * kep.n_filters_per_replica
                  for kep in kernel_expanders.parameterizations)
        
        super(KernelizedConvRectifiedLinear, self).__init__(
            output_channels=n_channels,
            irange=0.0, # HACK: avoid ConvRectifiedLinear complaining about this, we really set it ourselves
            **kwargs)

        self.kernel_expanders = kernel_expanders
        assert self.max_kernel_norm is None

    def set_input_space(self, space):
        self.input_space = space
        rng = self.mlp.rng

        if self.border_mode == 'valid':
            output_shape = [self.input_space.shape[0] - self.kernel_shape[0] + 1,
                            self.input_space.shape[1] - self.kernel_shape[1] + 1]
        elif self.border_mode == 'full':
            output_shape = [self.input_space.shape[0] + self.kernel_shape[0] - 1,
                            self.input_space.shape[1] + self.kernel_shape[1] - 1]

        self.detector_space = Conv2DSpace(
            shape=output_shape,
            num_channels=self.output_channels,
            axes=['b', 'c', 0, 1])

        W_expanders, vs = self.kernel_expanders.generate(
            layer_name=self.layer_name,
            rng=self.mlp.rng)

        self.transformer = KernelizedConv2D(
            W_expanders, vs,
            input_space=self.input_space,
            output_space=self.detector_space,
            kernel_shape=self.kernel_shape,
            batch_size=self.mlp.batch_size,
            subsample=(1,1),
            border_mode=self.border_mode,
            rng=self.mlp.rng)

        W, = self.transformer.get_params()
        W.name = self.layer_name + '_W'

        self.b = sharedX(self.detector_space.get_origin() + self.init_bias)
        self.b.name = self.layer_name + '_b'

        print "Input space:", self.input_space.shape
        print "Detector space:", self.detector_space.shape

        assert self.pool_type in ['max']
        dummy_batch_size = self.mlp.batch_size
        assert dummy_batch_size is not None
        dummy_detector = sharedX(self.detector_space.get_origin_batch(dummy_batch_size))
        dummy_p = mlp.max_pool(
            bc01=dummy_detector,
            pool_shape=self.pool_shape,
            pool_stride=self.pool_stride,
            image_shape=self.detector_space.shape)
        dummy_p = dummy_p.eval()
        self.output_space = Conv2DSpace(
            shape=[dummy_p.shape[2], dummy_p.shape[3]],
            num_channels=self.output_channels,
            axes=['b','c',0,1])

        print "Output space:", self.output_space.shape

    def get_params(self):
        return self.transformer.get_real_params() + [self.b]


    def get_weights_topo(self):
        return self.transformer.get_topo_params()
        

    def get_monitoring_channels(self):
        return []

    def censor_updates(self, updates):
        pass



##############################################
# This stuff probably doesn't work anymore:
#

class TiledBoxExpanderGenerator(object):
    def __init__(self, dim, smoothing, shape, window_shape, kernel):
        self.dim = dim
        self.smoothing = smoothing
        self.kernel = kernel
        self.shape = shape
        self.window_shape = window_shape

    def generate(self):
        tile_locations = itertools.product(
                xrange(0, self.shape[0], self.window_shape[0]),
                xrange(1, self.shape[1], self.window_shape[1]))

        tile_locations = list(tile_locations)
        print "Generating {} expanders".format(len(tile_locations))

        box_processes = []
        for loc in tile_locations:
            bp = BoxSpatialProcess(
                shape=self.shape,
                location=loc,
                window_shape=self.window_shape)
            
            box_processes.append(
                {
                    'n_replicas': 1,
                    'n_filters_per_replica': self.dim,
                    'expander': KernelExpander(
                        spatial_process=bp,
                        smoothing=self.smoothing,
                        kernel=self.kernel)
                })

        return box_processes


class RandomClustered2DPointExpanderGenerator(object):
    def __init__(self, dim, smoothing, shape, stddev, n_points_per, n_expanders, kernel):
        self.dim = dim
        self.smoothing = smoothing
        self.shape = shape
        self.stddev = stddev
        self.n_points_per = n_points_per
        self.n_expanders = n_expanders
        self.kernel = kernel
    
    def generate(self):
        point_locations = [
            [random.randint(1, self.shape[0])-1, random.randint(1, self.shape[1])-1]
            for i in xrange(self.n_expanders)]

        point_processes = []
        for point in point_locations:
            sp = Clustered2DPointSpatialProcess(
                location=point,
                stddev=self.stddev,
                n_points=self.n_points_per,
                shape=self.shape)
            point_processes.append(
                {
                    'n_replicas': 1,
                    'n_filters_per_replica': self.dim,
                    'expander': KernelExpander(
                        spatial_process=sp,
                        smoothing=self.smoothing,
                        kernel=self.kernel)
                })
        return point_processes
