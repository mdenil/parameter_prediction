"""
Low rank layers for MLPs.
"""

__authors__ = "Misha Denil"
__credits__ = ["Misha Denil"]
__license__ = "3-clause BSD"

import numpy as np
import theano.tensor as T

from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX


from pylearn2.models import mlp
from pylearn2.linear.linear_transform import LinearTransform
from parameter_prediction.kernelize.expander_ops import LowRankMatrixMul


class LowRankLinear(mlp.Linear):
    def __init__(self, rank, **kwargs):
        super(LowRankLinear, self).__init__(**kwargs)
        self.rank = rank

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

        rng = self.mlp.rng

        U = sharedX(rng.uniform(
            -self.irange, self.irange,
            [self.input_space.get_total_dimension(), self.rank]))
        U.name = self.layer_name + "_U"
        
        V = sharedX(rng.uniform(
            -self.irange, self.irange,
            [self.rank, self.output_space.get_total_dimension()]))
        V.name = self.layer_name + "_V"
        
        self.transformer = LowRankMatrixMul(U, V)
        W, = self.transformer.get_params()
        W.name = self.layer_name + "_W"

    def get_params(self):
        return self.transformer.get_real_params() + [self.b]

    def get_weights(self):
        return np.dot(self.transformer.U.get_value(borrow=True),
                      self.transformer.V.get_value(borrow=True))

class LowRankLinearFixedU(LowRankLinear):
    def get_params(self):
        return self.transformer.get_real_params()[1:] + [self.b]


class LowRankSigmoid(mlp.BasicSigmoid, LowRankLinear):
    def __init__(self, **kwargs):
        super(LowRankSigmoid, self).__init__(**kwargs)

class LowRankSigmoidFixedU(mlp.BasicSigmoid, LowRankLinearFixedU):
    pass

