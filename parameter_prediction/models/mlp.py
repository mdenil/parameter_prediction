import theano.tensor as T
from pylearn2.models import mlp
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.linear.matrixmul import MatrixMul
from collections import OrderedDict

#class PureLinear(mlp.Linear):
#    def cost(self, *args, **kwargs):
#        raise NotImplementedError()
#
#    def fprop(self, state_below):
#        return self._linear_part(state_below)

#class AutoencoderLayer(mlp.Linear):
#    def set_input_space(self, space):
#        self.input_space = space
#        self.output_space = space
#
#        if isinstance(space, VectorSpace):
#            self.requires_reformat = False
#            self.input_dim = space.dim
#        else:
#            self.requires_reformat = True
#            self.input_dim = space.get_total_dimension()
#            self.desired_space = VectorSpace(self.input_dim)
#
#        rng = self.mlp.rng
#        W = rng.uniform(-self.irange, self.irange, (self.input_dim, self.dim))
#        W = sharedX(W)
#        W.name = self.layer_name + "_W"
#
#        self.transformer = MatrixMul(W)
#
#    def fprop(self, state_below):
#        return self.reconstruct(state_below)
#
#    def upward_pass(self, state_below):
#        return T.nnet.sigmoid(self.transformer.lmul(state_below) + self.b)
#
#    def downward_pass(self, state_above):
#        return self.transformer.lmul_T(state_above)
#
#    def reconstruct(self, state_below):
#        return self.downward_pass(self.upward_pass(state_below))
#
#class PretrainedAutoencoderLayer(mlp.PretrainedLayer):
#    def get_output_space(self):
#        return VectorSpace(self.layer_content.dim)

class CompositeLayer(mlp.CompositeLayer):
    @property
    def dim(self):
        return sum(layer.dim for layer in self.layers)

class PretrainedLayer(mlp.PretrainedLayer):
    def fprop(self, *args, **kwargs):
        return self.layer_content.fprop(*args, **kwargs)

    def get_weight_decay(self, coeff):
        return self.layer_content.get_weight_decay(coeff)

    def get_l1_weight_decay(self, coeff):
        return self.layer_content.get_weight_decay(coeff)

class SubsampledDictionaryLayer(mlp.Layer):
    def __init__(self, dim, layer_name, dictionary):
        self.dim = dim
        self.layer_name = layer_name
        self.dictionary = dictionary

    def fprop(self, state_below):
        self.input_space.validate(state_below)

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below)

        return z

    def set_input_space(self, space):
        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.dim)

        self.rng = self.mlp.rng

        # sanity checking
        assert self.dictionary.input_dim == self.input_dim
        assert self.dictionary.size >= self.dim

        indices = self.rng.permutation(self.dictionary.size)
        indices = indices[:self.dim]
        indices.sort()

        W = self.dictionary.get_subdictionary(indices)

        # dictionary atoms are stored in rows but transformers expect them to
        # be in columns.
        W = sharedX(W.T)
        W.name = self.layer_name + "_W"
        self.transformer = MatrixMul(W)

    # This is a static layer, there is no cost, no parameters, no updates, etc, etc
    def get_params(self):
        return []
        #return self.transformer.get_params()

    def cost(self, Y, Y_hat):
        return 0.0

    def cost_from_cost_matrix(self, cost_matrix):
        return 0.0

    def cost_matrix(self, Y, Y_hat):
        return T.zeros_like(Y)

    def get_weight_decay(self, coeff):
        return 0.0

    def get_l1_weight_decay(self, coeff):
        return 0.0
