from pylearn2.space import VectorSpace
from pylearn2.models import mlp

class LayerFromBlock(mlp.Layer):
    def __init__(self, block, nhid, layer_name):
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

