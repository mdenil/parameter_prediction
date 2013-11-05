import numpy as np
import theano.tensor as T
from pylearn2.base import Block
from pylearn2.models import Model
from pylearn2.space import VectorSpace

def _identity(x):
    return x
def _rectified(x):
    return x * (x > 0)

# Don't put lambda functions in here or pickle will yell at you when you try to
# save an Autoencoder.
DECODER_FUNCTION_MAP = {
    'linear': _identity,
    'rectified': _rectified,
    'sigmoid': T.nnet.sigmoid,
    }

class Autoencoder(Block, Model):
    """
    A basic Autoencoder that uses a Layer for the upward pass.

    The reconstruction function is:

        x_recon = act_dec(layer.transformer.lmul_T(layer.fprop(x)))

    act_dec is specified as a string and can be any of the following:

        'linear': act_dec(x) = x
        'rectified': act_dec(x) = x * (x > 0)
        'sigmoid' : act_dec(x) = T.nnet.sigmoid(x)

    """
    def __init__(self, nvis, layer, act_dec='linear', seed=None):
        super(Autoencoder, self).__init__()

        self.input_space = VectorSpace(nvis)
        self.output_space = VectorSpace(layer.dim)

        self.act_dec = DECODER_FUNCTION_MAP[act_dec]

        # self is not really an mlp, but the only thing layer.mlp is used for
        # internally is getting access to rng, which we have
        self.rng = np.random.RandomState(seed)
        layer.mlp = self

        layer.set_input_space(self.input_space)
        self.layer = layer

    def upward_pass(self, inputs):
        return self.encode(inputs)

    def encode(self, inputs):
        return self.layer.fprop(inputs)

    def decode(self, hiddens):
        return self.act_dec(self.layer.transformer.lmul_T(hiddens))

    def reconstruct(self, inputs):
        return self.decode(self.encode(inputs))

    def get_weights(self, borrow=False):
        W, = self.layer.transformer.get_params()
        return W.get_value(borrow=borrow)

    def get_weights_format(self):
        return ['v', 'h']

    def get_params(self):
        return self.layer.get_params()

    def __call__(self, inputs):
        return self.encode(inputs)

    # Use version defined in Model, rather than Block (which raises
    # NotImplementedError).
    get_input_space = Model.get_input_space
    get_output_space = Model.get_output_space

