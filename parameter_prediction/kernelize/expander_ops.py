import numpy as np
import theano.tensor as T

from pylearn2.linear.conv2d import Conv2D
from pylearn2.linear.linear_transform import LinearTransform

from parameter_prediction.kernelize.kernel_expander import expand_to_matrix_operator
from parameter_prediction.kernelize.kernel_expander import expand_to_matrix_operator_numpy
from parameter_prediction.kernelize.kernel_expander import expand_to_filter_stack
from parameter_prediction.kernelize.kernel_expander import expand_to_filter_stack_numpy


class KernelizedMatrixMul(LinearTransform):
    def __init__(self, W_expanders, vs):
        self.W_expanders = W_expanders
        self.vs = vs
        self.W_full = T.concatenate(map(
            expand_to_matrix_operator,
            self.W_expanders, self.vs),
            axis=1)

    def get_params(self):
        return [ self.W_full ]

    def get_real_params(self):
        return self.vs

    def lmul(self, x):
        return T.dot(x, self.W_full)

    def lmul_T(self, x):
        return T.dot(x, self.W_full.T)



def _n_out_channels_block(W_expander, v, kernel_shape, input_space):
    n_out_channels = W_expander.get_value(borrow=True).shape[0] * v.get_value(borrow=True).shape[1]
    n_out_channels /= kernel_shape[0] * kernel_shape[1] * input_space.num_channels
    return n_out_channels

class KernelizedConv2D(Conv2D):
    def __init__(
            self,
            W_expanders,
            vs,
            input_space,
            output_space,
            kernel_shape,
            batch_size,
            subsample,
            border_mode,
            rng):

        self.input_space = input_space
        self.output_space = output_space
        self.kernel_shape = kernel_shape
        self.batch_size = batch_size
        
        self.W_expanders = W_expanders
        self.vs = vs

        Ws = []
        Wnps = []
        for W_expander, v, in zip(self.W_expanders, self.vs):
            n_out_channels = _n_out_channels_block(W_expander, v, self.kernel_shape, self.input_space)
            
            W = expand_to_filter_stack(
                W_expander, v,
                shape=kernel_shape,
                n_channels=input_space.num_channels,
                batch_size=n_out_channels,
                axes=output_space.axes)
            Ws.append(W)

            print self.output_space.axes

            # Debugging only
            # Wnp = expand_to_filter_stack_numpy(
            #     W_expander.get_value(borrow=True), v.get_value(borrow=True),
            #     shape=kernel_shape,
            #     n_channels=input_space.num_channels,
            #     batch_size=n_out_channels,
            #     axes=output_space.axes)
            # Wnps.append(Wnp)

        W = T.concatenate(Ws, axis=output_space.axes.index('b'))

        # Debugging only
        # Wnp = np.concatenate(Wnps, axis=output_space.axes.index('b'))
        # np.savez("Wnp", Wnp=Wnp)

        filters_shape = [
            output_space.num_channels,
            input_space.num_channels,
            kernel_shape[0],
            kernel_shape[1]
            ]

        super(KernelizedConv2D, self).__init__(
            filters=W,
            batch_size=batch_size,
            input_space=input_space,
            output_axes=output_space.axes,
            subsample=subsample,
            border_mode=border_mode,
            filters_shape=filters_shape,
            message="")
        
    def get_topo_params(self):
        Ws = []
        for W_expander, v in zip(self.W_expanders, self.vs):
            n_out_channels = _n_out_channels_block(W_expander, v, self.kernel_shape, self.input_space)

            W = expand_to_filter_stack_numpy(
                W_expander.get_value(borrow=True), v.get_value(borrow=True),
                shape=self.kernel_shape,
                n_channels=self.input_space.num_channels,
                batch_size=n_out_channels,
                axes=self.output_space.axes)
            Ws.append(W)
        W = np.concatenate(Ws, axis=self.output_space.axes.index('b'))
        W = W.transpose([self.output_space.axes.index(x) for x in ['b',0,1,'c']])
        return W


    def get_real_params(self):
        return self.vs





class LowRankMatrixMul(LinearTransform):
    def __init__(self, U, V):
        self.U = U
        self.V = V
        self.W = T.dot(self.U, self.V)
        
    def get_params(self):
        return [ self.W ]

    def get_real_params(self):
        return [self.U, self.V]

    def lmul(self, x):
        return T.dot(x, self.W)

    def lmul_T(self, x):
        return T.dot(x, self.W.T)
