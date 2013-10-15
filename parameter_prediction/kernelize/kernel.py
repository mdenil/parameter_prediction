import numpy as np


from scipy.special import gammaln
from scipy.constants import pi

def mean_random_uniform_point_distance(n, d, k): 
    """The mean distance from a given point to its kth nearest neighbour in a
    unit volume of d dimensional euclidian space containing exactly n points
    (including the reference point).

    http://mathoverflow.net/questions/22592/mean-minimum-distance-for-k-random-points-on-a-n-dimensional-hyper-cube

    http://arxiv.org/pdf/math/0212230.pdf
    
    """
    lnt0 = (1./d)*gammaln(d/2. + 1.) - 0.5*np.log(pi)
    lnt1 = gammaln(k + 1./d) - gammaln(k)
    lnt2 = gammaln(n) - gammaln(n + 1./d)

    return np.exp(lnt0 + lnt1 + lnt2)



class SquaredExponentialKernel(object):
    def __init__(self, scale, mask=None):
        try:
          iter(scale)
          scale_iterable = True
        except TypeError as e:
          scale_iterable = False
        
        
        if scale_iterable:
          if scale[0] == 'auto':
            self.scale = scale[2]*mean_random_uniform_point_distance(scale[1], 2, 1)
          else:
            self.scale = 1.0
        else:
            self.scale = scale
        
        if mask is not None:
            self.corr_mask = np.asarray(mask, dtype=np.bool)
            self.ind_mask = np.logical_not(self.corr_mask)

            # If x[ind_mask] != y[ind_mask] then x and y have zero correlation.
            # only one dimension can be specified for ind_mask
            assert np.sum(self.ind_mask) == 1

    def apply(self, xs, ys):
        if not hasattr(self, 'corr_mask'):
            self.corr_mask = np.asarray([True]*xs.shape[1])
            self.ind_mask = np.asarray([False]*xs.shape[1])
        
        xs_corr = xs[:, self.corr_mask]
        ys_corr = ys[:, self.corr_mask]
        
        x2 = np.sum(xs_corr**2, axis=1).reshape((-1,1))
        y2 = np.sum(ys_corr**2, axis=1).reshape((-1,1))
        xy = np.dot(xs_corr, ys_corr.T)
        K = np.exp(-0.5 * (x2 + y2.T - 2*xy) / self.scale**2)

        if np.any(self.ind_mask):
            K_ind = np.logical_not(
                np.equal.outer(xs[:,self.ind_mask].ravel(),
                               ys[:,self.ind_mask].ravel()))
            K[K_ind] = 0

        return K
        
        
