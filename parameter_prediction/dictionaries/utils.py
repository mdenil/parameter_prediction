import numpy as np
import itertools

def enumerate_space(extent):
    """Enumerate all the points in a space with the given extent.  Use this
    function to ensure spaces are always enumerated in the same order.
    """
    return itertools.imap(np.asarray, itertools.product(*map(xrange, extent)))

def get_data(inputs):
    """
    Extracts the raw data numpy matrix from the inputs.
    inputs: an instance of Dataset or TransformerDataset.
    """
    from pylearn2.datasets.dataset import Dataset
    from pylearn2.datasets.transformer_dataset import TransformerDataset
    
    if isinstance(inputs, TransformerDataset):
        X = inputs.transformer.perform(inputs.raw)
    else:
        assert isinstance(inputs, Dataset)
        X = inputs.X
    return X

def whiten(data, per_keep=1.1, tol=1e-10, smoothing=0.0, method='pca',
        force_svd=False):
    """
    Whitens the rows of a matrix (via the svd).

    Parameters
    ===========
    data: 2D matrix

        A 2D matrix of data where the rows are instances and the
        columns are features.

    per_keep: scalar, optional, default: 1.1

        Proportion of the variance to keep.  This defaults to slightly
        higher than 1.0 because sometimes rounding errors mean that
        summing up the eigenvalues and normalizing doesn't give exactly
        1.0.

    tol: scalar, optional, default: 1e-10

        Eigenvalues less than tol will be treated as zero.

        NB: If you _really_, don't want to discard dimensions then set
            per_keep > 1 and tol < 0.  You almost certainly don't want
            to do this.

    smoothing: scalar, optional, default: 0.0

        An alternative method for handling small eigenvalues.  The value
        of smoothing is added to all of the eigenvalues to bound them 
        away from zero.  If smoothing > 0.0 then tol is ignored.

    method: string, optional, default: 'pca'

        Can be set to either 'pca' or 'zca' to control the type of
        whitening that will be done.

        Don't discard eigenvalues when you do this.

    Returns
    ========

    dataW: 2D matrix

        The whitened data matrix.

    W: 2D matrix

        The whitening matrix.

        dataW = dot(data - m, W)

    invW: 2D matrix

        The unwhitening matrix.

        data = dot(dataW, invW) + m

    m: vector

        Mean of the data matrix.

    """
    from numpy import *

    m = mean(data, axis=0)
    data = data - m

    if data.shape[0] <= data.shape[1] or force_svd:
        # If data is nxm then U is nxk and V is kxm where k = min(n,m)
        # If n >> m then U is very big and wastes a lot of memory.
        _,s,Vh = linalg.svd(data/sqrt(data.shape[0]-1), full_matrices=0)
        assert all(s >= 0)
        del _
    else:
        w,V = linalg.eigh(dot(data.T,data)/(data.shape[0]-1))
        s = w[::-1]
        assert all(s >= 0)
        s[s>0] = sqrt(s[s>0])
        Vh = V[:,::-1]
        Vh = Vh.T
        del w
        del V

    if smoothing == 0.0:
        keep = (cumsum(s)/sum(s) <= per_keep) & (s > tol)
        s = s[keep]
        Vh = Vh[keep,:]

    if method == 'pca':
        W = dot(Vh.T,diag(1.0/(s + smoothing)))
        invW = dot(diag(s + smoothing), Vh)
    elif method == 'zca':
        W = dot(Vh.T, dot(diag(1.0/(s + smoothing)), Vh))
        invW = dot(Vh.T, dot(diag(s + smoothing), Vh))
    else:
        raise Exception("Invalid whitening method {}".format(repr(method)))

    dataW = dot(data,W)

    return dataW, W, invW, m

def kmeans(X, n_features, batch_size, n_iterations, verbose=False):
    """
    Finds kmeans for X.

    Parameters:
        X: 2D matrix with shape: (n_samples, dimensions)
        config:
            n_features: number of centers (k). NOTE: the final
                k can be less than n_features since this code
                discards far-away centers in the end.
            batch_size: batchsize for training.
            n_iterations: number of iterations for training.
    Returns:
        means: the centers found by kmeans.   
    """
    closest = np.zeros(X.shape[0])

    means = 0.1*np.random.standard_normal(size=(n_features, X.shape[1]))

    for it in xrange(n_iterations):
        if verbose:
            print "kmeans {}/{}".format(it+1, n_iterations)

        half_m2 = 0.5*(means**2).sum(axis=1).reshape((-1,1))

        for i in xrange(0, X.shape[0], batch_size):

            # want
            # argmin_{m_i} ||x - m_i||
            #
            # but,
            # ||x - m_i|| = <x-m_i,x-m_i> = <x,x> - 2<x,m_i> + <m_i,m_i>
            #
            # So,
            # min_{m_i} ||x - m_i|| = <x,x> - ( min_{m_i} 2<x,m_i> - <m_i,m_i> )
            #
            # and therefore,
            # argmin_{m_i} ||x - m_i|| = argmax_{m_i} <x,m_i> - 0.5*<m_i,m_i>

            closest[i:i+batch_size] = np.argmax(
                np.dot(means, X[i:i+batch_size,:].T) - half_m2,
                axis=0)

        for m in xrange(n_features):
            these = closest == m
            if these.sum() > 0:
                means[m,:] = X[these,:].mean(axis=0)

        # remove means that aren't closest to anyone
        neighbourly = np.equal.outer(closest, np.arange(n_features)).sum(axis=0) != 0
        if verbose & (neighbourly.sum() < means.shape[0]):
            print "Lost {} features(s)".format(means.shape[0] - neighbourly.sum())
        means = means[neighbourly, :]

    if verbose:
        print "Finished with {} features (started with {})".format(means.shape[0], n_features)

    return means

