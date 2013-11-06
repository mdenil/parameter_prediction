import numpy as np

from parameter_prediction.dictionaries.explicit import ExplicitDictionary
from parameter_prediction.dictionaries.utils import *

class AutoencoderDictionary(ExplicitDictionary):
    def __init__(self, ae_model):
        """
        ae_model should be a pylearn2 autoencoder model.
        """
        self.W = self._get_W_from_model(ae_model)

    def _get_W_from_model(self, ae_model):
        """
        Extracts dictionary weights from an autoencoder model.
        """

        # Make sure the weights in the autoencoder are tied. 
        assert ae_model.tied_weights == True
        
        W = ae_model.get_weights().T
        
        #Sanity check.
        assert W.shape == (ae_model.nhid, ae_model.input_space.dim)

        return W

class CovarianceDictionary(ExplicitDictionary):
    """
    Creates a dictionary using empirical covariance of the input data.
    """
    def __init__(self, inputs):
        """
        inputs : an instance of Dataset or TransformerDataset.
        """
        self.W = np.cov(get_data(inputs).T)

class KmeansDictionary(ExplicitDictionary):
    """
    Creates a dictionary using kmeans centers.
    """
    def __init__(self, inputs, kmeans_opts=None, whitening_opts=None):
        """
        inputs: an instance of Dataset or TransformerDataset.
        k: number of kmeans centers.
        whitening: if true the input data is whitened before finding running
            the kmeans.
        """
        X = get_data(inputs)
        if whitening_opts != None:
            [X_white, _, invV, m] = whiten(X, **whitening_opts)
            W_white = kmeans(X_white, **kmeans_opts)
            self.W = np.dot(W_white, invV) + m
        else:
            self.W = kmeans(X, **kmeans_opts)















