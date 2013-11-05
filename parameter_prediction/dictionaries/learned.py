from parameter_prediction.dictionaries.explicit import ExplicitDictionary

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

