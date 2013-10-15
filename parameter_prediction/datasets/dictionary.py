

class ExplicitDictionary(object):
    def __init__(self, W):
        """
        W should be a numpy array.
        The dictionary is composed of columns of W.
        """
        self.W = W

    def get_subdictionary(self, indices):
        return self.W[:,indices]

    @property
    def input_dim(self):
        return self.W.shape[0]

    @property
    def size(self):
        return self.W.shape[1]
