import numpy as np
import os
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import string_utils

class TIMIT(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, preprocessor=None):
        """
        which_set can be train, test, valid or test_valid.
        """

        if which_set == "train":
            X, Y = self._load_train()

        elif which_set == "test":
            X, Y = self._load_test()

        elif which_set == "valid":
            X, Y = self._load_valid()

        elif which_set == "test_valid":
            X1, Y1 = self._load_test()
            X2, Y3 = self._load_valid()

            X = np.concatenate([X1, X2], axis=0)
            Y = np.concatenate([Y1, Y2], axis=0)

        else:
            raise Exception("TIMIT doesn't understand which_set='{}'".format(which_set))

        super(TIMIT,self).__init__(X=X, y=Y, axes=('b', 0))

        assert not np.any(np.isnan(self.X))
        assert not np.any(np.isnan(self.y))

        if preprocessor:
            preprocessor.apply(self)

    def _load_file(self, name):
        data = np.load(name)
        return data['X'], data['Y']

    def _load_train(self):
        n_batches = 5

        Xs = []
        Ys = []
        for b in xrange(1, n_batches+1):
            fname = os.path.join(
                    string_utils.preprocess("${PYLEARN2_DATA_PATH}"),
                    "timit",
                    "timit_train_b" + str(b) + ".npz")
            X, Y = self._load_file(fname)
            Xs.append(X)
            Ys.append(Y)

        X = np.concatenate(Xs, axis=0)
        Y = np.concatenate(Ys, axis=0)

        return X, Y

    def _load_test(self):
        fname = os.path.join(
                string_utils.preprocess("${PYLEARN2_DATA_PATH}"),
                "timit",
                "timit_test.npz")
        return self._load_file(fname)

    def _load_valid(self):
        fname = os.path.join(
                string_utils.preprocess("${PYLEARN2_DATA_PATH}"),
                "timit",
                "timit_valid.npz")
        return self._load_file(fname)

