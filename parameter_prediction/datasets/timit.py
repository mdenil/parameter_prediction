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
            X, Y, sentence_ids = self._load_train()

        elif which_set == "test":
            X, Y, sentence_ids = self._load_test()

        elif which_set == "valid":
            X, Y, sentence_ids = self._load_valid()

        elif which_set == "test_valid":
            X1, Y1, sentence_ids_1 = self._load_test()
            X2, Y2, sentence_ids_2 = self._load_valid()

            sentence_ids_2 += 1 + sentence_ids_1.max()

            X = np.concatenate([X1, X2], axis=0)
            Y = np.concatenate([Y1, Y2], axis=0)
            sentence_ids = np.concatenate([sentence_ids_1, sentence_ids_2])

        else:
            raise Exception("TIMIT doesn't understand which_set='{}'".format(which_set))

        super(TIMIT,self).__init__(X=X, y=Y, axes=('b', 0))
        self.sentence_ids = sentence_ids

        assert not np.any(np.isnan(self.X))
        assert not np.any(np.isnan(self.y))
        assert not np.any(np.isnan(self.sentence_ids))

        if preprocessor:
            preprocessor.apply(self)

    def _load_file(self, name):
        data = np.load(name)
        return data['X'], data['Y'], data['sentence_ids'].ravel()

    def _load_train(self):
        n_batches = 5

        Xs = []
        Ys = []
        sentence_ids = []
        for b in xrange(1, n_batches+1):
            fname = os.path.join(
                    string_utils.preprocess("${PYLEARN2_DATA_PATH}"),
                    "timit",
                    "timit_train_b" + str(b) + ".npz")
            X, Y, ids = self._load_file(fname)
            Xs.append(X)
            Ys.append(Y)
            sentence_ids.append(ids)

        X = np.concatenate(Xs, axis=0)
        Y = np.concatenate(Ys, axis=0)
        sentence_ids = np.concatenate(sentence_ids)

        return X, Y, sentence_ids

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

