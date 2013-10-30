import numpy as np
import os
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import string_utils

class TIMIT(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, combine_stress=False, preprocessor=None):
        """
        which_set can be 'train', 'test', 'valid' (dev) or 'test_valid'.
        """

        # A list mapping a phone index to the name of that phone (indexes are with inflections combined)
        self.phone_index = self._load_phone_index()

        # A table with the standard rules for folding phone classes
        self.fold_table = self._get_fold_table()

        # Multiply raw label vectors on the right by this matrix to combine
        # different inflections of the same phone
        self.phone_stress_combining_matrix = np.kron(np.eye(61), np.array([[1],[1],[1]]))

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

        if combine_stress:
            self.stress_combined = True
            Y = np.dot(Y, self.phone_stress_combining_matrix)
        else:
            self.stress_combined = False

        super(TIMIT,self).__init__(X=X, y=Y, axes=('b', 0))
        self.sentence_ids = sentence_ids

        assert not np.any(np.isnan(self.X))
        assert not np.any(np.isnan(self.y))
        assert not np.any(np.isnan(self.sentence_ids))

        if preprocessor:
            preprocessor.apply(self)

    def _load_data_file(self, name):
        data = np.load(name)
        return data['X'], data['Y'], data['sentence_ids'].ravel()

    def _load_phone_index(self):
        file_name = os.path.join(
                string_utils.preprocess("${PYLEARN2_DATA_PATH}"),
                "timit",
                "timit_phone_index_table.txt")

        phones = []
        with open(file_name) as timit_file:
            for line in timit_file:
                idx, label = line.split(",")
                idx = int(idx) - 1 # -1 because 1-based indexing
                phone, junk = label.strip().split(" ")

                if phone not in phones:
                    phones.append(phone)

        return phones

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
            X, Y, ids = self._load_data_file(fname)
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
        return self._load_data_file(fname)

    def _load_valid(self):
        fname = os.path.join(
                string_utils.preprocess("${PYLEARN2_DATA_PATH}"),
                "timit",
                "timit_valid.npz")
        return self._load_data_file(fname)

    def _get_fold_table(self):
        # Source (KFL): http://repository.cmu.edu/cgi/viewcontent.cgi?article=2768&context=compsci
        #
        # The fold table maps name -> folded name
        # Names not in the fold_table aren't changed by folding.
        return {
            # Folding rules from the folding table in KFL.
            'ux': 'uw',
            'axr': 'er',
            'em': 'm',
            'nx': 'n',
            'eng': 'ng',
            'hv': 'hh',

            # Folding rules for closures and silence.
            #
            # KFL folds each type of closure and silence separately and then
            # ignores confusion between the different folded categories, we just
            # fold them all together into silence.
            'pcl': 'h#',
            'tcl': 'h#',
            'kcl': 'h#',
            'qcl': 'h#',
            'bcl': 'h#',
            'dcl': 'h#',
            'gcl': 'h#',
            'epi': 'h#',
            'pau': 'h#',
            '#h': 'h#',
            
            # These are not folded in KFL, but are part of groups where within
            # group confusions are ignored.  We handle this by picking an arbitrary
            # member of the group and folding the others into it.
            'l': 'el',
            'en': 'n',
            'zh': 'sh',
            'aa': 'ao',
            'ix': 'ih',
            'ax': 'ah',

            # KFL doesn't mention ax-h, but this fold seems reasonable and if we
            # don't do it we get the wrong number of folded phones at the end.
            'ax-h': 'ah',

            # in LFK q is "removed". I don't know what this means so I'm just going
            # to fold it into silence.
            'q': 'h#'
            }

