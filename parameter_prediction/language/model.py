import numpy as np
import itertools

def _sliding_window(iterable, n):
    window = []
    for x in iter(iterable):
        window.append(x)
        window = window[-n:]
        if len(window) == n:
            # copy the list or weird things happen
            yield list(window)

def _block_iterator(X, blocks):
    for i in xrange(blocks.max()):
        yield X[blocks == i]

class BigramModel(object):
    def fit(self, X, blocks):
        """
        X: target labels (1d)
        blocks: X[blocks == i] is sentence i
        """
        N = X.max() + 1
        Q = np.zeros((N, N))

        n = 0
        for X_block in _block_iterator(X, blocks):
            for i,j in _sliding_window(X_block, 2):
                Q[i,j] += 1
                n += 1

        self.Q = Q / Q.sum(axis=0)

