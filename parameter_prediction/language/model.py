import numpy as np
import daicrf
from parameter_prediction.util import iterator

class BigramModel(object):
    def fit(self, X, blocks):
        """
        X: target labels (1d integers)
        blocks: X[blocks == i] is sentence i
        """
        N = X.max() + 1
        Q = np.zeros((N, N))

        n = 0
        for X_block in iterator.blocks(X, blocks):
            for i,j in iterator.sliding_window(X_block, 2):
                Q[i,j] += 1
                n += 1

        self.Q = Q / Q.sum(axis=0)

    def predict(self, X_prob, blocks):
        """
        X_prob: X_prob[i,j] is the probability that phone i has label j.
        blocks: X_prob[blocks == i] is sentence i

        Returns X_decoded which is a (1d) label vector where X_decoded[i] is
        the label assigned to phone i by viterbi decoding.

        The model used for decoding a sentence with n phones is a CRF with the
        following structure:

                 /-------+-------+------------ Edge potentials from language model (this object)
                 |       |       |
                 v       v       v
            X_1 --- X_2 --- ... --- X_n    <-- P(X_n=j | Y_n) provided in X_prob
             ^       ^               ^
             |       |               |
            P_1     P_2             P_n    <-- Phone observations (never seen by this model)

        Decoding is done separately for each sentence.  The potentials for each
        X_i -- X_{i+1} edge are the same.
        
        """

        # upper bound on sentence length, +10 because I'm too lazy to think about off-by-1 errors.
        max_sentence_length = np.max(np.diff(np.nonzero(np.diff(blocks) != 0))) + 10

        # Construct the edges and pairwise potentials for a chain CRF.
        # We construct the largest CRF we'll need here, and use sections of the chain for shorter sentences.
        edges = np.arange(max_sentence_length)
        edges = np.c_[edges[:-1], edges[1:]]
        pairwise = np.repeat(self.Q[np.newaxis,:,:], max_sentence_length, axis=0)

        decoded_labels = []
        for X_prob_sent in iterator.blocks(X_prob, blocks):
            sentence_length = X_prob_sent.shape[0]
            decoded = daicrf.mrf(
                X_prob_sent.astype(np.float),
                edges[:sentence_length-1],
                pairwise[:sentence_length-1,:,],
                verbose=0,
                alg='jt')
            decoded_labels.append(decoded)

        X_decoded = np.concatenate(decoded_labels)

        return X_decoded
