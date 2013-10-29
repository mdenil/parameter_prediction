import numpy as np
import sys
import theano
import theano.tensor as T
from pylearn2.utils import serial
from parameter_prediction.datasets.timit import TIMIT
from parameter_prediction.language.model import BigramModel
import daicrf
import argparse


def load_name_index(file_name):
    """
    Returns names: a list where names[i] is the name of phone[i]
    """
    names = []
    with open(file_name) as timit_file:
        for line in timit_file:
            idx, label = line.split(",")
            idx = int(idx) - 1 # -1 because 1-based indexing
            name, junk = label.strip().split(" ")
            
            if name not in names:
                names.append(name)

    return names

def make_folding_index(names):
    """
    Creates a folding index based on the phone folding rules in KFL:

       http://repository.cmu.edu/cgi/viewcontent.cgi?article=2768&context=compsci

    Returns a list fold_index where fold_index[i] is the index of the folded
    version of phone[i] in names.

    i.e: name[i] folds to name[fold_index[i]]
    """
    # fold_table maps name -> folded name
    # Names not in the fold_table aren't changed by folding.
    fold_table = {
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

    # Don't be clever by folding a->b and b->c and expecting a->c.  That
    # will work or fail depending on what order the rules are processed in.
    # This assert will yell at you if you try.  Instead fold a->c and b->c
    # directly.
    assert len(set(fold_table.keys()) & set(fold_table.values())) == 0

    # build the fold_index
    fold_index = range(len(names))
    for i,name in enumerate(names):
        if name in fold_table:
            fold_index[i] = names.index(fold_table[name])
    
    return fold_index

def make_folding_matrix(fold_index):
    """
    Creates a matrix F which allows you to fold label vectors with a dot product.

    If Y is a label matrix with labels in rows then np.dot(Y, F) is the folded
    label vector.  Folding is done by adding elements of the label vectors so
    this will work for 1-hot encodings and probabilities.

    The folding matrix expects the label vectors to have 3 adjacent entries per
    phone.  The result will have them all folded together.
    """
    F = np.zeros((len(fold_index), len(fold_index)))

    for i,fi in enumerate(fold_index):
        F[i,fi] = 1

    F = F[:,np.logical_not(F.sum(axis=0) == 0)]

    assert F.shape ==  ( len(folded_indexes), len(set(folded_indexes)) )

    # re introduces the groups of 3
    F = np.kron(F, np.array([[1],[1],[1]]))

    return F


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes predictions of a model on TIMIT using phone folding.")
    parser.add_argument("model_file_name")
    args = parser.parse_args()

    names = load_name_index("data/timit_phone_index_table.txt")
    folded_indexes = make_folding_index(names)
    F = make_folding_matrix(folded_indexes)

    mlp = serial.load(args.model_file_name)
    X = T.matrix('X')
    fprop = theano.function(
            inputs=[X],
            outputs=mlp.fprop(X))

    print "Predicting test set"

    timit_test = TIMIT('test')
    Y = fprop(timit_test.X)
    Y_hat = np.equal.outer(
            np.argmax(Y, axis=1),
            np.arange(Y.shape[1]),
            ).astype(np.float)

    Y_folded = np.equal.outer(
            np.argmax(np.dot(Y, F), axis=1),
            np.arange(F.shape[1]))
    test_Y_folded = np.equal.outer(
            np.argmax(np.dot(timit_test.y, F), axis=1),
            np.arange(F.shape[1]))

    Y_raw_acc = np.all(Y_hat == timit_test.y, axis=1).mean()
    Y_folded_acc = np.all(Y_folded == test_Y_folded, axis=1).mean()

    print "Frame classification error       :", 1-Y_raw_acc
    print "Folded frame classification error:", 1-Y_folded_acc

    print "Building language model"
    timit_train = TIMIT('test')
    bigram = BigramModel()
    bigram.fit(np.argmax(timit_train.y, axis=1), timit_train.sentence_ids)

    print "Decoding predictions ... (this will take a while)"
    max_sentence_length = np.max(np.diff(np.nonzero(np.diff(timit_test.sentence_ids) != 0))) + 10

    edges = np.arange(max_sentence_length)
    edges = np.c_[edges[:-1], edges[1:]]
    pairwise = np.repeat(bigram.Q[np.newaxis,:,:], max_sentence_length, axis=0) + 1e-6
    
    def _block_iterator(X, blocks):
        for i in xrange(blocks.max()):
            yield X[blocks == i]

    decoded_labels = []
    for Y_sent in _block_iterator(Y, timit_test.sentence_ids):
        print Y_sent.sum()
        #print ".",
        sentence_length = Y_sent.shape[0]
        decoded = daicrf.mrf(
                Y_sent.astype(np.float),
                edges[:sentence_length-1],
                pairwise[:sentence_length-1,:,], 
                verbose=0,
                alg='jt')
        decoded_labels.append(decoded)
        if len(decoded_labels) > 5:
            break
    print ""

    Y_decoded = np.equal.outer(
            np.concatenate(decoded_labels),
            np.arange(timit_test.y.shape[1]))
    Y_decoded_folded = np.equal.outer(
            np.argmax(np.dot(Y_decoded, F), axis=1),
            np.arange(F.shape[1]))

    print Y_decoded.shape, Y_decoded_folded.shape
    print timit_test.y.shape, test_Y_folded.shape

    timit_test.y = timit_test.y[:Y_decoded.shape[0]]
    print (Y_decoded == timit_test.y).all(axis=1)
            
    Y_decoded_raw_acc = (Y_decoded == timit_test.y).all(axis=1).mean()
    Y_decoded_folded_acc = (Y_decoded_folded == test_Y_folded).all(axis=1).mean()

    print "Frame decoding error       :", 1-Y_decoded_raw_acc
    print "Folded frame decoding error:", 1-Y_decoded_folded_acc




    
