import numpy as np
import argparse
import nltk
import sys
import theano
import theano.tensor as T
from pylearn2.utils import serial
from parameter_prediction.datasets.timit import TIMIT
from parameter_prediction.language.model import BigramModel
import daicrf

def load_phone_index(file_name):
    """
    Returns phones: a list where phones[i] is the name of phone[i]
    """
    phones = []
    with open(file_name) as timit_file:
        for line in timit_file:
            idx, label = line.split(",")
            idx = int(idx) - 1 # -1 because 1-based indexing
            phone, junk = label.strip().split(" ")
            
            if phone not in phones:
                phones.append(phone)

    return phones

def get_model_predictions(model, dataset):
    X = T.matrix('X')
    fprop = theano.function(
            inputs=[X],
            outputs=model.fprop(X))

    Y_prob = fprop(dataset.X)
    Y_hat = np.equal.outer(
            np.argmax(Y_prob, axis=1),
            np.arange(Y_prob.shape[1]),
            ).astype(np.float)

    return Y_hat, Y_prob

def get_phone_sequences(Y_hat, sentence_ids):
    def _block_iterator(X, blocks):
        for i in xrange(blocks.max()+1):
            yield X[blocks == i]

    phone_index = load_phone_index("data/timit_phone_index_table.txt")

    phone_sequences = []
    Y_idx = Y_hat.argmax(axis=1)
    for Y_idx_sent in _block_iterator(Y_idx, sentence_ids):
        phone_sent = [phone_index[i] for i in Y_idx_sent]

        trimmed_phone_sent = phone_sent
        #trimmed_phone_sent = []
        #for i,phone in enumerate(phone_sent):
        #    if i == 0 or phone != phone_sent[i-1]:
        #        trimmed_phone_sent.append(phone)

        phone_sequences.append(trimmed_phone_sent)

    return phone_sequences

def phone_error_rate(predicted, truth):
    err = 0.0
    total = 0.0
    for p,t in zip(predicted, truth):
        err += nltk.metrics.distance.edit_distance(p, t)
        total += len(t)
    return err/total

def fold_phone_sequences(sequences):
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
    folded_sequences = []
    for seq in sequences:
        folded_seq = [fold_table.get(p,p) for p in seq]
        folded_sequences.append(folded_seq)
    return folded_sequences

def phone_stress_combining_matrix():
    return np.kron(np.eye(61), np.array([[1],[1],[1]]))

def remove_silence(sequences):
    no_silence_begin = []
    for seq in sequences:
        for i,p in enumerate(seq):
            if p != 'h#':
                no_silence_begin.append(seq[i:])
                break
    no_silence_end = []
    for seq in no_silence_begin:
        for i,p in enumerate(reversed(seq)):
            if p != 'h#':
                no_silence_end.append(seq[:-i])
                break
    return no_silence_end

if __name__ == "__main__":

#
#    phone_index = load_phone_index("data/timit_phone_index_table.txt")
#    Q = np.zeros([len(phone_index)]*2)
#
#    
#
#    # http://nltk.org/api/nltk.corpus.reader.html#id1
#    timit_train = nltk.corpus.reader.TimitCorpusReader(
#            root=nltk.data.FileSystemPathPointer('data/timit-nltk/train'))
#    bigrams = nltk.collocations.BigramCollocationFinder.from_words(timit_train.phones())
#
#    viewitems = bigrams.ngram_fd.viewitems()
#    print type(bigrams.ngram_fd)
#    for x in bigrams.ngram_fd.iteritems():
#        print x
#        break
#    exit(0)


    parser = argparse.ArgumentParser(description="Computes predictions of a model on TIMIT using phone folding.")
    parser.add_argument("model_file_name")
    args = parser.parse_args()

    timit_test = TIMIT('test')

    mlp = serial.load(args.model_file_name)
    Y_hat, Y_prob = get_model_predictions(mlp, timit_test)

    # combine phone stresses
    Y_hat = np.dot(Y_hat, phone_stress_combining_matrix())
    Y_prob = np.dot(Y_prob, phone_stress_combining_matrix())
    timit_test_labels_stress_combined = np.dot(timit_test.y, phone_stress_combining_matrix())

    frame_classification_accuracy = np.all(Y_hat == timit_test_labels_stress_combined, axis=1).mean()
    print "Frame classification error (raw): ", 1-frame_classification_accuracy

    true_phone_sequences = get_phone_sequences(timit_test_labels_stress_combined, timit_test.sentence_ids)
    predicted_phone_sequences = get_phone_sequences(Y_prob, timit_test.sentence_ids)

    print "Phone error rate (raw):", phone_error_rate(predicted_phone_sequences, true_phone_sequences)

    true_phone_sequences_folded = fold_phone_sequences(true_phone_sequences)
    predicted_phone_sequences_folded = fold_phone_sequences(predicted_phone_sequences)

    print "Phone error rate (folded):", phone_error_rate(predicted_phone_sequences_folded, true_phone_sequences_folded)

    true_phone_sequences_folded_nosilence = remove_silence(true_phone_sequences_folded)
    predicted_phone_sequences_folded_nosilence = remove_silence(predicted_phone_sequences_folded)

    print "Phone error rate (folded, no silence):", phone_error_rate(predicted_phone_sequences_folded_nosilence, true_phone_sequences_folded_nosilence)








    timit_train = TIMIT('train')
    timit_train_labels_stress_combined = np.dot(timit_train.y, phone_stress_combining_matrix())
    bigram = BigramModel()
    bigram.fit(np.argmax(timit_train_labels_stress_combined, axis=1), timit_train.sentence_ids)

    print "Decoding predictions ... (this will take a while)"
    max_sentence_length = np.max(np.diff(np.nonzero(np.diff(timit_test.sentence_ids) != 0))) + 10

    edges = np.arange(max_sentence_length)
    edges = np.c_[edges[:-1], edges[1:]]
    pairwise = np.repeat(bigram.Q[np.newaxis,:,:], max_sentence_length, axis=0)
    
    def _block_iterator(X, blocks):
        for i in xrange(blocks.max()+1):
            yield X[blocks == i]

    decoded_labels = []
    for Y_prob_sent in _block_iterator(Y_prob, timit_test.sentence_ids):
        sentence_length = Y_prob_sent.shape[0]
        decoded = daicrf.mrf(
                Y_prob_sent.astype(np.float),
                edges[:sentence_length-1],
                pairwise[:sentence_length-1,:,], 
                verbose=0,
                alg='jt')
        decoded_labels.append(decoded)
        #if len(decoded_labels) > 5:
        #    break
    #print ""

    Y_decoded = np.equal.outer(
            np.concatenate(decoded_labels),
            np.arange(timit_test_labels_stress_combined.shape[1]))

    Y_decoded_raw_acc = (Y_decoded == timit_test_labels_stress_combined).all(axis=1).mean()

    print "Phone classification error (decoded):", 1-Y_decoded_raw_acc

    decoded_phone_sequences = get_phone_sequences(Y_decoded, timit_test.sentence_ids)

    print "Phone error rate (decoded):", phone_error_rate(decoded_phone_sequences, true_phone_sequences)

    decoded_phone_sequences_folded = fold_phone_sequences(decoded_phone_sequences)

    print "Phone error rate (decoded, folded):", phone_error_rate(decoded_phone_sequences_folded, true_phone_sequences_folded)

