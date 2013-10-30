import numpy as np
import argparse
import nltk
import theano
import theano.tensor as T
from pylearn2.utils import serial
from parameter_prediction.datasets.timit import TIMIT
from parameter_prediction.language.model import BigramModel
from parameter_prediction.util import iterator

def get_model_predictions(model, timit):
    X_symbolic = T.matrix('X')
    fprop = theano.function(
            inputs=[X_symbolic],
            outputs=model.fprop(X_symbolic))

    Y_prob = fprop(timit.X)
    Y_hat = np.equal.outer(
            np.argmax(Y_prob, axis=1),
            np.arange(Y_prob.shape[1]),
            ).astype(np.float)

    if timit.stress_combined:
        Y_hat = np.dot(Y_hat, timit.phone_stress_combining_matrix)
        Y_prob = np.dot(Y_prob, timit.phone_stress_combining_matrix)

    return Y_hat, Y_prob

def get_phone_sequences(Y_hat, sentence_ids, phone_index):
    phone_sequences = []
    Y_idx = Y_hat.argmax(axis=1)
    for Y_idx_sent in iterator.blocks(Y_idx, sentence_ids):
        phone_sent = [phone_index[i] for i in Y_idx_sent]
        phone_sequences.append(phone_sent)

    return phone_sequences

def phone_error_rate(predicted, truth):
    err = 0.0
    total = 0.0
    for p,t in zip(predicted, truth):
        err += nltk.metrics.distance.edit_distance(p, t)
        total += len(t)
    return err/total

def fold_phone_sequences(fold_table, sequences):
    folded_sequences = []
    for seq in sequences:
        folded_seq = [fold_table.get(p,p) for p in seq]
        folded_sequences.append(folded_seq)
    return folded_sequences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
            "Computes predictions of a model on TIMIT using phone folding.")
    parser.add_argument("model_file_name")
    args = parser.parse_args()

    timit_test = TIMIT('test', combine_stress=True)

    mlp = serial.load(args.model_file_name)
    Y_hat, Y_prob = get_model_predictions(mlp, timit_test)

    frame_classification_accuracy = np.all(Y_hat == timit_test.y, axis=1).mean()
    print "Frame classification error (raw): ", 1-frame_classification_accuracy

    true_phone_sequences = get_phone_sequences(timit_test.y, timit_test.sentence_ids, timit_test.phone_index)
    predicted_phone_sequences = get_phone_sequences(Y_prob, timit_test.sentence_ids, timit_test.phone_index)
    print "Phone error rate (raw):", phone_error_rate(predicted_phone_sequences, true_phone_sequences)

    true_phone_sequences_folded = fold_phone_sequences(timit_test.fold_table, true_phone_sequences)
    predicted_phone_sequences_folded = fold_phone_sequences(timit_test.fold_table, predicted_phone_sequences)
    print "Phone error rate (folded):", phone_error_rate(predicted_phone_sequences_folded, true_phone_sequences_folded)


    print "Training language model..."
    timit_train = TIMIT('train', combine_stress=True)
    bigram = BigramModel()
    bigram.fit(np.argmax(timit_train.y, axis=1), timit_train.sentence_ids)
    del timit_train

    print "Decoding predictions..."
    Y_decoded = np.equal.outer(
            bigram.predict(Y_prob, timit_test.sentence_ids),
            np.arange(timit_test.y.shape[1]))

    Y_decoded_raw_acc = np.all(Y_decoded == timit_test.y, axis=1).mean()
    print "Phone classification error (decoded):", 1-Y_decoded_raw_acc

    decoded_phone_sequences = get_phone_sequences(Y_decoded, timit_test.sentence_ids, timit_test.phone_index)
    print "Phone error rate (decoded):", phone_error_rate(decoded_phone_sequences, true_phone_sequences)

    decoded_phone_sequences_folded = fold_phone_sequences(timit_test.fold_table, decoded_phone_sequences)
    print "Phone error rate (decoded, folded):", phone_error_rate(decoded_phone_sequences_folded, true_phone_sequences_folded)

