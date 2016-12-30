"""Run sklearn SVM wth banch of models and check their accuracy.
Example usage:
python results_validation/svm_clusterization_test.py --test_cases rbm:0 aec_rbm:0 aec_rbm:1
"""
import os
from time import time
import warnings
import argparse
import csv

import numpy as np
from sklearn import metrics, svm, preprocessing
from tensorflow.examples.tutorials import mnist

from utils import get_notes_from_case, binarize_encodings

MAX_ITER = 50

parser = argparse.ArgumentParser()
parser.add_argument(
    '--test_cases', type=str, required=True, nargs='+',
    help="Indexes of runs that should be tested."
         "Should be provided as net_type:idx"
         "ex: --test_cases rbm:0 rbm:1 rbm_aec:0 rbm_aec:1")
parser.add_argument(
    '--csv_res_path', type=str,
    default='/tmp/svm_clusterizationn_test_results.csv',
    help="Where metrics from evaluation should be saved")
args = parser.parse_args()

test_cases = [
    {'type': 'default_mnist',
     'notes': 'default mnist dataset'},
]

for case in args.test_cases:
    case_type, run_no = case.split(':')
    notes = get_notes_from_case(case_type, run_no)
    test_cases.append(
        {'type': case_type,
         'run_no': run_no,
         'notes': notes
        }
    )


def get_data(main_path, run_no):
    encodings_train = np.load(
        os.path.join(main_path, '%s_encodings_train_set.npy' % run_no))
    labels_train_one_hot = np.load(
        os.path.join(main_path, '%s_labels_train_set.npy' % run_no))
    encodings_test = np.load(
        os.path.join(main_path, '%s_encodings_test_set.npy' % run_no))
    labels_test_one_hot = np.load(
        os.path.join(main_path, '%s_labels_test_set.npy' % run_no))
    return (encodings_train, labels_train_one_hot,
            encodings_test, labels_test_one_hot)


def test_svm_estimator(estimator, notes, encodings_train, labels_train,
                       encodings_test, labels_test):
    t0 = time()
    estimator.fit(encodings_train, labels_train)
    print("Time cons: %.2fs, type: %s" % (time() - t0, notes))
    predicted = estimator.predict(encodings_test)
    accuracy = metrics.accuracy_score(labels_test, predicted)
    print("Accuracy: %.5f" % accuracy)
    report = metrics.classification_report(labels_test, predicted)
    print(report)
    prec_recall_f_score = metrics.precision_recall_fscore_support(
        labels_test, predicted)
    print('-' * 10)
    prec_recall_f_score_dict = {
        'prec': np.mean(prec_recall_f_score[0]),
        'recall': np.mean(prec_recall_f_score[1]),
        'f_score': np.mean(prec_recall_f_score[2])
    }
    return accuracy, prec_recall_f_score_dict


def append_results(all_results, accuracy, prec_recall_f_score, notes):
    res = {
        'accuracy': accuracy,
        'notes': notes
    }
    res.update(prec_recall_f_score)
    all_results.append(res)
    return all_results


def run_estimator(all_results, notes, encodings_train, labels_train,
                  encodings_test, labels_test):
    verbose = False
    max_iter = MAX_ITER

    estimator = svm.SVC(
        kernel='rbf', verbose=verbose, max_iter=max_iter,
        decision_function_shape='ovr')
    accuracy, prec_recall_f_score = test_svm_estimator(
        estimator, notes=notes,
        encodings_train=encodings_train,
        labels_train=labels_train,
        encodings_test=encodings_test,
        labels_test=labels_test)
    all_results = append_results(
        all_results, accuracy, prec_recall_f_score, notes)
    return all_results


all_results = []
for test_case in test_cases:
    test_type = test_case['type']

    if test_type == 'default_mnist':
        mnist_data = mnist.input_data.read_data_sets(
                    "/tmp/MNIST_data/", one_hot=True)
        encodings_train_bin = mnist_data.train.images
        labels_train_one_hot = mnist_data.train.labels
        encodings_test_bin = mnist_data.test.images
        labels_test_one_hot = mnist_data.test.labels

    if test_type == 'rbm':
        main_path = '/tmp/rbm_reconstr'
        (encodings_train_bin, labels_train_one_hot,
         encodings_test_bin, labels_test_one_hot) = get_data(
            main_path, test_case['run_no'])

    if test_type == 'aec_rbm':
        main_path = '/tmp/rbm_aec_reconstr'
        (encodings_train, labels_train_one_hot,
         encodings_test, labels_test_one_hot) = get_data(
            main_path, test_case['run_no'])
        # binarize encodings with threshold
        encodings_train_bin = np.copy(encodings_train)
        encodings_train_bin = binarize_encodings(encodings_train_bin)
        encodings_test_bin = np.copy(encodings_test)
        encodings_test_bin = binarize_encodings(encodings_test_bin)

    labels_train = np.argmax(labels_train_one_hot, axis=1)
    labels_test = np.argmax(labels_test_one_hot, axis=1)


    # ignore all sklearn warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        all_results = run_estimator(
            all_results,
            notes=test_case['notes'],
            encodings_train=encodings_train_bin,
            labels_train=labels_train,
            encodings_test=encodings_test_bin,
            labels_test=labels_test)

# save results as csv file
with open(args.csv_res_path, 'w') as f:
    fieldnames = ['notes', 'accuracy', 'prec', 'f_score', 'recall']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    for row in all_results:
        writer.writerow(row)

    print("Results were saved to %s" % args.csv_res_path)
