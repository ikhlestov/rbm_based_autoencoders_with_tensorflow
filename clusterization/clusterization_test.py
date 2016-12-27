import os
from time import time
import pickle
import warnings

import numpy as np
from sklearn import metrics, svm, preprocessing
from sklearn.cluster import KMeans, SpectralClustering
from tensorflow.examples.tutorials import mnist

MAX_ITER = 200
TEST_SCALED = False

test_cases = [
    {'type': 'default_mnist',
     'notes': 'SVM based on default mnist dataset'},
    {'type': 'rbm',
     'notes': 'RMB based encodings, training with sigmoids',
     'run_no': '0', },
    {'type': 'rbm',
     'notes': 'RBM based encodings, training with binary mode',
     'run_no': '1', },
    {'type': 'aec_rbm',
     'notes': 'Autoencoder based on sigmoid RBM',
     'run_no': '0', },
    {'type': 'aec_rbm',
     'notes': 'Autoencoder based on binary mode RBM',
     'run_no': '1', },
]


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
    return accuracy, prec_recall_f_score


def append_results(all_results, accuracy, prec_recall_f_score, notes):
    all_results.append({
        'accuracy': accuracy,
        'prec_recall_f_score': prec_recall_f_score,
        'notes': notes
    })
    return all_results


def binarize_encodings(encodings, threshold=0.2):
    encodings[encodings > threshold] = 1
    encodings[encodings <= threshold] = 0
    return encodings


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
        encodings_train = mnist_data.train.images
        labels_train_one_hot = mnist_data.train.labels
        encodings_test = mnist_data.test.images
        labels_test_one_hot = mnist_data.test.labels

    if test_type == 'rbm':
        main_path = '/tmp/rbm_reconstr'
        (encodings_train, labels_train_one_hot,
         encodings_test, labels_test_one_hot) = get_data(
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

        # test without data scaling
        all_results = run_estimator(
            all_results,
            notes=test_case['notes'],
            encodings_train=encodings_train,
            labels_train=labels_train,
            encodings_test=encodings_test,
            labels_test=labels_test)

        if TEST_SCALED:
            encodings_train = preprocessing.scale(encodings_train)
            encodings_test = preprocessing.scale(encodings_test)

            # test with data scaling
            new_notes = test_case['notes'] + ', scaled_data'
            all_results = run_estimator(
                all_results,
                notes=new_notes,
                encodings_train=encodings_train,
                labels_train=labels_train,
                encodings_test=encodings_test,
                labels_test=labels_test)

        # test also with binary mode for AutoEncoder based on rbm
        if test_type == 'aec_rbm':
            new_notes = test_case['notes'] + ', binarized encodings'
            all_results = run_estimator(
                all_results,
                notes=new_notes,
                encodings_train=encodings_train_bin,
                labels_train=labels_train,
                encodings_test=encodings_test_bin,
                labels_test=labels_test)

            # and also with scaled data
            if TEST_SCALED:
                encodings_train_bin = preprocessing.scale(encodings_train_bin)
                encodings_test_bin = preprocessing.scale(encodings_test_bin)

                new_notes += ', scaled_data'
                all_results = run_estimator(
                    all_results,
                    notes=new_notes,
                    encodings_train=encodings_train_bin,
                    labels_train=labels_train,
                    encodings_test=encodings_test_bin,
                    labels_test=labels_test)

with open('clusterization_results_%d_iters.pkl' % MAX_ITER, 'wb') as f:
    pickle.dump(all_results, f)
