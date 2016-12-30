"""Whith some distance metric(hamming or dot product) retrieve 10 most similiar
embeddings to provided one and measure how many labels are correct.
Preform this for all dataset and calculate mean accuracy.
Example usage:
python results_validation/found_similiar.py --test_cases rbm:0 aec_rbm:0 aec_rbm:1
"""
import time
import argparse
import os
import csv


from tqdm import tqdm
import numpy as np
from tensorflow.examples.tutorials import mnist

from utils import get_notes_from_case, binarize_encodings


parser = argparse.ArgumentParser()
parser.add_argument(
    '--test_cases', type=str, required=True, nargs='+',
    help="Indexes of runs that should be tested."
         "Should be provided as net_type:idx"
         "ex: --test_cases rbm:0 rbm:1 rbm_aec:0 rbm_aec:1")
parser.add_argument(
    '--csv_res_path', type=str,
    default='/tmp/found_similiar_accuracy_results.csv',
    help="Where metrics from evaluation should be saved")
args = parser.parse_args()

fetch_qtty = 10
weighted_dist = False

test_cases = [
    {'type': 'default_mnist',
     'notes': 'default mnist dataset'},
]

for case in args.test_cases:
    case_type, run_no = case.split(':')
    notes = get_notes_from_case(case_type, run_no)
    test_cases.append({'type': case_type,
                       'run_no': run_no,
                       'notes': notes})


def get_data(main_path, run_no):
    encodings_test = np.load(
        os.path.join(main_path, '%s_encodings_test_set.npy' % run_no))
    labels_test_one_hot = np.load(
        os.path.join(main_path, '%s_labels_test_set.npy' % run_no))
    return encodings_test, labels_test_one_hot


def found_similiar_indexes_dot_product(test_array, test_set, qtty=10):
    dot_prod = (np.dot(test_set, test_array) /
                np.linalg.norm(test_set, axis=1) /
                np.linalg.norm(test_array))
    sorted_indexes = np.argsort(- dot_prod)
    most_sim_indexes = sorted_indexes[1: qtty + 1]
    most_dim_distances = dot_prod[most_sim_indexes]
    return most_sim_indexes, most_dim_distances


def found_most_similiar_indexes_hamming(test_array, test_set, qtty=10):
    hamming_dist = np.sum(np.bitwise_xor(test_set, test_array), axis=1)
    sorted_indexes = np.argsort(hamming_dist)
    most_sim_indexes = sorted_indexes[1: qtty + 1]
    most_dim_distances = hamming_dist[most_sim_indexes]
    return most_sim_indexes, most_dim_distances


def test_similarity_metric(metric_name, test_labels, test_set, fetch_qtty,
                           notes):
    distances_metrics = {
        'dot_product': found_similiar_indexes_dot_product,
        'hamming_dist': found_most_similiar_indexes_hamming
    }
    dist_func = distances_metrics[metric_name]
    correct_total = 0
    fetched_total = 0
    start_time = time.time()
    desc = metric_name + '/' + notes
    for idx in tqdm(range(test_labels.shape[0]), desc=desc):
        test_label = test_labels[idx]
        test_array = test_set[idx]
        most_sim_indexes, most_dim_distances = dist_func(
            test_array, test_set, qtty=fetch_qtty)
        most_sim_labels = test_labels[most_sim_indexes]
        equal_to_label = most_sim_labels == test_label
        correct_qtty = np.sum(equal_to_label)
        correct_total += correct_qtty
        fetched_total += fetch_qtty
    accuracy = correct_total / fetched_total
    time_cons = time.time() - start_time
    return accuracy, time_cons


all_results = []
for test_case in test_cases:
    test_type = test_case['type']

    if test_type == 'default_mnist':
        mnist_data = mnist.input_data.read_data_sets(
            "/tmp/MNIST_data/", one_hot=True)
        encodings_test = mnist_data.test.images
        labels_test_one_hot = mnist_data.test.labels

    if test_type == 'rbm':
        main_path = '/tmp/rbm_reconstr'
        encodings_test, labels_test_one_hot = get_data(
            main_path, test_case['run_no'])

    if test_type == 'aec_rbm':
        main_path = '/tmp/rbm_aec_reconstr'
        encodings_test, labels_test_one_hot = get_data(
            main_path, test_case['run_no'])

    # binarize encodings with threshold, even initial images from MNIST
    encodings_test_bin = np.copy(encodings_test)
    encodings_test_bin = binarize_encodings(encodings_test_bin)
    test_labels = np.argmax(labels_test_one_hot, axis=1)

    # test hamming distance
    notes = test_case['notes']
    hamming_accuracy, hamming_time_cons = test_similarity_metric(
        'hamming_dist', test_labels, encodings_test_bin, fetch_qtty, notes)
    dot_product_accuracy, dot_product_time_cons = test_similarity_metric(
        'dot_product', test_labels, encodings_test_bin, fetch_qtty, notes)

    print(notes)
    print(
        "\tHamming: accuracy - {ha}, time consumption {hts} seconds\n"
        "\tDot product: accuracy - {da}, time consumption {dts} seconds\n".format(
            ha=hamming_accuracy, hts=hamming_time_cons,
            da=dot_product_accuracy, dts=dot_product_accuracy)
    )
    res_dict = {
        'notes': test_case['notes'],
        'hamming_accuracy': hamming_accuracy,
        'hamming_time_cons': hamming_time_cons,
        'dot_product_accuracy': dot_product_accuracy,
        'dot_product_time_cons': dot_product_time_cons
    }
    all_results.append(res_dict)

# save results as csv file
with open(args.csv_res_path, 'w') as f:
    fieldnames = [
        'notes', 'hamming_accuracy', 'hamming_time_cons',
        'dot_product_accuracy', 'dot_product_time_cons']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    for row in all_results:
        writer.writerow(row)

    print("Results were saved to %s" % args.csv_res_path)
