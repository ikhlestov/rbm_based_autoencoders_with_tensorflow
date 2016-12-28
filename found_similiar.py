import time
from datetime import timedelta

from tqdm import tqdm
import numpy as np
from PIL import Image
from scipy.misc import toimage
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials import mnist


def binarize_encodings(encodings, threshold=0.2):
    encodings[encodings > threshold] = 1
    encodings[encodings <= threshold] = 0
    encodings = encodings.astype('i4')
    return encodings


MNIST = True
if not MNIST:
    print("Not MNIST data")
    test_set = np.load('/tmp/rbm_aec_reconstr/0_encodings_test_set.npy')
    test_set = binarize_encodings(test_set)
    test_labels = np.load('/tmp/rbm_aec_reconstr/0_labels_test_set.npy')
    test_labels = np.argmax(test_labels, axis=1)

if MNIST:
    print("MNIST data")
    mnist_data = mnist.input_data.read_data_sets(
                "/tmp/MNIST_data/", one_hot=False)
    test_set = mnist_data.test.images
    test_set = binarize_encodings(test_set)
    test_labels = mnist_data.test.labels


def found_similiar_indexes_dot_product(test_array, qtty=10):
    dot_prod = (np.dot(test_set, test_array) /
                np.linalg.norm(test_set, axis=1) /
                np.linalg.norm(test_array))
    sorted_indexes = np.argsort(- dot_prod)
    most_sim_indexes = sorted_indexes[1 : qtty + 1]
    most_dim_distances = dot_prod[most_sim_indexes]
    return most_sim_indexes, most_dim_distances


def found_most_similiar_indexes_hamming(test_array, qtty=10):
    hamming_dist = np.sum(np.bitwise_xor(test_set, test_array), axis=1)
    sorted_indexes = np.argsort(hamming_dist)
    most_sim_indexes = sorted_indexes[1 : qtty + 1]
    most_dim_distances = hamming_dist[most_sim_indexes]
    return most_sim_indexes, most_dim_distances


# Some params
dist_metric_type = 'hamming_dist'
fetch_qtty = 10
weighted_dist = False

distances_metrics = {
    'dot_product': found_similiar_indexes_dot_product,
    'hamming_dist': found_most_similiar_indexes_hamming
}

correct_total = 0
fetched_total = 0
start_time = time.time()
mask_array = np.array(list(reversed(range(1, fetch_qtty + 1))))
mask_aray_sum = sum(mask_array)
if weighted_dist:
    fetch_qtty_to_add = mask_aray_sum
else:
    fetch_qtty_to_add = fetch_qtty

for idx in tqdm(range(test_labels.shape[0])):
    test_label = test_labels[idx]
    test_array = test_set[idx]
    most_sim_indexes, most_dim_distances = distances_metrics[dist_metric_type](
        test_array, qtty=fetch_qtty)
    most_sim_labels = test_labels[most_sim_indexes]
    equal_to_label = most_sim_labels == test_label
    if weighted_dist:
        equal_to_label = equal_to_label * mask_array
    correct_qtty = np.sum(equal_to_label)
    correct_total += correct_qtty
    fetched_total += fetch_qtty_to_add

print("Time consumption: %s" % timedelta(seconds=time.time() - start_time))
print("Fetched_qtty: %d, weighted_dist: %s, distance metric type: %s."
      "\nAccuracy %.4f." % (
    fetch_qtty, weighted_dist, dist_metric_type,
    correct_total / fetched_total))


# def print_most_similiar_labels(idx, qtty=10):
#     test_array = test_set[idx]
#     print("Test on label: %d" % test_labels[idx])
#     most_sim_indexes, most_dim_distances = found_most_similiar_indexes(
#         test_array, qtty=qtty)
#     codes = test_set[most_sim_indexes]
#     # toimage(codes).show()
#     # img = mpimg.imread('/home/legatsap/Pictures/99737.jpg')
#     # plt.imshow(codes)
#     # plt.show()
#     print("Most similiar labels are:")
#     for sim_label, sim_dist, sim_idx in zip(test_labels[most_sim_indexes], most_dim_distances, most_sim_indexes):
#         print("\t%d\tdist: %d, \tidx: %d" % (sim_label, sim_dist, sim_idx))


# while True:
#     idx = int(input("Enter index to test\n>>> "))
#     print_most_similiar_labels(idx)
