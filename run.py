import argparse

import tensorflow as tf

from rbm_1_class_based import RBM
from rbm_3_dynamic_restored import RBMDynamic
from data_providers import MNISTDataProvider

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--dynamic', action='store_true')
parser.add_argument('--run_no', type=str)
parser.add_argument(
    '--train_set', action='store_true',
    help='Should we use train set for evaluation')
parser.add_argument(
    '--plot_images', action='store_true',
    help='Plot some weights/reconstruction at the evaluation')
args = parser.parse_args()

params = {
    'epochs': 2,
    'learning_rate': 0.1,  # 0.01 initial
    'batch_size': 100,
    'validate': True,
    'shuffle': True,
    # it seems that sampling step 1 works better
    'gibbs_sampling_steps': 1,
    'layers_qtty': 3,
    'layers_sizes': [784, 484, 196, 100],  # [n_input_features, layer_1, ...]
    # 'layers_sizes': [784, 784, 784, 100],  # [n_input_features, layer_1, ...]
    # 'layers_qtty': 2,
    # 'layers_sizes': [784, 200, 100],  # [n_input_features, layer_1, ...]
    # 'layers_qtty': 1,
    # 'layers_sizes': [784, 100],  # [n_input_features, layer_1, ...]
    'bin_type': True,
}

if args.dynamic:
    notes = 'dynamic__'
else:
    notes = 'static__'
notes += 'bin_type=%s' % params['bin_type']
params['notes'] = notes

mnist_provider = MNISTDataProvider()
if args.dynamic:
    print("Use dynamic model generator")

initial_params = dict(params)

if not args.test:
    if args.dynamic:
        for layers_qtty in range(1, params['layers_qtty'] + 1):
            tf.reset_default_graph()
            print("Train layers pair %d and %d" % (layers_qtty - 1, layers_qtty))
            params['layers_qtty'] = layers_qtty
            params['layers_sizes'] = initial_params['layers_sizes'][:layers_qtty + 1]
            rmb_model = RBMDynamic(
                data_provider=mnist_provider,
                params=params)
            params = rmb_model.train()
    else:
        params['epochs'] = params['epochs'] * params['layers_qtty']
        tf.reset_default_graph()
        rmb_model = RBM(
            data_provider=mnist_provider,
            params=params)
        rmb_model.train()

if args.test:
    if not args.run_no:
        print("\nYou should provide run_no of model to test!\n")
        exit()
    if args.dynamic:
        rmb_model = RBMDynamic(
            data_provider=mnist_provider,
            params=params)
    else:
        rmb_model = RBM(
            data_provider=mnist_provider,
            params=params)
    rmb_model.test(
        run_no=args.run_no,
        train_set=args.train_set,
        plot_images=args.plot_images,
    )
