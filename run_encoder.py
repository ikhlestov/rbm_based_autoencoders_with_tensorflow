import argparse

from autoencoder_1 import Encoder
from data_providers import MNISTDataProvider


parser = argparse.ArgumentParser()
# args for training
parser.add_argument(
    '--rbm_run_no', type=str,
    help="Get from what RMB run_no autoencoder model should be preloaded")
# args for testing
parser.add_argument(
    '--test', action='store_true')
parser.add_argument(
    '--run_no', type=str,
    help='What training model should be tested')
parser.add_argument(
    '--train_set', action='store_true',
    help='Should we use train set for evaluation')
parser.add_argument(
    '--plot_images', action='store_true',
    help='Plot some weights/reconstruction at the evaluation')
args = parser.parse_args()

params = {
    'epochs': 6,
    'learning_rate': 0.5,
    'batch_size': 100,
    'validate': True,
    'shuffle': True,
    'layers_qtty': 3,
    # [n_input_features, layer_1, ...]
    'layers_sizes': [784, 484, 196, 100],
}

if args.rbm_run_no:
    params['notes'] = 'rbm_initialized_model'
if not args.rbm_run_no:
    params['notes'] = 'new_initialized_model'

mnist_provider = MNISTDataProvider(bin_code_width=params['layers_sizes'][-1])
model = Encoder(
    data_provider=mnist_provider,
    params=params,
    rbm_run_no=args.rbm_run_no)

if not args.test:
    print("Training the model")
    model.train()

if args.test:
    print("Testing the model")
    model.test(
        run_no=args.run_no,
        train_set=args.train_set,
        plot_images=args.plot_images
    )
