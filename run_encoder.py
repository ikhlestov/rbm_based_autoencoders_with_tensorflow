"""Run autoencoder with weight initialized from RBM or newly created.
Example usages:
# train newly created autoencoder
`python run_encoder.py`
# train autoencoder initialized from RBM
`python run_encoder.py --rbm_run_no=0`
Also take a look at the params for noise adding and testing in place after
training.
"""
import argparse

from autoencoder import Encoder
from data_providers import MNISTDataProvider


parser = argparse.ArgumentParser()
# args for training
parser.add_argument(
    '--rbm_run_no', type=str,
    help="Get from what RMB run_no autoencoder model should be preloaded")
parser.add_argument(
    '--without_noise', action='store_true',
    help='Train autoencoder without Gaussian noise prior embeddings level')
parser.add_argument(
    '--test_trained', action='store_true',
    help="Should trained model be fetched for embeddings in place")
# args for testing
parser.add_argument(
    '--test', action='store_true',
    help="Get embeddings for required model")
parser.add_argument(
    '--run_no', type=str,
    help='What training model should be tested')
parser.add_argument(
    '--plot_images', action='store_true',
    help='Plot some weights/reconstruction at the evaluation')
args = parser.parse_args()

params = {
    'epochs': 6,
    'learning_rate': 1.0,
    'batch_size': 100,
    'validate': True,
    'shuffle': True,
    'layers_qtty': 3,
    # [n_input_features, layer_1, ...]
    'layers_sizes': [784, 484, 196, 100],
    'without_noise': args.without_noise
}

if args.rbm_run_no:
    notes = 'rbm_initialized_model'
if not args.rbm_run_no:
    notes = 'new_initialized_model'

if not args.without_noise:
    notes += '__with_Gaussian_noise'
if args.without_noise:
    notes += '__without_Gaussian_noise'
params['notes'] = notes

mnist_provider = MNISTDataProvider(bin_code_width=params['layers_sizes'][-1])
model = Encoder(
    data_provider=mnist_provider,
    params=params,
    rbm_run_no=args.rbm_run_no)

test_run_no = None
if not args.test:
    model.train()
    if args.test_trained:
        test_run_no = model.run_no

if args.test:
    if not args.run_no:
        print("\nYou should provide run_no of model to test!\n")
        exit()
    else:
        test_run_no = args.run_no

if test_run_no is not None:
    model.test(
        run_no=test_run_no,
        plot_images=args.plot_images
    )
