import argparse

import tensorflow as tf

from autoencoder_1 import Encoder
from data_providers import MNISTDataProvider


parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument(
    '--run_no', type=str, required=True,
    help="Get from what RMB run_no autoencoder model should be preloaded")
args = parser.parse_args()

params = {
    'epochs': 6,
    'learning_rate': 0.5,
    'batch_size': 100,
    'validate': True,
    'shuffle': True,
    'gibbs_sampling_steps': 1,
    'layers_qtty': 3,
    'layers_sizes': [784, 484, 196, 100],  # [n_input_features, layer_1, ...]
    'bin_type': False,
}

mnist_provider = MNISTDataProvider(bin_code_width=params['layers_sizes'][-1])

initial_params = dict(params)

model = Encoder(
    data_provider=mnist_provider,
    params=initial_params,
    run_no=args.run_no)

if not args.test:
    print("Training the model")
    model.train()
else:
    print("Testing the model")
    model.test(run_no=args.run_no)
