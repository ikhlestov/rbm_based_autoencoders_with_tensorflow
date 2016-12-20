import argparse

from rbm_1_class_based import RBM
from data_providers import MNISTDataProvider

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--run_no', type=str)
args = parser.parse_args()

params = {
    'epochs': 5,
    'learning_rate': 0.01,
    'batch_size': 100,
    'validate': True,
    'shuffle': True,
    # it seems that sampling step 1 works better
    'gibbs_sampling_steps': 3,
    'layers_qtty': 3,
    'layers_sizes': [784, 500, 200, 100],  # [n_input_features, layer_1, ...]
    # 'layers_qtty': 2,
    # 'layers_sizes': [784, 200, 100],  # [n_input_features, layer_1, ...]
    # 'layers_qtty': 1,
    # 'layers_sizes': [784, 100],  # [n_input_features, layer_1, ...]

}

mnist_provider = MNISTDataProvider()
rmb_model = RBM(
    data_provider=mnist_provider,
    params=params)
if not args.test:
    rmb_model.train()
else:
    if not args.run_no:
        print("\nYou should provide run_no of model to test!\n")
        exit()
    rmb_model.test(run_no=args.run_no)
