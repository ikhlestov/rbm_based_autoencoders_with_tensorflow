"""Run RMB in various modes
Example usages:
python run_rbm.py --pair_training --test_trained
python run_rbm.py --binary
"""
import argparse

from rbm_all_layers_at_once import RBMAllAtOnce
from rbm_train_by_pair_layers import RBMTrainByPairs
from data_providers import MNISTDataProvider


parser = argparse.ArgumentParser()
# training params
parser.add_argument(
    '--pair_training', action='store_true',
    help="Should all RBM layers be trained at once or only by pairs")
parser.add_argument(
    '--binary', action='store_true',
    help="Should all layers be binary encoded(True) or only last one(False)")
parser.add_argument(
    '--test_trained', action='store_true',
    help="Should trained model be fetched for embeddings in place")
# testing params
parser.add_argument(
    '--test', action='store_true',
    help="Get embeddings for required model")
parser.add_argument(
    '--run_no', type=str,
    help="What model should be tested")
parser.add_argument(
    '--plot_images', action='store_true',
    help='Plot some weights/reconstruction at the evaluation')
args = parser.parse_args()

params = {
    'epochs': 6,
    'learning_rate': 0.01,
    'batch_size': 100,
    'validate': True,
    'shuffle': True,
    'gibbs_sampling_steps': 1,
    'layers_qtty': 3,
    # [n_input_features, layer_1, ...]
    'layers_sizes': [784, 484, 196, 100],
    'bin_type': args.binary,
}

if not args.pair_training:
    print("Train model all layers at once")
    notes = 'train_all_layers_at_once__'
    ModelClass = RBMAllAtOnce
if args.pair_training:
    print("Train model by pair layers")
    notes = 'train_layers_by_pairs__'
    ModelClass = RBMTrainByPairs

if not args.binary:
    print("Only last layer is binarized")
    notes += 'last_layer_binarized'
if args.binary:
    print("All layers are binarized")
    notes += 'all_layers_binarized'

params['notes'] = notes
initial_params = dict(params)

mnist_provider = MNISTDataProvider()
test_run_no = None

if not args.test:
    rbm_model = ModelClass(
        data_provider=mnist_provider,
        params=params)
    rbm_model.train()
    if args.test_trained:
        test_run_no = rbm_model.run_no

if args.test:
    if not args.run_no:
        print("\nYou should provide run_no of model to test!\n")
        exit()
    else:
        test_run_no = args.run_no

if test_run_no is not None:
    rbm_model = ModelClass(
        data_provider=mnist_provider,
        params=params)
    rbm_model.test(
        run_no=test_run_no,
        plot_images=args.plot_images,
    )
