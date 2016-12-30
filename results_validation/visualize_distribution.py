"""Visualize distribution of one or two models probability embeddings
Example usage:
python results_validation/visualize_distribution.py --emb_path /tmp/rbm_aec_reconstr/0_encodings_test_set.npy /tmp/rbm_aec_reconstr/1_encodings_test_set.npy
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    '--emb_path', type=str, required=True, nargs='+',
    help="Path to numpy files of probability emmbeddings")
args = parser.parse_args()


def get_notes_from_filepath(path):
    idx = os.path.basename(path).split('_')[0]
    if 'aec' in path:
        logs_dir = '/tmp/rbm_aec_logs'
        net_type = 'autoencoder: '
    else:
        logs_dir = '/tmp/rbm_logs'
        net_type = 'rbm: '
    all_logs = os.listdir(logs_dir)
    log_name = [l for l in all_logs if l.startswith(idx + '_')][0]
    notes = log_name.strip(idx + '_')
    return net_type + notes

handled_pathes = args.emb_path

notes = [get_notes_from_filepath(path) for path in handled_pathes]
data = [np.load(path) for path in handled_pathes]

# plot data
nrows = len(data)
fig, axes = plt.subplots(
    nrows=nrows, ncols=1, figsize=(16*2, 4*nrows), sharey=True)
for i in range(len(data)):
    if nrows == 1:
        ax = axes
    else:
        ax = axes[i]
    ax.hist(data[i], bins=100)
    ax.grid(True)
    ax.set_title(notes[i])

fig.savefig('/tmp/rbm_aec_embeddings_distribution.png')
plt.show()
