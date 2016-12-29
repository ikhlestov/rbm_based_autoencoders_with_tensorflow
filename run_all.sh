#!/usr/bin/env bash
rm -rf /tmp/rbm_*
printf '\n===== Train RBM =====\n'
python run_rbm.py --pair_training --test_trained
printf '\n===== Train AutoEncoder initialized from RBM =====\n'
python run_encoder.py --rbm_run_no=0 --test_trained
printf '\n===== Train AutoEncoder initialized with new variables =====\n'
python run_encoder.py --test_trained
printf '\n===== Evaluate embeddings though SVM =====\n'
printf '\n===== Evaluate embeddings though most similar accuracy\n'
