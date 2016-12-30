#!/usr/bin/env bash
rm -rf /tmp/rbm_*
# training part
printf '\n===== Train RBM =====\n'
python run_rbm.py --pair_training --test_trained
printf '\n===== Train AutoEncoder initialized from RBM =====\n'
python run_encoder.py --rbm_run_no=0 --test_trained
printf '\n===== Train AutoEncoder initialized with new variables =====\n'
python run_encoder.py --test_trained
printf '\n===== Train AutoEncoder initialized from RBM without Gaussian noise=====\n'
python run_encoder.py --rbm_run_no=0 --without_noise --test_trained
printf '\n===== Train AutoEncoder initialized with new variables without Gaussian noise =====\n'
python run_encoder.py --test_trained --without_noise
# evaluating part
printf '\n===== Display embeddings distribution from autoencoders(as background process) =====\n'
BASE_PATH='/tmp/rbm_aec_reconstr'
python results_validation/visualize_distribution.py --emb_path $BASE_PATH/0_encodings_test_set.npy $BASE_PATH/1_encodings_test_set.npy $BASE_PATH/2_encodings_test_set.npy $BASE_PATH/3_encodings_test_set.npy &
printf '\n===== Evaluate embeddings though SVM =====\n'
python results_validation/svm_clusterization_test.py --test_cases rbm:0 aec_rbm:0 aec_rbm:1
printf '\n===== Evaluate embeddings though most similar accuracy\n'
python results_validation/found_similiar.py --test_cases rbm:0 aec_rbm:0 aec_rbm:1
