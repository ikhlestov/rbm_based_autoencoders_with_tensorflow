import time
import datetime
import os

import tensorflow as tf
import numpy as np
import tflearn
from PIL import Image
from scipy.misc import toimage

from vis_utils import tile_raster_images


class Encoder:
    def __init__(self, data_provider, params, rbm_run_no=None):
        self.data_provider = data_provider
        params['global_step'] = params.get('global_step', 0)
        self.params = dict(params)
        self.n_features = data_provider.shapes['inputs']
        self.layers_qtty = params.get('layers_qtty', 1)
        self.rbm_saves_dir = "/tmp/rbm_saves"
        self.main_save_dir = "/tmp/rbm_aec_saves"
        self.main_logs_dir = "/tmp/rbm_aec_logs"
        self.bin_code_width = params['layers_sizes'][-1]
        self.pickles_folder = '/tmp/rbm_aec_reconstr'
        self.rbm_run_no = rbm_run_no

    def build_model(self):
        self._create_placeholders()
        self._create_variables()

        result = self.inputs
        # forward layers
        for layer_no in range(self.layers_qtty - 1):
            layer_from = layer_no
            layer_to = layer_no + 1
            weights = getattr(self, self._get_w_name(layer_from, layer_to))
            bias = self._get_bias_tensor(layer_to)
            result = tf.sigmoid(tf.matmul(result, weights) + bias)

        # bin encoding
        layer_from = self.layers_qtty - 1
        layer_to = self.layers_qtty
        weights = getattr(self, self._get_w_name(layer_from, layer_to))
        bias = self._get_bias_tensor(layer_to)
        mat_mul_result = tf.matmul(result, weights) + bias
        # add noise only in case of training
        is_training = tflearn.get_training_mode()
        mat_mul_result = tf.cond(
            is_training,
            lambda: tf.add(mat_mul_result, self.noise),
            lambda: mat_mul_result)
        result = tf.sigmoid(mat_mul_result)
        self.encoded_array = result

        # backward layers
        for layer_no in range(self.layers_qtty, self.layers_qtty * 2):
            layer_from = layer_no
            layer_to = layer_no + 1
            weights = getattr(self, self._get_w_name(layer_from, layer_to))
            bias = self._get_bias_tensor(layer_to)
            result = tf.matmul(result, weights, transpose_b=True) + bias
            if layer_to != (self.layers_qtty * 2):
                result = tf.sigmoid(result)

        self.reconstr = result
        self.reconstr_prob = tf.sigmoid(result)

        # define cost and optimizer
        self.cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                self.reconstr, self.inputs))
        optimizer = tf.train.GradientDescentOptimizer(
            self.params['learning_rate'])
        self.minimization = optimizer.minimize(self.cost)

        tf.scalar_summary("train_loss", self.cost)
        self.summary = tf.merge_all_summaries()

    def _create_placeholders(self):
        self.inputs = tf.placeholder(
            tf.float32,
            [None, self.n_features],
            name="inputs")
        self.noise = tf.placeholder(
            tf.float32,
            [None, self.bin_code_width],
            name="noise")

    def _get_w_name(self, layer_no_from, layer_no_to):
        return 'W_%d_%d' % (layer_no_from, layer_no_to)

    def _get_bias_name(self, layer_no):
        return "bias_%d" % layer_no

    def _get_bias_tensor(self, layer_no):
        return getattr(self, self._get_bias_name(layer_no))

    def _create_variables(self):
        layers_sizes = self.params['layers_sizes']
        created_variables = []
        for layer_idx in range(self.layers_qtty):
            layer_no_from = layer_idx
            layer_no_to = layer_idx + 1
            w_name = self._get_w_name(layer_no_from, layer_no_to)
            weights = tf.Variable(
                tf.truncated_normal(
                    shape=[layers_sizes[layer_no_from], layers_sizes[layer_no_to]],
                    stddev=0.1),
                name=w_name)
            setattr(self, w_name, weights)
            created_variables.append(weights)

            bias_name = self._get_bias_name(layer_no_to)
            bias = tf.Variable(
                tf.constant(0.1, shape=[layers_sizes[layer_no_to]]),
                name=bias_name)
            setattr(self, bias_name, bias)
            created_variables.append(bias)

        layer_counter = self.layers_qtty
        for layer_idx in reversed(range(self.layers_qtty)):
            layer_no_from = layer_counter
            layer_no_to = layer_counter + 1
            w_name = self._get_w_name(layer_no_from, layer_no_to)
            weights = tf.Variable(
                tf.truncated_normal(
                    shape=[layers_sizes[layer_idx], layers_sizes[layer_idx + 1]],
                    stddev=0.1),
                name=w_name)
            setattr(self, w_name, weights)
            created_variables.append(weights)

            bias_name = self._get_bias_name(layer_no_to)
            bias = tf.Variable(
                tf.constant(0.1, shape=[layers_sizes[layer_idx]]),
                name=bias_name)
            setattr(self, bias_name, bias)
            created_variables.append(bias)

            layer_counter += 1

        self._print_variables(
            message="Such variables were created:",
            variables_list=created_variables)

    @staticmethod
    def _print_variables(message, variables_list):
        print(message)
        for var in variables_list:
            print("\t {name}: {shape}".format(
                name=var.name, shape=var.get_shape()))

    def _get_restored_variables_names_forward_layers(self):
        restore_dict = {}
        restored_variables = []
        for layer_no in range(1, self.params['layers_qtty'] + 1):
            w_name = self._get_w_name(layer_no - 1, layer_no)
            weights = getattr(self, w_name)
            restore_dict[w_name] = weights
            restored_variables.append(weights)

            bias_name = "bias_%d" % layer_no
            bias = getattr(self, bias_name)
            restored_variables.append(bias)
            restore_dict[bias_name] = bias

        self._print_variables(
            message="Try to restore such variables for forward layers:",
            variables_list=restored_variables)
        return restore_dict

    def _get_restored_variables_names_backward_layers(self):
        restore_dict = {}
        restored_variables = []

        layer_counter = self.layers_qtty
        for layer_no in reversed(range(self.layers_qtty)):
            w_name_attr = self._get_w_name(layer_counter, layer_counter + 1)
            weights_attr = getattr(self, w_name_attr)
            w_name_saved = self._get_w_name(layer_no, layer_no + 1)
            restore_dict[w_name_saved] = weights_attr
            restored_variables.append(weights_attr)

            bias_name_attr = self._get_bias_name(layer_counter + 1)
            bias_attr = getattr(self, bias_name_attr)
            bias_name_saved = self._get_bias_name(layer_no)
            restore_dict[bias_name_saved] = bias_attr
            restored_variables.append(bias_attr)

            layer_counter += 1

        self._print_variables(
            message="Try to restore such variables for backward layers:",
            variables_list=restored_variables)
        return restore_dict

    def _epoch_train_step(self):
        params = self.params
        batches = self.data_provider.get_train_set_iter(
            params['batch_size'], params['shuffle'], noise=True)
        fetches = [self.minimization, self.summary]
        valid_batches = self.data_provider.get_validation_set_iter(
            params['batch_size'], params['shuffle'], noise=True)
        for batch_no, train_batch in enumerate(batches):
            # train with Gaussian noise prior embeddings layer
            if not self.params['without_noise']:
                tflearn.is_training(True)
            # train without Gaussian noise prior embedding layer
            if self.params['without_noise']:
                tflearn.is_training(False)
            train_inputs, train_targets, train_noise = train_batch
            feed_dict = {
                self.inputs: train_inputs,
                self.noise: train_noise,
            }
            fetched = self.tf_session.run(fetches, feed_dict=feed_dict)
            summary_str = fetched[-1]
            self.summary_writer.add_summary(
                summary_str, self.params['global_step'])
            self.params['global_step'] += 1

            # perform validation
            if batch_no % 11 == 0 and self.params['validate']:
                valid_batch = next(valid_batches)
                valid_inputs, valid_targets, valid_noise = valid_batch
                feed_dict = {
                    self.inputs: valid_inputs,
                    self.noise: valid_noise
                }

                # validation without random noise
                tflearn.is_training(False)
                valid_loss = self.tf_session.run(
                    self.cost, feed_dict=feed_dict)
                valid_loss = float(valid_loss)
                summary_str = tf.Summary(value=[
                    tf.Summary.Value(
                        tag="valid_loss",
                        simple_value=valid_loss
                    )]
                )
                self.summary_writer.add_summary(
                    summary_str, self.params['global_step'])

    def train(self):
        self.build_model()
        self.define_runner_folders()
        # saver to save all params
        saver = tf.train.Saver()
        # restorer to restore only previous variables
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.tf_session = sess

            # initialize encoder from previously trained RBM
            if self.rbm_run_no:
                self.get_rbm_saves_path()
                print("Preload variables from previous RMB run_no: %s" %
                      self.rbm_run_no)
                restore_vars_dict_forward = \
                    self._get_restored_variables_names_forward_layers()
                restorer_forward = tf.train.Saver(restore_vars_dict_forward)
                restorer_forward.restore(self.tf_session, self.preload_path)

                restore_vars_dict_backward = \
                    self._get_restored_variables_names_backward_layers()
                restorer_backward = tf.train.Saver(restore_vars_dict_backward)
                restorer_backward.restore(self.tf_session, self.preload_path)

            # initialize encoder with new variables
            else:
                tf.initialize_all_variables().run()

            self.summary_writer = tf.train.SummaryWriter(
                self.logs_dir, self.tf_session.graph)
            for epoch in range(self.params['epochs']):
                start = time.time()
                self._epoch_train_step()
                time_cons = time.time() - start
                time_cons = str(datetime.timedelta(seconds=time_cons))
                print("Epoch: %d, time consumption: %s" % (epoch, time_cons))
            saver.save(self.tf_session, self.saves_path)
        return self.params

    def test(self, run_no, train_set=False, plot_images=False):
        tf.reset_default_graph()
        self.build_model()
        self.saver = tf.train.Saver()
        self.get_saves_path(run_no)
        os.makedirs(self.pickles_folder, exist_ok=True)
        with tf.Session() as self.tf_session:
            tflearn.is_training(False)
            self.saver.restore(self.tf_session, self.saves_path)
            print("Get embeddings for test set")
            self._get_embeddings(
                self.tf_session, run_no, train_set=False,
                plot_images=plot_images)
            print("Get embeddings for train set")
            self._get_embeddings(
                self.tf_session, run_no, train_set=True,
                plot_images=plot_images)

    def _get_embeddings(self, sess, run_no, train_set=False,
                        plot_images=False):
        batch_size = self.params['batch_size']
        if not train_set:
            test_batches = self.data_provider.get_test_set_iter(
                batch_size, shuffle=False, noise=True)
            total_examples = 10000
        else:
            test_batches = self.data_provider.get_train_set_iter(
                batch_size, shuffle=False, noise=True)
            total_examples = 55000
        reconstructs = np.zeros((total_examples, 784))
        encodings = [
            np.zeros((total_examples, 100)),
            np.zeros((total_examples, 10))
        ]
        for batch_no, batch in enumerate(test_batches):
            feed_dict = {
                self.inputs: batch[0],
                self.noise: batch[2]}
            fetches = []
            for layer_no in range(self.params['layers_qtty']):
                fetches.append(
                    getattr(self, "W_%d_%d" % (layer_no, layer_no + 1)))
            fetches.append(self.encoded_array)
            fetches.append(self.reconstr_prob)
            fetched = self.tf_session.run(
                fetches, feed_dict=feed_dict)
            reconstr = fetched[-1]
            encoded = fetched[-2]
            weights = fetched[:-2]

            slice_start = batch_no * batch_size
            slice_end = (batch_no + 1) * batch_size
            encodings[0][slice_start: slice_end] = encoded
            encodings[1][slice_start: slice_end] = batch[1]
            reconstructs[slice_start: slice_end] = reconstr

            if plot_images:
                tile_h_w = 10
                images_stack = []
                tiled_initial_images = tile_raster_images(
                    batch[0],
                    img_shape=(28, 28),
                    tile_shape=(tile_h_w, tile_h_w),
                    tile_spacing=(2, 2)
                )
                images_stack.append(tiled_initial_images)

                for weight in weights:
                    print("weight shape: ", weight.shape)
                    img_shape = int(np.sqrt(weight.shape[0]))
                    tiled_weights_image = Image.fromarray(tile_raster_images(
                        weight.T,
                        img_shape=(img_shape, img_shape),
                        tile_shape=(tile_h_w, tile_h_w),
                        tile_spacing=(2, 2))
                    )
                    tiled_weights_image.show()
                    toimage(weight.T).show()

                tiled_reconst_image = Image.fromarray(tile_raster_images(
                    reconstr,
                    img_shape=(28, 28),
                    tile_shape=(tile_h_w, tile_h_w),
                    tile_spacing=(2, 2)))
                images_stack.append(np.array(tiled_reconst_image))

                stacked_images = np.hstack(images_stack)
                Image.fromarray(stacked_images).show()
                plot_images = False

        def handle_filename(f_name, train_set):
            """Depends on train or test set change filename"""
            if train_set:
                f_name += '_train_set'
            else:
                f_name += '_test_set'
            return f_name

        reconstr_file = os.path.join(
            self.pickles_folder, '%s_reconstr' % run_no)
        reconstr_file = handle_filename(reconstr_file, train_set)
        np.save(reconstr_file, reconstructs)

        encoded_file = os.path.join(
            self.pickles_folder, '%s_encodings' % run_no)
        encoded_file = handle_filename(encoded_file, train_set)
        np.save(encoded_file, encodings[0])

        labels_file = os.path.join(
            self.pickles_folder, '%s_labels' % run_no)
        labels_file = handle_filename(labels_file, train_set)
        np.save(labels_file, encodings[1])

    def get_saves_path(self, run_no):
        saves_dir = os.path.join(self.main_save_dir, run_no)
        os.makedirs(saves_dir, exist_ok=True)
        self.saves_path = os.path.join(saves_dir, "model.ckpt")

    def get_rbm_saves_path(self):
        """Get pathes where pretrained RMB model was saved"""
        preload_dir = os.path.join(self.rbm_saves_dir, self.rbm_run_no)
        if not os.path.exists(preload_dir):
            print("\t!!!RBM model save folder with run_no: "
                  "%s not exists!!!" % self.rbm_run_no)
            exit()
        self.preload_path = os.path.join(preload_dir, "model.ckpt")
        if not os.path.exists(self.preload_path):
            print("\t!!!RBM model save file .ckpt with run_no: "
                  "%s not exists!!!" % self.rbm_run_no)
            exit()

    def define_runner_folders(self):
        os.makedirs(self.main_logs_dir, exist_ok=True)
        run_no = str(len(os.listdir(self.main_logs_dir)))
        self.params['run_no'] = run_no
        print("Training model no: %s" % run_no)
        self.logs_dir = os.path.join(self.main_logs_dir, run_no)
        notes = self.params.get('notes', False)
        if notes:
            self.logs_dir = self.logs_dir + '_' + notes
        self.get_saves_path(run_no)
        self.run_no = run_no
