import os
import time
import datetime

import tensorflow as tf
import numpy as np
from PIL import Image

from utils import tile_raster_images


class RBM:
    def __init__(self, data_provider, params):
        self.data_provider = data_provider
        params['global_step'] = params.get('global_step', 0)
        self.params = dict(params)
        self.n_features = data_provider.shapes['inputs']
        self.gibbs_sampling_steps = params.get('gibbs_sampling_steps', 1)
        self.layers_qtty = params.get('layers_qtty', 1)
        self.main_save_dir = "/tmp/rbm_saves"
        self.main_logs_dir = "/tmp/rbm_logs"
        self.model_was_builded = False
        self.bin_type = True

    def build_model(self):
        self._create_placeholders()
        self._create_variables()

        self.updates = []
        inputs = self.inputs
        for layer_no in range(self.layers_qtty):
            layer_from = layer_no
            layer_to = layer_no + 1
            tmp_res = self.rbm_block(
                inputs=inputs, layer_from=layer_from, layer_to=layer_to)
            updates, vprob_last, hprob_last, hstate_last = tmp_res
            if self.bin_type:
                inputs = hstate_last
            else:
                inputs = hprob_last
            self.updates.extend(updates)

        if self.bin_type:
            last_out = hstate_last
        else:
            last_out = hprob_last
        for vis_layer_no in list(reversed(range(self.layers_qtty))):
            last_out = self._sample_visible_from_hidden(
                hidden_units=last_out, vis_layer_no=vis_layer_no)
        self.reconstruction = last_out
        # add some summaries
        self.cost = tf.sqrt(tf.reduce_mean(
            tf.square(tf.sub(self.inputs, self.reconstruction))))
        tf.scalar_summary("train_loss", self.cost)

        self.summary = tf.merge_all_summaries()

    def rbm_block(self, inputs, layer_from, layer_to, vis_unit_type='gauss'):
        # shapes:
        # self.inputs: (?, 784)
        # hprob_first: (?, 100)
        # hstate_first: (?, 100)
        # vprob_last: (?, 784)
        # hprob_last: (?, 100)
        # hstate_last: (?, 100)

        visib_inputs_initial = inputs
        vprob_last = inputs
        for step_no in range(self.gibbs_sampling_steps):
            gibbs_sample = self._gibbs_sampling_step(
                vprob_last, hid_layer_no=layer_to)
            hprob, hstate, vprob_last, hprob_last, hstate_last = gibbs_sample
            if step_no == 0:
                hprob_first = hprob
                hstate_first = hstate

        learning_rate = self.params['learning_rate']
        batch_size = self.params['batch_size']

        updates = []
        ### define updates for layer
        # diff between inputs and last reconstruction
        bias_layer_from = getattr(self, 'bias_%d' % layer_from)
        bias_layer_from_upd = bias_layer_from.assign_add(
            tf.mul(learning_rate, tf.reduce_mean(
                tf.sub(visib_inputs_initial, vprob_last), 0)
            )
        )
        updates.append(bias_layer_from_upd)

        # diff between first hid.units state and last hid.units state
        bias_layer_to = getattr(self, 'bias_%d' % layer_to)
        if self.bin_type:
            bias_layer_to_sub = tf.sub(hstate_first, hstate_last)
        else:
            bias_layer_to_sub = tf.sub(hprob_first, hprob_last)
        bias_layer_to_upd = bias_layer_to.assign_add(
            tf.mul(learning_rate, tf.reduce_mean(
                bias_layer_to_sub, 0)
            )
        )
        updates.append(bias_layer_to_upd)

        # inside original code positive depends on 'bin' or 'gauss'
        # visible units type
        # in case of 'bin' we compare visible with states
        # in case of 'gauss' we compare visible with probabilities
        # diff on the first iteration betweeen inputs and dreaming
        if self.bin_type:
            positive = tf.matmul(
                tf.transpose(visib_inputs_initial), hstate_first)
            negative = tf.matmul(tf.transpose(vprob_last), hstate_last)
        else:
            positive = tf.matmul(
                tf.transpose(visib_inputs_initial), hprob_first)
            negative = tf.matmul(tf.transpose(vprob_last), hprob_last)

        weights_from_to = getattr(self, "W_%d_%d" % (layer_from, layer_to))
        weights_from_to_upd = weights_from_to.assign_add(
            (learning_rate / batch_size) * (positive - negative))
        updates.append(weights_from_to_upd)

        return updates, vprob_last, hprob_last, hstate_last

    def _create_placeholders(self):
        self.inputs = tf.placeholder(
            tf.float32,
            [None, self.n_features],
            name="inputs")
        tf.stop_gradient(self.inputs)

    def _create_variables(self):
        # weights for connection between layers 0 and 1
        layers_sizes = self.params['layers_sizes']
        for layer_idx in range(self.layers_qtty):
            layer_no_from = layer_idx
            layer_no_to = layer_idx + 1
            w_name = 'W_%d_%d' % (layer_no_from, layer_no_to)
            weights = tf.Variable(
                tf.truncated_normal(
                    shape=[layers_sizes[layer_no_from], layers_sizes[layer_no_to]],
                    stddev=0.1),
                name=w_name)
            tf.stop_gradient(weights)
            setattr(self, w_name, weights)

            bias_name = "bias_%d" % layer_no_to
            bias = tf.Variable(
                tf.constant(0.1, shape=[layers_sizes[layer_no_to]]),
                name=bias_name)
            tf.stop_gradient(bias)
            setattr(self, bias_name, bias)

        self.bias_0 = tf.Variable(
            tf.constant(0.1, shape=[layers_sizes[0]]),
            name='bias_0')
        tf.stop_gradient(self.bias_0)

    @staticmethod
    def _sample_prob(probs):
        """Return binary samples 0 or 1"""
        rand = tf.random_uniform(tf.shape(probs), 0, 1)
        return tf.nn.relu(tf.sign(probs - rand))

    def _sample_hidden_from_visible(self, visible_units, hid_layer_no):
        vis_layer_no = hid_layer_no - 1
        weights = getattr(self, 'W_%d_%d' % (vis_layer_no, hid_layer_no))
        bias = getattr(self, 'bias_%d' % hid_layer_no)
        hid_probs = tf.nn.sigmoid(
            tf.add(
                tf.matmul(visible_units, weights),
                bias
            )
        )
        hid_states = self._sample_prob(hid_probs)
        return hid_probs, hid_states

    def _sample_visible_from_hidden(self, hidden_units, vis_layer_no):
        hid_layer_no = vis_layer_no + 1
        weights = getattr(self, 'W_%d_%d' % (vis_layer_no, hid_layer_no))
        bias = getattr(self, 'bias_%d' % vis_layer_no)
        vis_probs = tf.nn.sigmoid(
            tf.add(
                tf.matmul(hidden_units, tf.transpose(weights)),
                bias
            )
        )
        return vis_probs

    def _gibbs_sampling_step(self, visible, hid_layer_no=1):
        """Perform one step of gibbs sampling.
        """
        vis_layer_no = hid_layer_no - 1
        hprobs, hstates = self._sample_hidden_from_visible(
            visible, hid_layer_no)
        vprobs = self._sample_visible_from_hidden(
            hprobs, vis_layer_no)
        if self.bin_type and hid_layer_no > 1:
            vprobs = self._sample_prob(vprobs)
        hprobs1, hstates1 = self._sample_hidden_from_visible(
            vprobs, hid_layer_no)

        return hprobs, hstates, vprobs, hprobs1, hstates1

    def _compute_positive_association(self, visible, hidden_states):
        positive = tf.matmul(tf.transpose(visible), hidden_states)
        return positive

    def _create_feed_dict(self, data):
        return {
            self.inputs: data,
        }

    def _epoch_train_step(self):
        params = self.params
        batches = self.data_provider.get_train_set_iter(
            params['batch_size'], params['shuffle'])
        fetches = [self.updates, self.summary]
        valid_batches = self.data_provider.get_validation_set_iter(
            params['batch_size'], params['shuffle'])
        for batch_no, train_batch in enumerate(batches):
            train_inputs, train_targets = train_batch
            feed_dict = {self.inputs: train_inputs}
            fetched = self.tf_session.run(fetches, feed_dict=feed_dict)
            summary_str = fetched[-1]
            self.summary_writer.add_summary(
                summary_str, self.params['global_step'])
            self.params['global_step'] += 1
            # perform validation
            if batch_no % 11 == 0 and self.params['validate']:
                valid_batch = next(valid_batches)
                valid_inputs, valid_targets = valid_batch
                feed_dict = {self.inputs: valid_inputs}
                valid_loss = self.tf_session.run(
                    self.cost, feed_dict=feed_dict)
                valid_loss = float(valid_loss)
                summary_str = tf.Summary(value=[
                    tf.Summary.Value(
                        tag="valid_loss", simple_value=valid_loss)]
                )
                self.summary_writer.add_summary(
                    summary_str, self.params['global_step'])

    def _epoch_validate_step(self):
        pass

    def train(self):
        if not self.model_was_builded:
            self.build_model()
            self.model_was_builded = True
        prev_run_no = self.params.get('run_no', None)
        self.define_runner_folders()
        self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as self.tf_session:
            if prev_run_no:
                self.saver.restore(self.tf_session, self.saves_path)
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
            self.saver.save(self.tf_session, self.saves_path)
        return self.params

    def test(self, run_no):
        self.build_model()
        self.saver = tf.train.Saver()
        self.get_saves_path(run_no)
        show = True
        with tf.Session() as self.tf_session:
            self.saver.restore(self.tf_session, self.saves_path)
            batch_size = self.params['batch_size']
            test_batches = self.data_provider.get_test_set_iter(
                batch_size, shuffle=False)
            results = np.zeros((10000, 784))
            for batch_no, batch in enumerate(test_batches):
                feed_dict = {self.inputs: batch[0]}
                weights, reconstr = self.tf_session.run(
                    [self.W_0_1, self.reconstruction], feed_dict=feed_dict)
                results[batch_no * batch_size: (batch_no + 1) * batch_size] = reconstr

                if show:
                    images_stack = []
                    tiled_initial_images = tile_raster_images(
                        batch[0],
                        img_shape=(28, 28),
                        tile_shape=(10, 10),
                        tile_spacing=(2, 2)
                    )
                    images_stack.append(tiled_initial_images)

                    side_h = 10
                    curr_W = weights
                    # import ipdb; ipdb.set_trace()
                    tiled_weights_image = Image.fromarray(tile_raster_images(
                        curr_W.T,
                        img_shape=(28, 28),
                        tile_shape=(side_h, side_h),
                        tile_spacing=(2, 2))
                    )
                    images_stack.append(np.array(tiled_weights_image))

                    tiled_reconst_image = Image.fromarray(tile_raster_images(
                        reconstr,
                        img_shape=(28, 28),
                        tile_shape=(10, 10),
                        tile_spacing=(2, 2)))
                    images_stack.append(np.array(tiled_reconst_image))

                    stacked_images = np.hstack(images_stack)
                    Image.fromarray(stacked_images).show()
                    show = False

            np.save('/tmp/reconstr', results)

    def get_saves_path(self, run_no):
        saves_dir = os.path.join(self.main_save_dir, run_no)
        os.makedirs(saves_dir, exist_ok=True)
        self.saves_path = os.path.join(saves_dir, "model.ckpt")

    def define_runner_folders(self):
        run_no = self.params.get('run_no', None)
        os.makedirs(self.main_logs_dir, exist_ok=True)
        if run_no is None:
            run_no = str(len(os.listdir(self.main_logs_dir)))
            self.params['run_no'] = run_no
        print("Training model no: %s" % run_no)
        self.logs_dir = os.path.join(self.main_logs_dir, run_no)
        self.get_saves_path(run_no)
