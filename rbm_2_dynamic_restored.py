import os
import time
import datetime

import tensorflow as tf
from PIL import Image

from utils import tile_raster_images
from rbm_1_class_based import RBM


class RBMDynamic(RBM):
    def build_model(self):
        self._create_placeholders()
        self._create_variables()

        # pass input through previous layers to construct right input
        inputs = self.inputs
        for layer_no in range(self.layers_qtty - 1):
            hid_layer_no = layer_no + 1
            hid_probs, hid_states = self._sample_hidden_from_visible(
                inputs, hid_layer_no)
            if self.bin_type:
                inputs = hid_states
            else:
                inputs = hid_probs

        # now enable RBM for last two layers
        self.updates = []
        layer_from = self.layers_qtty - 1
        layer_to = self.layers_qtty
        tmp_res = self.rbm_block(
            inputs=inputs, layer_from=layer_from, layer_to=layer_to)
        updates, vprob_last, hprob_last, hstate_last = tmp_res
        self.updates.extend(updates)

        self.encoded_array = hstate_last

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

    def _get_restored_variables_names(self):
        """Get variables that should be restored from previous run for
        all layers but last one.
        """
        restore_dict = {}
        for layer_no in range(self.params['layers_qtty']):
            bias_name = "bias_%d" % layer_no
            restore_dict[bias_name] = getattr(self, bias_name)
            if layer_no > 0:
                w_name = "W_%d_%d" % (layer_no - 1, layer_no)
                restore_dict[w_name] = getattr(self, w_name)
        return restore_dict

    def _get_new_variables_names(self):
        """Get variables for last layer - it should be initialized in
        any case.
        """
        last_layer = self.params['layers_qtty']
        w_name = "W_%d_%d" % (last_layer - 1, last_layer)
        w = getattr(self, w_name)
        bias_name = "bias_%d" % last_layer
        bias = getattr(self, bias_name)
        new_vars = [w, bias]
        return new_vars

    def train(self):
        self.build_model()
        prev_run_no = self.params.get('run_no', None)
        self.define_runner_folders()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.tf_session = sess

            if prev_run_no:
                print("Restore variables from previous run:")
                restore_vars_dict = self._get_restored_variables_names()
                for var_name in restore_vars_dict.keys():
                    print("\t%s" % var_name)
                restorer = tf.train.Saver(restore_vars_dict)
                restorer.restore(sess, self.saves_path)
                print("Initialize not restored variables:")
                new_variables = self._get_new_variables_names()
                for var in new_variables:
                    print("\t%s" % var.name)
                sess.run(tf.initialize_variables(new_variables))

            else:
                print("Initialize new variables")
                tf.initialize_all_variables().run()
            self.summary_writer = tf.train.SummaryWriter(
                self.logs_dir, sess.graph)
            for epoch in range(self.params['epochs']):
                start = time.time()
                self._epoch_train_step()
                time_cons = time.time() - start
                time_cons = str(datetime.timedelta(seconds=time_cons))
                print("Epoch: %d, time consumption: %s" % (epoch, time_cons))

            # Save all trained variables
            saver = tf.train.Saver()
            saver.save(sess, self.saves_path)
        return self.params
