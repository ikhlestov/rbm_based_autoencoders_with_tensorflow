import time
import datetime

import tensorflow as tf

from rbm_base import RBMBase


class RBMAllAtOnce(RBMBase):
    def build_model(self):
        self._create_placeholders()
        self._create_variables()

        self.updates = []
        inputs = self.inputs
        layers_qtty = self.params.get('layers_qtty', 1)
        for layer_no in range(layers_qtty):
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

        self.encoded_array = hstate_last

        if self.bin_type:
            last_out = hstate_last
        else:
            last_out = hprob_last
        for vis_layer_no in list(reversed(range(layers_qtty))):
            last_out = self._sample_visible_from_hidden(
                hidden_units=last_out, vis_layer_no=vis_layer_no)
        self.reconstruction = last_out
        # add some summaries
        self.cost = tf.sqrt(tf.reduce_mean(
            tf.square(tf.sub(self.inputs, self.reconstruction))))
        tf.scalar_summary("train_loss", self.cost)

        self.summary = tf.merge_all_summaries()

    def train(self):
        self.build_model()
        prev_run_no = self.params.get('run_no', None)
        self.define_runner_folders()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.sess = sess
            if prev_run_no:
                self.saver.restore(sess, self.saves_path)
            else:
                tf.initialize_all_variables().run()
            self.summary_writer = tf.train.SummaryWriter(
                self.logs_dir, sess.graph)
            for epoch in range(self.params['epochs']):
                start = time.time()
                self._epoch_train_step()
                time_cons = time.time() - start
                time_cons = str(datetime.timedelta(seconds=time_cons))
                print("Epoch: %d, time consumption: %s" % (epoch, time_cons))

            self.saver = tf.train.Saver()
            self.saver.save(sess, self.saves_path)
