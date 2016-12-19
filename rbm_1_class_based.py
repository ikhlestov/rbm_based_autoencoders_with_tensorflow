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
        self.params = params
        self.n_features = data_provider.shapes['inputs']
        self.global_step = 0

    def build_model(self):
        self._create_placeholders()
        self._create_variables()
        self.hidden_encode = self._sample_hidden_from_visible(self.inputs)[0]
        self.reconstruction = self._sample_visible_from_hidden(
            self.hidden_encode)

        hprob0, hstate0, vprob, hprob1, hstate1 = self._gibbs_sampling_step(
            self.inputs)

        positive = self._compute_positive_association(
            self.inputs, hstate0)
        negative = tf.matmul(tf.transpose(vprob), hprob1)

        # Update variables
        learning_rate = self.params['learning_rate']
        batch_size = self.params['batch_size']

        self.w_upd8 = self.W.assign_add(
            learning_rate * (positive - negative) / batch_size)

        self.bias_hidden_upd8 = self.bias_hidden.assign_add(
            tf.mul(learning_rate, tf.reduce_mean(
                tf.sub(hprob0, hprob1), 0)
            )
        )

        self.bias_visible_upd8 = self.bias_visible.assign_add(
            tf.mul(learning_rate, tf.reduce_mean(
                tf.sub(self.inputs, vprob), 0)
            )
        )

        self.cost = tf.sqrt(tf.reduce_mean(
            tf.square(tf.sub(self.inputs, vprob))))
        tf.scalar_summary("train_loss", self.cost)
        self.summary = tf.merge_all_summaries()

    def _create_placeholders(self):
        self.inputs = tf.placeholder(
            tf.float32,
            [None, self.n_features],
            name="inputs")
        self.hrand = tf.placeholder(
            tf.float32,
            [None, self.params['num_hidden']],
            name='hrand')
        self.vrand = tf.placeholder(
            tf.float32,
            [None, self.n_features],
            name='vrand')

    def _create_variables(self):
        self.W = tf.Variable(
            tf.truncated_normal(
                shape=[self.n_features, self.params['num_hidden']],
                stddev=0.1),
            name='weights')
        self.bias_hidden = tf.Variable(
            tf.constant(0.1, shape=[self.params['num_hidden']]),
            name='hidden-bias')
        self.bias_visible = tf.Variable(
            tf.constant(0.1, shape=[self.n_features]),
            name='visible-bias')

    @staticmethod
    def _sample_prob(probs, rand):
        return tf.nn.relu(tf.sign(probs - rand))

    def _sample_hidden_from_visible(self, visible_units):
        hid_probs = tf.nn.sigmoid(
            tf.add(
                tf.matmul(visible_units, self.W),
                self.bias_hidden
            )
        )
        hid_states = self._sample_prob(hid_probs, self.hrand)
        return hid_probs, hid_states

    def _sample_visible_from_hidden(self, hidden_units):
        vis_probs = tf.nn.sigmoid(
            tf.add(
                tf.matmul(hidden_units, tf.transpose(self.W)),
                self.bias_visible
            )
        )
        return vis_probs

    def _gibbs_sampling_step(self, visible):
        """Perform one step of gibbs sampling.
        """
        hprobs, hstates = self._sample_hidden_from_visible(visible)
        vprobs = self._sample_visible_from_hidden(hprobs)
        hprobs1, hstates1 = self._sample_hidden_from_visible(vprobs)

        return hprobs, hstates, vprobs, hprobs1, hstates1

    def _compute_positive_association(self, visible, hidden_states):
        positive = tf.matmul(tf.transpose(visible), hidden_states)
        return positive

    def _create_feed_dict(self, data):
        return {
            self.inputs: data,
            self.hrand: np.random.rand(data.shape[0], self.params['num_hidden']),
            self.vrand: np.random.rand(data.shape[0], self.n_features)
        }

    def _epoch_train_step(self):
        params = self.params
        batches = self.data_provider.get_train_set_iter(
            params['batch_size'], params['shuffle'])
        fetches = [self.w_upd8, self.bias_hidden_upd8,
                   self.bias_visible_upd8, self.summary]
        valid_batches = self.data_provider.get_validation_set_iter(
            params['batch_size'], params['shuffle'])
        for batch_no, batch in enumerate(batches):
            data = batch[0]
            feed_dict = self._create_feed_dict(data=data)
            fetched = self.tf_session.run(fetches, feed_dict=feed_dict)
            summary_str = fetched[-1]
            self.summary_writer.add_summary(summary_str, self.global_step)
            self.global_step += 1
            if batch_no % 11 == 0 and self.params['validate']:
                valid_batch = next(valid_batches)
                data = valid_batch[0]
                feed_dict = {self.inputs: data}
                valid_loss = self.tf_session.run(
                    self.cost, feed_dict=feed_dict)
                valid_loss = float(valid_loss)
                summary_str = tf.Summary(value=[
                    tf.Summary.Value(
                        tag="valid_loss", simple_value=valid_loss)]
                )
                self.summary_writer.add_summary(summary_str, self.global_step)

    def _epoch_validate_step(self):
        pass

    def train(self):
        self.build_model()
        self.define_runner_folders()
        self.saver = tf.train.Saver()
        with tf.Session() as self.tf_session:
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
                    [self.W, self.reconstruction], feed_dict=feed_dict)
                results[batch_no * batch_size: (batch_no + 1) * batch_size] = reconstr

                if show:
                    import ipdb; ipdb.set_trace()
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
        main_save_dir = "/tmp/rbm_saves"
        saves_dir = os.path.join(main_save_dir, run_no)
        os.makedirs(saves_dir, exist_ok=True)
        self.saves_path = os.path.join(main_save_dir, "model.ckpt")

    def define_runner_folders(self):
        main_logs_dir = "/tmp/rbm_logs"
        os.makedirs(main_logs_dir, exist_ok=True)
        run_no = str(len(os.listdir(main_logs_dir)))
        self.logs_dir = os.path.join(main_logs_dir, run_no)

        self.get_saves_path(run_no)
