import time

import tensorflow as tf
from PIL import Image
import numpy as np

from tensorflow.python.ops import control_flow_ops
from tensorflow.examples.tutorials.mnist import input_data

from utils import tile_raster_images
from data_providers import MNISTDataProvider

# size_vis is the size of the visiable layer
# size_hid is the size of the hidden layer
display = False
side_h = 10
size_vis = 28 * 28
size_hid = side_h * side_h
batch_size = 100  # batch size
learning_rate = 0.01
epochs = 5


k = tf.constant(1)


# helper function
def sampleInt(probs):
    # return binary probabilities from 0 or 1
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


def sample(probs):
    return tf.to_float(sampleInt(probs))


# placeholders
inputs = tf.placeholder(tf.float32, [size_vis, batch_size])
lr = tf.placeholder(tf.float32)
# variables
W = tf.Variable(tf.random_uniform([size_vis, size_hid], -0.005, 0.005))
bias_hidden = tf.Variable(tf.random_uniform([size_hid, 1], -0.005, 0.005))
bias_visible = tf.Variable(tf.random_uniform([size_vis, 1], -0.005, 0.005))

# define graph/algorithm

hidden_state_0 = sample(
    tf.sigmoid(tf.matmul(tf.transpose(W), inputs) + bias_hidden))


# CD-k
# we use tf.while_loop to achieve the multiple (k - 1) gibbs sampling
def rbmGibbs(xx, hh, count, k):
    hid_state = sampleInt(
        tf.sigmoid(tf.matmul(W, hh) + bias_visible))
    reconstruction = sampleInt(
        tf.sigmoid(tf.matmul(tf.transpose(W), hid_state) + bias_hidden))
    return reconstruction, hid_state, count + 1, k


def lessThanK(xk, hk, count, k):
    return count <= k


gibbs_counter = tf.constant(1)

[reconstructed, hid_state, _, _] = control_flow_ops.While(
    lessThanK, rbmGibbs, [inputs, hidden_state_0, gibbs_counter, k], 1, False)

# update rule
positive_weights_update = tf.matmul(inputs, tf.transpose(hidden_state_0))
negative_weights_update = tf.matmul(reconstructed, tf.transpose(hid_state))
w_upd = tf.mul(
    lr / batch_size,
    tf.sub(positive_weights_update, negative_weights_update)
)
bias_hidden_upd = tf.mul(
    lr / batch_size,
    tf.reduce_sum(
        tf.sub(hidden_state_0, hid_state), 1, True)
)
bias_visible_upd = tf.mul(
    lr / batch_size,
    tf.reduce_sum(tf.sub(inputs, reconstructed), 1, True)
)

updates = [
    W.assign_add(w_upd),
    bias_hidden.assign_add(bias_hidden_upd),
    bias_visible.assign_add(bias_visible_upd)
]

# stop gradient to save time and mem
tf.stop_gradient(hidden_state_0)
tf.stop_gradient(reconstructed)
tf.stop_gradient(hid_state)
tf.stop_gradient(w_upd)
tf.stop_gradient(bias_hidden_upd)
tf.stop_gradient(bias_visible_upd)

# run session
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


# loop with batch
mnist_provider = MNISTDataProvider()
step_counter = 0
# alpha = min(0.05, 100 / i)
alpha = learning_rate
for epoch in range(epochs):
    batches = mnist_provider.get_train_set_iter(batch_size=batch_size)
    for batch in batches:
        step_counter += 1
        tr_x = batch[0]
        tr_x = np.transpose(tr_x)
        sess.run(updates, feed_dict={inputs: tr_x, lr: alpha})
        if step_counter % 100 == 0:
            print("epoch: %d, step: %d" % (epoch, step_counter))

        # visualization
        if step_counter % 2500 == 1 and display:
            images_stack = []
            imagex = Image.fromarray(tile_raster_images(
                np.transpose(tr_x),
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(2, 2))
            )
            images_stack.append(np.array(imagex))

            curr_W = sess.run(W)
            image = Image.fromarray(tile_raster_images(
                curr_W.T,
                img_shape=(28, 28),
                tile_shape=(side_h, side_h),
                tile_spacing=(2, 2))
            )
            images_stack.append(np.array(image))

            reconstructed = sess.run(xk1, feed_dict={inputs: tr_x})
            imagexk = Image.fromarray(tile_raster_images(
                reconstructed.T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(2, 2)))
            images_stack.append(np.array(imagexk))

            stacked_images = np.hstack(images_stack)
            Image.fromarray(stacked_images).show()
