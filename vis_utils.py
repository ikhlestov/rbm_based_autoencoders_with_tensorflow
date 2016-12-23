# coding: utf-8
import numpy as np
%pylab

encoded = np.load('/tmp/rbm_reconstr/1_encodings.npy')
encoded_bin = np.load('/tmp/rbm_reconstr/3_encodings.npy')
labels = np.load('/tmp/rbm_reconstr/1_labels.npy')
labels_bin = np.load('/tmp/rbm_reconstr/3_labels.npy')


def get_indexes(index_no):
    return np.where(np.argmax(labels, axis=1) == index_no)[0]


def show_two_images_same(index_no, limit=200):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('sigmoid: %d' % index_no)
    pylab.imshow(encoded[get_indexes(index_no)][:limit])
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('binary: %d' % index_no)
    pylab.imshow(encoded_bin[get_indexes(index_no)][:limit])


def show_two_images_diff(index_no_1, index_no_2, limit=200):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('number %d' % index_no_1)
    pylab.imshow(encoded_bin[get_indexes(index_no_1)][:limit])
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('number %d' % index_no_2)
    pylab.imshow(encoded_bin[get_indexes(index_no_2)][:limit])


bottom = None
bin_type = True
if bin_type:
    encodings = encoded_bin
else:
    encodings = encoded

test_no_1 = 4
test_no_2 = 9
pylab.imshow(encodings[get_indexes(test_no_1)][:200] - encodings[get_indexes(test_no_2)][:200])

pylab.plot(np.mean(encodings[get_indexes(test_no_1)], axis=0) - 
           np.mean(encodings[get_indexes(test_no_2)], axis=0))

# show distribution of indexes for required label
for label in range(2):
    heights = np.sum(encodings[get_indexes(label)], axis=0)
    if bottom is not None:
        pylab.bar(
            range(100),
            heights,
            bottom=bottom,
            color='r')
    else:
        pylab.bar(range(100), heights)
    bottom = heights


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = list('bgrcmy'*2)
numbers = [1, 2]
for label, z in zip(numbers, range(10)):
    xs = range(100)
    ys = np.mean(encodings[get_indexes(label)], axis=0)
    zs = z * 10

    cs = [colors[label]] * 100
    ax.bar(xs, ys, zs=zs, zdir='y', alpha=.7, color=cs)
