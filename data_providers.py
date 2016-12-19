import numpy as np
from tensorflow.examples.tutorials import mnist


class BaseDataProvider:
    def get_shapes(self):
        """Return shapes of inputs and targets"""
        raise NotImplementedError

    def get_train_set(self):
        raise NotImplementedError

    def get_validation_set(self):
        raise NotImplementedError

    def get_test_set(self):
        raise NotImplementedError

    def get_train_set_iter(self, batch_size):
        """Return generator with train set"""
        raise NotImplementedError

    def get_validation_set_iter(self, batch_size):
        """Return generator with validation set"""
        raise NotImplementedError

    def get_test_set_iter(self, batch_size):
        """Return generator with test set"""
        raise NotImplementedError


class MNISTDataProvider(BaseDataProvider):
    def __init__(self):
        self.mnist = mnist.input_data.read_data_sets(
            "/tmp/MNIST_data/", one_hot=True)
        self._shapes = None

    @property
    def shapes(self):
        if not self._shapes:
            self.get_shapes()
        return self._shapes

    def get_shapes(self):
        batch = self.mnist.train.next_batch(1)
        self._shapes = {
            "inputs": batch[0].shape[-1],
            "tragets": batch[1].shape[-1],
        }

    def get_train_set(self):
        data = self.mnist.train
        return data.images, data.labels

    def get_validation_set(self):
        data = self.mnist.validation
        return data.images, data.labels

    def get_test_set(self):
        data = self.mnist.test
        return data.images, data.labels

    def get_generator(self, data, batch_size, shuffle=True):
        quantity = data.images.shape[0]
        if shuffle:
            indexes = np.random.permutation(quantity)
            images_perm = data.images[indexes]
            labels_perm = data.labels[indexes]
        else:
            images_perm = data.images
            labels_perm = data.labels
        for i in range(quantity // batch_size):
            start = i * batch_size
            end = (i + 1) * batch_size
            images = images_perm[start: end]
            labels = labels_perm[start: end]
            yield images, labels

    def get_train_set_iter(self, batch_size, shuffle=True):
        data = self.mnist.train
        return self.get_generator(data, batch_size, shuffle)

    def get_validation_set_iter(self, batch_size, shuffle=True):
        data = self.mnist.validation
        return self.get_generator(data, batch_size, shuffle)

    def get_test_set_iter(self, batch_size, shuffle=True):
        data = self.mnist.test
        return self.get_generator(data, batch_size, shuffle)
