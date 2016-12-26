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
    def __init__(self, bin_code_width=100):
        self.mnist = mnist.input_data.read_data_sets(
            "/tmp/MNIST_data/", one_hot=True)
        self._shapes = None
        self.bin_code_width = bin_code_width
        self.load_noise()

    @property
    def shapes(self):
        if not self._shapes:
            self.get_shapes()
        return self._shapes

    def load_noise(self):
        self.train_noise_path = "/tmp/MNIST_data/noise_train"
        self.valid_noise_path = "/tmp/MNIST_data/noise_valid"
        self.test_noise_path = "/tmp/MNIST_data/noise_test"
        try:
            # try to load previously generated noise
            self.noise_train = np.load(self.train_noise_path + '.npy')
            self.noise_valid = np.load(self.valid_noise_path + '.npy')
            self.noise_test = np.load(self.test_noise_path + '.npy')
            if self.noise_train.shape[-1] != self.bin_code_width:
                print("Existed noise was preloaded, but bin shape %d != %d" %
                    (self.noise_train.shape[-1], self.bin_code_width))
                raise FileNotFoundError
            print("Existed noise was preloaded")
        except FileNotFoundError:
            # if no previous noise was found - generate and savee new one
            pathes = [
                self.train_noise_path,
                self.valid_noise_path,
                self.test_noise_path
            ]
            shapes = [
                self.mnist.train.images.shape,
                self.mnist.validation.images.shape,
                self.mnist.test.images.shape,
            ]
            for shape, path in zip(shapes, pathes):
                self.generate_save_noise(shape, path)
            print("New noise was generated")

    def generate_save_noise(self, shape, path):
        data_qtty = shape[0]
        noise_name = path.split('/')[-1]
        mean = 0
        SD = 4
        # variance = SD ** 2
        result = np.zeros((data_qtty, self.bin_code_width))
        for i in range(data_qtty):
            result[i] = np.random.normal(mean, SD, self.bin_code_width)
        setattr(self, noise_name, result)
        # save results for future use
        np.save(path, result)

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

    def get_generator(self, data, batch_size, shuffle=True, noise_data=None):
        quantity = data.images.shape[0]
        if noise_data is None:
            noise_perm = None
        if shuffle:
            indexes = np.random.permutation(quantity)
            images_perm = data.images[indexes]
            labels_perm = data.labels[indexes]
            if noise_data is not None:
                noise_perm = noise_data[indexes]
        else:
            images_perm = data.images
            labels_perm = data.labels
            noise_perm = noise_data
        for i in range(quantity // batch_size):
            start = i * batch_size
            end = (i + 1) * batch_size
            images = images_perm[start: end]
            labels = labels_perm[start: end]
            if noise_perm is not None:
                noise = noise_perm[start: end]
                yield images, labels, noise
            else:
                yield images, labels

    def get_train_set_iter(self, batch_size, shuffle=True, noise=False):
        data = self.mnist.train
        noise_data = None
        if noise:
            noise_data = self.noise_train
        return self.get_generator(data, batch_size, shuffle, noise_data)

    def get_validation_set_iter(self, batch_size, shuffle=True, noise=False):
        data = self.mnist.validation
        noise_data = None
        if noise:
            noise_data = self.noise_valid
        return self.get_generator(data, batch_size, shuffle, noise_data)

    def get_test_set_iter(self, batch_size, shuffle=True, noise=False):
        data = self.mnist.test
        noise_data = None
        if noise:
            noise_data = self.noise_test
        return self.get_generator(data, batch_size, shuffle, noise_data)
