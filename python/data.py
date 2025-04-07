import numpy as np
import tensorflow as tf


class DataLoaderMNIST:
    """Provide train, validation, and test datasets of the MNIST dataset."""

    def __init__(self, validation_dataset_size=5000, mini_batch_size=32):
        # Load MNIST data
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Split training set -> training + validation
        x_valid = x_train[:validation_dataset_size]
        y_valid = y_train[:validation_dataset_size]

        x_train = x_train[validation_dataset_size:]
        y_train = y_train[validation_dataset_size:]

        num_classes = 10

        def preprocess(x, y):
            # Convert images to float32 and normalize to [0, 1]
            x = tf.cast(x, tf.float32) / 255.0
            # One-hot endcoding
            y = tf.one_hot(y, depth=num_classes)
            return x, y

        self._train_dataset = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(buffer_size=10000)
            .batch(mini_batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        self._valid_dataset = (
            tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(mini_batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        self._test_dataset = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(mini_batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )


    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @property
    def test_dataset(self):
        return self._test_dataset