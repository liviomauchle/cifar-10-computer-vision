import numpy as np
import tensorflow as tf


class DataLoaderCIFAR:
    """Provide train, validation, and test datasets of the CIFAR dataset."""

    def __init__(self, validation_dataset_size=5000, mini_batch_size=32):
        # Load CIFAR data
        cifar = tf.keras.datasets.cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar.load_data()

        # Split training set -> training + validation
        valid_images = train_images[:validation_dataset_size]
        valid_labels = train_labels[:validation_dataset_size]

        train_images = train_images[validation_dataset_size:]
        train_labels = train_labels[validation_dataset_size:]

        def preprocess(x, y):
            # Normalize pixel values to be between 0 and 1
            x = tf.cast(x, tf.float32) / 255.0
            return x, y

        self._train_dataset = (
            tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(buffer_size=10000)
            .batch(mini_batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        self._valid_dataset = (
            tf.data.Dataset.from_tensor_slices((valid_images, valid_labels))
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(mini_batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        self._test_dataset = (
            tf.data.Dataset.from_tensor_slices((test_images, test_labels))
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