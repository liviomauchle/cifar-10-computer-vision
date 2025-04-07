import tensorflow as tf
import keras

from data import DataLoaderMNIST
from model import MyModel


class Tester:
    def __init__(self, model):
        self._model = model

    def __call__(self, test_dataset):
        # TODO: Implement testing
        model.evaluate(test_dataset, verbose=2)


if __name__ == "__main__":
    model = keras.models.load_model("my_model.keras")

    data_loader = DataLoaderMNIST()
    test_dataset = data_loader.test_dataset

    test = Tester(model)
    test(test_dataset)
