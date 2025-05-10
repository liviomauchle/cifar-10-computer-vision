import tensorflow as tf
import keras
from data import DataLoaderCIFAR


class Tester:
    def __init__(self, model):
        self._model = model

    def __call__(self, test_dataset):
        test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
        
        print("Test Loss: ", test_loss)
        print("Test Accuracy: ", test_acc)


if __name__ == "__main__":
    model = keras.models.load_model("fullyConnectedModel.keras")

    data_loader = DataLoaderCIFAR()
    test_dataset = data_loader.test_dataset

    test = Tester(model)
    test(test_dataset)
