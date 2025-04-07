import tensorflow as tf
import keras

from data import DataLoaderMNIST
from model import MyModel


class Tester:
    def __init__(self, model):
        self._model = model

    def __call__(self, test_dataset):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
        
        model.evaluate(test_dataset, verbose=2)


if __name__ == "__main__":
    model = keras.models.load_model("my_model.keras")

    data_loader = DataLoaderMNIST()
    test_dataset = data_loader.test_dataset

    test = Tester(model)
    test(test_dataset)
