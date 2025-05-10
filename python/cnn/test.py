import keras
from data import DataLoaderCIFAR

class Tester:
    def __init__(self, model): self._model = model

    def __call__(self, test_dataset):
        loss, acc, top5 = self._model.evaluate(test_dataset, verbose=0)
        print(f"Test-set accuracy: {acc*100:5.2f}%  |  Top-5: {top5*100:5.2f}%")
        return acc

if __name__ == "__main__":
    model = keras.models.load_model("best_modelCNN.keras")
    data_loader = DataLoaderCIFAR()
    Tester(model)(data_loader.test_dataset)
