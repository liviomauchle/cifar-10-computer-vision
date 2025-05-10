import keras
from data import DataLoaderCIFAR

class Tester:
    def __init__(self, model): self._model = model

    def __call__(self, test_dataset):
        loss, acc = model.evaluate(test_dataset, verbose=0)
        print(f"Loss={loss:.4f}, Accuracy={acc*100:.2f}%")
        return acc

if __name__ == "__main__":
    model = keras.models.load_model("best_modelCNN.tf")
    data_loader = DataLoaderCIFAR()
    Tester(model)(data_loader.test_dataset)
