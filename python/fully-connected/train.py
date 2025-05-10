from data import DataLoaderCIFAR
from modelFullyConnected import FullyConnectedModel
import tensorflow as tf

class Trainer:
    def __init__(self, model):
        self._model = model
        
    def __call__(self, train_dataset, valid_dataset, epochs):
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
    
        history = model.fit(
            train_dataset, 
            epochs=epochs, 
            validation_data=valid_dataset
        )
        
        # plt.plot(history.history['accuracy'], label='accuracy')
        # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.ylim([0.5, 1])
        # plt.legend(loc='lower right')


if __name__ == "__main__":
    model = FullyConnectedModel("CIFARClassifier")

    data_loader = DataLoaderCIFAR()
    train_dataset = data_loader.train_dataset
    valid_dataset = data_loader.valid_dataset
    
    train = Trainer(model)
    train(train_dataset, valid_dataset, epochs=30)

    model.save("fullyConnectedModel.keras")
