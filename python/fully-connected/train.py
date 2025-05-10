from data import DataLoaderCIFAR
from modelFullyConnected import FullyConnectedModel
import tensorflow as tf
import datetime, os

class Trainer:
    def __init__(self, model):
        self._model = model
        
    def __call__(self, train_dataset, valid_dataset, epochs, log_root="logs"):
        steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.1,
            decay_steps=epochs * steps_per_epoch,
            alpha=1e-3)

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr_schedule, momentum=0.9, nesterov=True)

        self._model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5)]
        )

        run_dir = os.path.join(log_root, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        callbacks = [
            tf.keras.callbacks.TensorBoard(run_dir, histogram_freq=1),
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, verbose=2)
        ]

        self._model.fit(train_dataset,
                        validation_data=valid_dataset,
                        epochs=epochs,
                        callbacks=callbacks)


if __name__ == "__main__":
    model = FullyConnectedModel("CIFARClassifier")

    data_loader = DataLoaderCIFAR()
    train_dataset = data_loader.train_dataset
    valid_dataset = data_loader.valid_dataset
    
    train = Trainer(model)
    train(train_dataset, valid_dataset, epochs=1)
    
    tf.keras.utils.plot_model(model, "fullyConnectedModel.png", show_shapes=True, dpi=120)

    model.save("fullyConnectedModel.keras")
