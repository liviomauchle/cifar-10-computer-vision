from data import DataLoaderCIFAR
from modelFullyConnected import FullyConnectedModel
import tensorflow as tf
import datetime, os
import visualkeras
from tensorflow.keras import Input, Model

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

    data_loader = DataLoaderCIFAR(mini_batch_size=512)
    train_dataset = data_loader.train_dataset
    valid_dataset = data_loader.valid_dataset
    
    train = Trainer(model)
    train(train_dataset, valid_dataset, epochs=300)
    
    # inp = Input(shape=(32, 32, 3))
    # out = model.call(inp)  
    # func_model = Model(inputs=inp, outputs=out)
    # visualkeras.layered_view(func_model, to_file='visualkeras_output.png', legend=True, show_dimension=True)

    model.save("fullyConnectedModel.keras")
