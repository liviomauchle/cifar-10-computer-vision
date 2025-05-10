from data import DataLoaderCIFAR
from modelCNN import ModelCNN
import tensorflow as tf
import datetime, os
tf.config.optimizer.set_jit('autoclustering')

class Trainer:
    def __init__(self, model):
        self._model = model

    def __call__(self, train_ds, valid_ds, epochs):
        self._model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=0.1,
            first_decay_steps=50, t_mul=2.0, m_mul=0.8)

        def lr_callback(epoch, lr): return lr_schedule(epoch)

        log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint("best_modelCNN.tf", save_best_only=True, monitor="val_accuracy", mode="max", save_format="tf"),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True),
            tf.keras.callbacks.LearningRateScheduler(lr_callback)
        ]

        self._model.fit(train_ds,
                        validation_data=valid_ds,
                        epochs=epochs,
                        callbacks=callbacks)

if __name__ == "__main__":
    model = ModelCNN(name="modelCNN")
    data_loader = DataLoaderCIFAR(mini_batch_size=256)
    trainer = Trainer(model)
    trainer(data_loader.train_dataset, data_loader.valid_dataset, epochs=300)
