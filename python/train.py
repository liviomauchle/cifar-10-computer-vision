from data import DataLoaderMNIST
from model import MyModel
import tensorflow as tf
import datetime
import numpy as np

class Trainer:
    def __init__(self, model):
        self._model = model
        
    def log_misclassified_images(self, dataset, writer, max_images=12):
        misclassified_images = []
            
        for x, y in dataset:
            true_labels = tf.argmax(y, axis=1)
            predictions = model(x, training=False)
            pred_labels = tf.argmax(predictions, axis=1)
   
            mask = tf.not_equal(true_labels, pred_labels)
            indices = tf.where(mask)[:, 0]
            if tf.size(indices) > 0:
                mis_images = tf.gather(x, indices)
                misclassified_images.append(mis_images)

            if sum([tf.shape(batch)[0] for batch in misclassified_images]) >= max_images:
                break
                
        if misclassified_images:
            misclassified_images = tf.concat(misclassified_images, axis=0)
            misclassified_images = np.reshape(misclassified_images[0:max_images], (-1, 28, 28, 1))
            
        with writer.as_default():
            tf.summary.image(str(max_images) + " Missclassified images", misclassified_images, max_outputs=12, step=0)
    

    def __call__(self, train_dataset, valid_dataset, epochs):
        @tf.function
        def train_step(x_batch_train, y_batch_train):
            with tf.GradientTape() as tape:
                pred = model(x_batch_train, training=True)
                loss = loss_fn(y_batch_train, pred)
                    
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
                
            train_loss(loss)
            train_accuracy(y_batch_train, pred)
            return loss
        
        @tf.function
        def test_step(x, y):
            predictions  = model(x, training=False)
            loss = loss_fn(y, predictions)
            
            val_loss(loss)
            val_accuracy(y,  predictions)
            
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False) # we use from_logits=False as we already used the softmax in the output layer
        
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
        val_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        val_accuracy = tf.keras.metrics.CategoricalAccuracy('test_accuracy')
        
        # Definitions for Keras TensorBoard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        
        # Log the Graph
        with train_summary_writer.as_default():
            tf.summary.trace_on(graph=True, profiler=False)
        sample_images, sample_labels = next(iter(train_dataset))
        _ = train_step(sample_images, sample_labels)
        with train_summary_writer.as_default():
            tf.summary.trace_export(name="model_trace", step=0)
        
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                train_step(x_batch_train, y_batch_train) 
                
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
                
                # Lost Histograms of all weights
                for layer in model.layers:
                    for weight in layer.trainable_variables:
                        tf.summary.histogram(weight.name, weight, step=epoch)
            
            for x_batch_val, y_batch_val in valid_dataset:
                test_step(x_batch_val, y_batch_val)
                
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)

            template = 'Epoch {}, Train Loss: {}, Train Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
            print (template.format(epoch+1,
                         train_loss.result(), 
                         train_accuracy.result()*100,
                         val_loss.result(), 
                         val_accuracy.result()*100))

            # Reset metrics every epoch
            train_loss.reset_state()
            val_loss.reset_state()
            train_accuracy.reset_state()
            val_accuracy.reset_state()
            
        self.log_misclassified_images(valid_dataset, val_summary_writer, 12)


if __name__ == "__main__":
    model = MyModel("MNISTClassifier")

    data_loader = DataLoaderMNIST()
    train_dataset = data_loader.train_dataset
    valid_dataset = data_loader.valid_dataset

    train = Trainer(model)
    train(train_dataset, valid_dataset, epochs=5)

    model.save("my_model.keras")
