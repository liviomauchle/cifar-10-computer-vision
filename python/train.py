from data import DataLoaderMNIST
from model import MyModel
import tensorflow as tf
import datetime

class Trainer:
    def __init__(self, model):
        self._model = model

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
        
        
        # TODO: Implement training loop
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False) # we use from_logits=False as we already used the softmax in the output layer
        
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
        val_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        val_accuracy = tf.keras.metrics.CategoricalAccuracy('test_accuracy')
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                train_step(x_batch_train, y_batch_train) 
                
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            
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
            

if __name__ == "__main__":
    model = MyModel("MNISTClassifier")

    data_loader = DataLoaderMNIST()
    train_dataset = data_loader.train_dataset
    valid_dataset = data_loader.valid_dataset

    train = Trainer(model)
    train(train_dataset, valid_dataset, epochs=20)

    model.save("my_model.keras")
