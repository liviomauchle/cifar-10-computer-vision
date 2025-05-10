from tensorflow.keras import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
import keras


# The decorator "@keras.saving.register_keras_serializable()" allows you to save
# your trained model to a .keras file and then load it from the file for testing.
@keras.saving.register_keras_serializable()
class FullyConnectedModel(Model):
    """Neural network to classify CIFAR images."""

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.flattened = Flatten(input_shape=(32, 32, 3), name="flatten")
        
        self.dense1 = Dense(512, kernel_initializer=HeNormal(), activation="relu", name="dense1")
        self.b1 = BatchNormalization()
        self.d1 = Dropout(0.3)
        
        self.dense2 = Dense(256, kernel_initializer=HeNormal(), activation="relu", name="dense1")
        self.b2 = BatchNormalization()
        self.d2 = Dropout(0.3)
        
        self.dense3 = Dense(256, kernel_initializer=HeNormal(), activation="relu", name="dense1")
        self.b3 = BatchNormalization()
        self.d3 = Dropout(0.3)
        
        self.dense4 = Dense(128, kernel_initializer=HeNormal(), activation="relu", name="dense1")
        self.b4 = BatchNormalization()
        self.d4 = Dropout(0.3)
        
        self.dense5 = Dense(128, activation="relu", kernel_initializer=HeNormal(), name="dense2")
        self.b5 = BatchNormalization()
        self.d5 = Dropout(0.3)
        
        self.dense6 = Dense(128, activation="relu", kernel_initializer=HeNormal(), name="dense2")
        self.b6 = BatchNormalization()
        self.d6 = Dropout(0.3)
        
        self.dense7 = Dense(64, activation="relu", kernel_initializer=HeNormal(), name="dense2")
        self.b7 = BatchNormalization()
        self.d7 = Dropout(0.3)
        
        self.dense8 = Dense(64, activation="relu", kernel_initializer=HeNormal(), name="dense3")
        self.b8 = BatchNormalization()
        self.d8 = Dropout(0.3)
        
        self.dense9 = Dense(32, activation="relu", kernel_initializer=HeNormal(), name="dense4")
        self.b9 = BatchNormalization()
        self.d9 = Dropout(0.3)
        
        self.out = Dense(10, name="output")

    def call(self, x, training=False):
        x = self.flattened(x)
        
        x = self.dense1(x)
        x = self.b1(x, training=training)
        # x = self.d1(x, training=training)
        
        x = self.dense2(x)
        x = self.b2(x, training=training)
        x = self.d2(x, training=training)
        
        x = self.dense3(x)
        x = self.b3(x, training=training)
        x = self.d3(x, training=training)
        
        x = self.dense4(x)
        x = self.b4(x, training=training)
        x = self.d4(x, training=training)
        
        x = self.dense5(x)
        x = self.b5(x, training=training)
        x = self.d5(x, training=training)
        
        x = self.dense6(x)
        x = self.b6(x, training=training)
        x = self.d6(x, training=training)
        
        x = self.dense7(x)
        x = self.b7(x, training=training)
        x = self.d7(x, training=training)
        
        x = self.dense8(x)
        x = self.b8(x, training=training)
        x = self.d8(x, training=training)
        
        x = self.dense9(x)
        x = self.b9(x, training=training)
        x = self.d9(x, training=training)
        
        out = self.out(x)
        return out
