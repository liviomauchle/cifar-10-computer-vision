from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
import keras

@keras.saving.register_keras_serializable()
class FullyConnectedModel(Model):

    def __init__(self, name="CIFARClassifier", weight_decay=1e-4, **kwargs):
        super().__init__(name=name, **kwargs)
        self.flatten = Flatten()

        self.fc1 = Dense(4096, activation="relu",
                         kernel_initializer=HeNormal(),
                         kernel_regularizer=l2(weight_decay))
        self.bn1 = BatchNormalization(); 
        self.do1 = Dropout(0.2)

        self.fc2 = Dense(2048, activation="relu",
                         kernel_initializer=HeNormal(),
                         kernel_regularizer=l2(weight_decay))
        self.bn2 = BatchNormalization(); 
        self.do2 = Dropout(0.2)

        self.fc3 = Dense(1024, activation="relu",
                         kernel_initializer=HeNormal(),
                         kernel_regularizer=l2(weight_decay))
        self.bn3 = BatchNormalization(); 
        self.do3 = Dropout(0.3)

        self.fc4 = Dense(512, activation="relu",
                         kernel_initializer=HeNormal(),
                         kernel_regularizer=l2(weight_decay))
        self.bn4 = BatchNormalization(); 
        self.do4 = Dropout(0.4)

        self.fc5 = Dense(256, activation="relu",
                         kernel_initializer=HeNormal(),
                         kernel_regularizer=l2(weight_decay))
        self.bn5 = BatchNormalization(); 
        self.do5 = Dropout(0.5)

        self.out = Dense(10, activation="softmax")

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.do1(self.bn1(self.fc1(x), training=training), training=training)
        x = self.do2(self.bn2(self.fc2(x), training=training), training=training)
        x = self.do3(self.bn3(self.fc3(x), training=training), training=training)
        x = self.do4(self.bn4(self.fc4(x), training=training), training=training)
        x = self.do5(self.bn5(self.fc5(x), training=training), training=training)
        return self.out(x)
