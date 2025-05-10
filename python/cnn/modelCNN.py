import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers, initializers
import keras

@keras.saving.register_keras_serializable()
class BasicBlock(layers.Layer):
    def __init__(self, filters, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, 3, stride, padding='same',
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=regularizers.l2(5e-4))
        self.bn1   = layers.BatchNormalization()
        self.relu  = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, 3, 1, padding='same',
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=regularizers.l2(5e-4))
        self.bn2   = layers.BatchNormalization()
        if stride != 1:
            self.shortcut = layers.Conv2D(filters, 1, stride,
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=regularizers.l2(5e-4))
        else:
            self.shortcut = lambda x, *_: x

    def call(self, x, training=False):
        out = self.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        out += self.shortcut(x)
        return self.relu(out)

@keras.saving.register_keras_serializable()
class ModelCNN(Model):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(64, 3, 1, padding='same',
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=regularizers.l2(5e-4))
        self.bn1   = layers.BatchNormalization()
        self.relu  = layers.ReLU()

        self.stage2 = self._make_stage(64, 2, stride=1)
        self.stage3 = self._make_stage(128, 2, stride=2)
        self.stage4 = self._make_stage(256, 2, stride=2)

        self.gap   = layers.GlobalAveragePooling2D()
        self.fc1   = layers.Dense(128, activation='relu',
                                  kernel_initializer=initializers.HeNormal())
        self.drop  = layers.Dropout(0.2)
        self.fc2   = layers.Dense(num_classes, activation='softmax')

    def _make_stage(self, filters, blocks, stride):
        layers_list = [BasicBlock(filters, stride)]
        for _ in range(1, blocks):
            layers_list.append(BasicBlock(filters, 1))
        return tf.keras.Sequential(layers_list)

    def call(self, x, training=False):
        x = self.relu(self.bn1(self.conv1(x), training=training))
        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.gap(x)
        x = self.drop(self.fc1(x), training=training)
        return self.fc2(x)
