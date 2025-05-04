from tensorflow.keras import Model, layers, saving, Sequential


# The decorator "@keras.saving.register_keras_serializable()" allows you to save
# your trained model to a .keras file and then load it from the file for testing.
@saving.register_keras_serializable()
class CNNModel(Model):
    """Neural network to classify CIFAR images."""

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.data_augment = Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1, 0.1),
        ])
        
        self.zero_pad = layers.ZeroPadding2D((3,3))
        
        # Conv block 1
        self.conv1 = layers.Conv2D(32, 3, padding='valid', use_bias=False)
        self.bn1   = layers.BatchNormalization()
        self.act1  = layers.Activation('relu')
        self.pool1 = layers.MaxPooling2D(2)
        
        # Conv block 2
        self.conv2 = layers.Conv2D(64, 3, padding='same', use_bias=False)
        self.bn2   = layers.BatchNormalization()
        self.act2  = layers.Activation('relu')
        self.pool2 = layers.MaxPooling2D(2)
        
        # Conv block 3
        self.conv3 = layers.Conv2D(64, 3, padding='same', use_bias=False)
        self.bn3   = layers.BatchNormalization()
        self.act3  = layers.Activation('relu')
        
        # Conv block 4
        self.conv4 = layers.Conv2D(128, 3, padding='same', use_bias=False)
        self.bn4   = layers.BatchNormalization()
        self.act4  = layers.Activation('relu')
        self.pool4 = layers.MaxPooling2D(2)
        
        # Head
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.4)
        self.dense1  = layers.Dense(512, activation='relu')
        self.out     = layers.Dense(10)

    def call(self, x, training=False):
        x = self.data_augment(x)
        x = self.zero_pad(x)
        
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)
        
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.act4(x)
        x = self.pool4(x)
        
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense1(x)
        return self.out(x)
