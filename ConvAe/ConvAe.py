import numerapi
import numpy as np
import pandas as pd
import keras as K
import matplotlib.pyplot as plt
from keras import models, layers
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28, 28, 1)

    return np.copy(images), np.copy(labels)

trainX, trainY = load_mnist('data', kind='train')
testX, testY = load_mnist('data', kind='t10k')

trainX = np.divide(trainX, 255)
testX = np.divide(testX, 255)

input = K.Input(shape=(28, 28, 1))

x = layers.Conv2D(16, (3, 3),  padding='same')(input)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3),  padding='same')(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3),  padding='same')(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Flatten()(x)
x = layers.Dense(4)(x)
x = layers.Activation('relu')(x)
encoded = layers.Dense(2)(x)
x = layers.Dense(4)(encoded)
x = layers.Activation('relu')(x)
x = layers.Reshape((2,2,1))(x)

x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3),  padding='same')(x)
x = layers.Activation('relu')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3),  padding='same')(x)
x = layers.Activation('relu')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3))(x)
x = layers.Activation('relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

model = K.Model(input, decoded)


model.summary()

model.compile(optimizer=K.optimizers.Adam(learning_rate=0.001), 
        #loss=keras.losses.mean_squared_error,
        loss=K.losses.binary_crossentropy,
        metrics=[])

stopping = K.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="auto",
        baseline=None,
        min_delta=0.0005,
        restore_best_weights=True)
hist = model.fit(trainX, trainX, 256, 100, validation_data=(testX, testX), 
                 callbacks=[stopping])
model.save('models/m')

def plotLoss(name, history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(name + '_loss.jpg')
    plt.close()

plotLoss('training', hist)

