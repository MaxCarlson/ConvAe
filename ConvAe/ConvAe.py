import numerapi
import numpy as np
import pandas as pd
import keras as K
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
                               offset=16).reshape(len(labels), 784)

    return np.copy(images), np.copy(labels)

trainX, trainY = load_mnist('data', kind='train')
testX, testY = load_mnist('data', kind='t10k')

#trainX.setflags(write=1)
#testX.setflags(write=1)

trainX = np.divide(trainX, 255)
testX = np.divide(testX, 255)

def addBlock(model, filters, kernel, encoder=True):
    model.add(layers.Conv2D(filters, kernel, padding='same') if encoder
              else layers.Conv2DTranspose(filters, kernel, padding='same'))
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D(padding='same') if encoder 
              else layers.UpSampling2D())

model = models.Sequential()
model.add(layers.Input((784,1)))
#model.add(layers.Dense(28))
model.add(layers.Reshape((28,28,1)))
addBlock(model, 32, (5,5))
addBlock(model, 16, (3,3))
addBlock(model, 8, (2,2))
model.add(layers.Flatten())
model.add(layers.Dense(4))
model.add(layers.Dense(2))
model.add(layers.Dense(4))
model.add(layers.Reshape((2,2,1)))
addBlock(model, 8, (2,2), False)
addBlock(model, 16, (3,3), False)
addBlock(model, 32, (5,5), False)
model.add(layers.Reshape((784,1)))
model.add(layers.Activation(K.activations.sigmoid))

model.compile(optimizer=K.optimizers.Adam(learning_rate=0.001), 
        #loss=keras.losses.mean_squared_error,
        loss=K.losses.binary_crossentropy,
        metrics=[])

model.fit(trainX, trainX, 32, 1)

