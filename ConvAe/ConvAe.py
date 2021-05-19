import os
import numpy as np
import pandas as pd
import keras as K
import matplotlib.pyplot as plt
from keras import models, layers
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

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

def addNoise(x):
    ret = np.copy(x)
    for i in range(len(x)):
        ret[i] += np.random.normal(0, 1, (28,28,1))
    return ret

# Add gaussian noise
trainXn = addNoise(trainX)
testXn = addNoise(testX)

#plt.imshow(np.reshape(testX[3], (28,28)), cmap='gray')
#plt.show()

input = K.Input(shape=(28, 28, 1))

x = layers.Conv2D(16, (3, 3),  padding='same')(input)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(8, (3, 3),  padding='same')(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(8, (3, 3),  padding='same')(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.BatchNormalization()(x)

x = layers.Flatten()(x)
x = layers.Dense(4)(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)
encoded = layers.Dense(2, name='encoder')(x)
x = layers.Dense(4)(encoded)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Reshape((2,2,1))(x)

x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3),  padding='same')(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3),  padding='same')(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3))(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

model = K.Model(input, decoded)
modelEncoded = K.Model(model.input, model.get_layer('encoder').output)


model.summary()

model.compile(optimizer=K.optimizers.Adam(learning_rate=0.001), 
        #loss=keras.losses.mean_squared_error,
        loss=K.losses.binary_crossentropy,
        metrics=[])

stopping = K.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=40,
        mode="auto",
        baseline=None,
        min_delta=0.0005,
        restore_best_weights=True)

# Training
#
epochs = 750
batchSize = 2048

#model = K.models.load_model('models/m')
#model = K.models.load_model('models/noise')

#hist = model.fit(trainXn, trainX, batchSize, epochs, validation_data=(testXn, testX), 
#                 callbacks=[stopping])

hist = model.fit(trainX, trainX, batchSize, epochs, validation_data=(testX, testX), 
                 callbacks=[stopping])

#model.save('models/m')
modelEncoded.save('models/m/encoder')
#model.save('models/noise')


out = modelEncoded.predict(testX)

# Plot encoded representation
colormap = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'brown', 'orange', 'pink'])
plt.scatter(out[:,0], out[:,1], c=colormap[testY])

import matplotlib.patches as mpatches
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandel', 'Shirt', 
                    'Sneaker', 'Bag', 'Ankle boot']
patches = [mpatches.Patch(color=c, label=l) for c,l in zip(colormap, classes)]
plt.legend(handles=patches)
plt.show()




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

#plotLoss('training', hist)


# Test image reconstruction
idxs = [0,1,2,6]
testCount = 4

# Reconstruct normal image
#out = model.predict(testX[idxs])
#fig, axs = plt.subplots(2, testCount)
#for i in range(2):
#    for ji, j in enumerate(idxs):
#        axs[i,ji].imshow(np.reshape(out[ji] if i != 0 else testX[j], 
#                                   (28,28)), cmap='gray')

# Reconstructed noise images
#out = model.predict(testXn[idxs])
#fig, axs = plt.subplots(3, testCount)
#for i in range(3):
#    for ji, j in enumerate(idxs):
#        if i == 0:
#            axs[i,ji].imshow(np.reshape(testX[j], (28,28)), cmap='gray')
#        elif i == 1:
#            axs[i,ji].imshow(np.reshape(testXn[j], (28,28)), cmap='gray')
#        else:
#            axs[i,ji].imshow(np.reshape(out[ji], (28,28)), cmap='gray')
#
#plt.show()
#plt.savefig('recon_noise.jpg')

#plt.imshow(np.reshape(out[0], (28,28)), cmap='gray')
#plt.show()
#plt.imshow(np.reshape(out[1], (28,28)), cmap='gray')
#plt.show()
#plt.imshow(np.reshape(out[2], (28,28)), cmap='gray')
#plt.show()



