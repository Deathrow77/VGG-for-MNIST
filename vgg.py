from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import MaxPool2D, Conv2D, Flatten, Dense, ZeroPadding2D, Dropout
import numpy as np
import keras.backend as K
from keras.datasets import mnist
import keras.utils as np_utils
import matplotlib.pyplot as plt

class VGG:
    @staticmethod
    def build(input_shape, classes):

        model = Sequential()
        
        # Adding VGG Modules
        model.add(ZeroPadding2D((1,1), input_shape=input_shape))
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(64,( 3,3), activation='relu'))
        model.add(MaxPool2D((2,2), strides=(2,2)))
        
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, (3,3), activation='relu'))
        model.add(MaxPool2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3,3), activation='relu'))
        model.add(MaxPool2D((2,2), strides=(2,2)))

        # Flattening the output and adding Fully Connected layers

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(classes, activation='sigmoid'))

        return model


K.set_image_dim_ordering('th')
(X_train, y_train), (X_test, y_test) = mnist.load_data()
IP_SHAPE = (1,28,28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

vgg = VGG.build(input_shape=IP_SHAPE, classes=10)
vgg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = vgg.fit(X_train, y_train, epochs=10, batch_size=128, verbose=1, validation_split=0.2)

score = vgg.evaluate(X_test, y_test, verbose=1)

print("Test Score : ", score[0])
print("Test Accuracy : ", score[1])
print(history.history.keys())
# Accuracy Visualization
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("LeNet Model Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
