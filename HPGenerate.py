import sys
import keras
from keras.preprocessing import image
from keras import models, layers
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from sys import exit

train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    shear_range=0.1)

val_datagen = image.ImageDataGenerator(
    rescale=1./255)

(train_data, train_labels), (test_data, test_labels) \
 = mnist.load_data()

train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

train_data = train_data.reshape((60000, 28, 28, 1))
test_data = test_data.reshape((10000, 28, 28, 1))

train_gen = train_datagen.flow(
                                train_data,
                                train_labels,
                                batch_size=128)

val_gen = val_datagen.flow(
                            test_data,
                            test_labels,
                            batch_size=128)

nn = models.Sequential()
nn.add(layers.Conv2D(
    32, (3, 3), activation='relu',
    kernel_regularizer=None, input_shape=(28, 28, 1)))
nn.add(keras.layers.Dropout(0.0))
nn.add(layers.MaxPooling2D((2, 2)))
nn.add(layers.Conv2D(
    64, (3, 3),
    activation='relu',
    kernel_regularizer=None))
nn.add(keras.layers.Dropout(0.5))
nn.add(layers.MaxPooling2D((2, 2)))
nn.add(layers.Conv2D(
    64, (3, 3),
    activation='relu',
    kernel_regularizer=None))
nn.add(keras.layers.Dropout(0.3))
nn.add(layers.Flatten())
nn.add(layers.Dense(64, activation='relu'))
nn.add(layers.Dense(10, activation='softmax'))

nn.compile(
    optimizer="rmsprop",
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

hst = nn.fit_generator(
    train_gen, epochs=10, steps_per_epoch=1500,
    validation_data=val_gen, validation_steps=250)

nn.save('MNIST.h5')
