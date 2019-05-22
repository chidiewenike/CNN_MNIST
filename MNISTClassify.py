import sys
import keras
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical
from sys import exit
import numpy as np

model = models.load_model('MNIST.h5')

(train_data, train_labels), (test_data, test_labels) \
 = mnist.load_data()

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

test_data = test_data.reshape((10000, 28, 28, 1))

test_data = test_data.astype('float32') / 255

for i in range(len(test_data)):
    output = model.predict(test_data[i].reshape((1, 28, 28, 1)))
    print(np.argmax(output))
