import sys
import keras
from keras.preprocessing import image
from keras import models, layers
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from sys import exit


class LyrParams:
    def __init__(self, channels, dropout=0, reg=None, reg_nm="None"):
        self.channels = channels
        self.dropout = dropout
        self.reg = reg
        self.reg_nm = reg_nm

# for line in sys.stdin.readlines():
file_in = open('layer_vals12.txt', 'r')
layer_list = []
convs = []

for line in file_in.readlines():
    for points in line.split(','):
        for point in points.split(','):

            channels = 0
            reg = None
            reg_nm = "None"
            dropout = 0
            channels = int(point.split(';')[0].replace('\n', ''))
            for vals in point.split(';')[1:]:
                if (vals[0] == 'l'):

                    if(vals[1] == '1'):
                        reg = keras.regularizers.l1(
                            float(vals.split('|')[1].replace('\n', '')))
                        reg_nm = "L1"
                    else:
                        reg = keras.regularizers.l2(
                            float(vals.split('|')[1].replace('\n', '')))
                        reg_nm = "L2"
                else:
                    dropout = float(vals.replace('\n', ''))
        layer_list.append(LyrParams(channels, dropout, reg, reg_nm))

    convs.append(layer_list)
    layer_list = []

train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
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

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

for i in range(len(convs)):
    nn = models.Sequential()

    nn.add(layers.Conv2D(
        convs[i][0].channels,
        (3, 3), activation='relu',
        kernel_regularizer=convs[i][0].reg,
        input_shape=(28, 28, 1)))

    nn.add(keras.layers.Dropout(convs[i][0].dropout))

    nn.add(layers.MaxPooling2D((2, 2)))

    nn.add(layers.Conv2D(
        convs[i][1].channels, (3, 3),
        activation='relu',
        kernel_regularizer=convs[i][0].reg))

    nn.add(keras.layers.Dropout(convs[i][1].dropout))

    nn.add(layers.MaxPooling2D((2, 2)))

    nn.add(layers.Conv2D(
        convs[i][2].channels, (3, 3),
        activation='relu',
        kernel_regularizer=convs[i][0].reg))

    nn.add(keras.layers.Dropout(convs[i][2].dropout))

    nn.add(layers.Flatten())

    nn.add(layers.Dense(64, activation='relu'))

    nn.add(layers.Dense(10, activation='softmax'))

    nn.compile(
        optimizer="rmsprop",
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    hst = nn.fit_generator(
                            train_gen, epochs=10, steps_per_epoch=1500,
                            validation_data=val_gen, validation_steps=250)

    max_val = max(hst.history['val_acc'])
    max_acc_ind = hst.history['val_acc'].index(max_val)

    print(
        "\n\nBest Result: Accuracy - " + str(max_val) +
        " Loss - " + str(hst.history['val_loss'][max_acc_ind]) +
        " in Epoch " + str(max_acc_ind+1))

    print(
        "Channels: " + str(convs[i][0].channels) + " " +
        str(convs[i][1].channels) + " " + str(convs[i][2].channels) +
        " Dropout: " + str(convs[i][0].dropout) + " " +
        str(convs[i][1].dropout) + " " + str(convs[i][2].dropout) + " Reg: " +
        str(convs[i][0].reg_nm) + " " + str(convs[i][1].reg_nm) + " " +
        str(convs[i][2].reg_nm) + "\n\n")

    nn.save('acc_' + str(max_val) + '.h5')
