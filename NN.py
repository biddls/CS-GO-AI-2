import math
import time

import PIL
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

def paddPerams(shape, off):
    y1 = math.floor(shape[0] * off)
    y2 = math.floor(shape[0] * (1-off))
    x1 = math.floor(shape[1] * off)
    x2 = math.floor(shape[1] * (1-off))

    return [y1, y2, x1, x2]

def saveNp():
    images = loadIm("training.npy")
    temp = None
    for x in images:
        if temp is None:
            temp = x/6
        else:
            temp += x/6
    temp = temp.astype(int)
    np.save("avHLP.npy", temp)

def imgPrep():
    os.chdir("HLP")
    files = [x for x in os.listdir() if x[-4:] == '.png']
    temp = None
    for file in files:
        frame = PIL.Image.open(file)
        frame = frame.convert("RGB")
        frame = np.asarray(frame)
        frame = cv2.resize(frame, dsize=(96, 108), interpolation=cv2.INTER_LANCZOS4)
        dims = paddPerams(frame.shape, 0.2)
        frame = frame[dims[0]:dims[1]][dims[2]:dims[3]]
        if temp is None:
            temp = frame
        else:
            temp = np.concatenate((temp, frame), axis= 1)

    np.save("training.npy", temp)

def loadIm(path):
    os.chdir("HLP")
    return np.split(np.load(path), 6, axis= 1)


def modelmake():
    #https://www.scitepress.org/papers/2018/67520/67520.pdf
    model = models.Sequential()
    model.add(layers.Conv2D(16,  (3, 3), strides= 1,   activation='relu', input_shape=(108, 144, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(32,  (4, 4),  strides= 2,  activation='relu'))
    model.add(layers.Conv2D(16,   (5, 5),  strides= 2,  activation='linear'))
    model.add(layers.Conv2D(8,   (6, 6),  strides= 2,  activation='linear'))
    model.add(layers.Flatten())
    model.add(layers.Dense(120,  activation= 'relu'))
    model.add(layers.Dense(70,  activation= 'relu'))
    model.add(layers.Dense(30,  activation= 'sigmoid'))
    model.add(layers.Dense(10,   activation= 'sigmoid'))
    model.add(layers.Dense(3,   activation= 'sigmoid'))
    return model


def trainRL(model, reward_, did_, nnout_, images_):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01, decay=0.99)
    for reward, did, image in zip(reward_, did_, images_):
        nnout = model.predict(image.reshape([-1, 1600, 1200, 3]))[0]  # gets NN outputed
        did = tf.convert_to_tensor(did, dtype= tf.float32)
        with tf.GradientTape() as t:
            catCross = tf.losses.categorical_crossentropy(tf.convert_to_tensor(did, dtype=tf.float32),
                                                          tf.convert_to_tensor(nnout, dtype=tf.float32))
            lossComp = tf.math.multiply(reward, catCross)

        gradients = t.gradient(lossComp, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def trainRL1Sample(model, actionOutcome, did):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01, decay=0.99)
    # np load in target frames
    for image in images:
        with tf.GradientTape() as t:
            error = tf.keras.losses.categorical_crossentropy(image, actionOutcome)
            #   categorical cross entropy on the 2 images to get error
            #   update weights to minimise the error

        grads = t.gradient(error, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return model

# todo: add shit here to load in all the pictures as training examples
if __name__=='__main__':
    os.chdir("HLP")
    plt.imshow(np.load("avHLP.npy"))
    plt.show()
