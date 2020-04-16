import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


def modelmake():
    model = models.Sequential()
    model.add(layers.Conv2D(200,  (7, 7), strides= 1,   activation='relu', input_shape=(1600, 1200, 3)))
    model.add(layers.MaxPooling2D (pool_size= (2, 2),   strides= 2))
    model.add(layers.Conv2D(100,  (5, 5),  strides= 2,  activation='relu'))
    model.add(layers.Conv2D(80,   (8, 5),  strides= 3,  activation='linear'))
    model.add(layers.Conv2D(60,   (5, 2),   strides= 3, activation='relu'))
    model.add(layers.Conv2D(64,   (1, 1),   strides= 1, activation='linear'))
    model.add(layers.Conv2D(128,  (3, 3),   strides= 2, activation='relu'))
    model.add(layers.Conv2D(128,  (3, 2),   strides= 1, activation='relu'))
    model.add(layers.Conv2D(128,  (3, 3),   strides= 1, activation='linear'))
    model.add(layers.Conv2D(256,  (3, 3),   strides= 1, activation='relu'))
    model.add(layers.Conv2D(256,  (3, 3),   strides= 1, activation='linear'))
    model.add(layers.Conv2D(512,  (3, 3),   strides= 2, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(5000, activation= 'relu'))
    model.add(layers.Dense(2400, activation= 'relu'))
    model.add(layers.Dense(1000, activation= 'relu'))
    model.add(layers.Dense(500,  activation= 'relu'))
    model.add(layers.Dense(256,  activation= 'relu'))
    model.add(layers.Dense(128,  activation= 'sigmoid'))
    model.add(layers.Dense(90,   activation= 'sigmoid'))
    model.add(layers.Dense(12,   activation= 'sigmoid'))

    return model

def grad(model, x, y, z):
    with tf.GradientTape() as tape:
        loss_value = losscalc(x, y, z)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def losscalc(x, y, z):
    return tf.reduce_sum(x * tf.losses.categorical_crossentropy(y, z))


def train(model, reward, did, nnout):

    optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.001, decay = 0.99)
    for x, y, z in zip(reward, did, nnout):
        """loss_value, grads = grad(model, x, y, z)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))"""
        loss = lambda: losscalc(x, y, z)
        var_list_fn = lambda: model.trainable_weights
        optimizer.minimize(loss, var_list_fn)

    return model

reward = np.array(np.load(r'reward.npy'))
did = np.load(r'did.npy')
nnout = np.load(r'nnout.npy')

model = modelmake()
train(model, reward, did, nnout)