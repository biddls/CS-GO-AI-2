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

def trainRL(model, reward_, did_, nnout_):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01, decay=0.99)

    for reward, did, nnout in zip(reward_, did_, nnout_):
        with tf.GradientTape() as t:
            catCross = tf.losses.categorical_crossentropy(tf.convert_to_tensor(did, dtype=tf.float32),
                                                          tf.convert_to_tensor(nnout, dtype=tf.float32))
            lossComp = tf.math.multiply(reward, catCross)

        gradients = t.gradient(lossComp, model.trainable_variables)
        print(len(gradients))
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


"""reward = np.array(np.load('reward.npy'))
did = np.load('did.npy')
nnout = np.load('nnout.npy')
print(len(did[0]))
model = modelmake()
trainRL(model, reward, did, nnout)"""