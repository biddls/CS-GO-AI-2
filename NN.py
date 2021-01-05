import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

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


def trainRL1Sample(model, reward_, did_):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01, decay=0.99)
    for reward, did in zip(reward_, did_):
        did = tf.convert_to_tensor(did, dtype= tf.float32)
        with tf.GradientTape() as t:
            pass

        # gradients = t.gradient(lossComp, model.trainable_variables)
        # optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return model

"""reward = np.array(np.load('reward.npy'))
did = np.load('did.npy')
nnout = np.load('nnout.npy')
print(len(did[0]))
model = modelmake()
trainRL(model, reward, did, nnout)"""