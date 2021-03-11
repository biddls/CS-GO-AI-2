import tensorflow as tf
from tensorflow.keras import layers, models
import util


def modelMake():
    #https://www.scitepress.org/papers/2018/67520/67520.pdf
    model = models.Sequential()
    model.add(layers.Conv2D(16,  (3, 3), strides=1,   activation='relu', input_shape=(108, 96, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(32,  (4, 4),  strides=2,  activation='relu'))
    model.add(layers.Conv2D(16,   (5, 5),  strides=2,  activation='linear'))
    model.add(layers.Conv2D(8,   (6, 6),  strides=2,  activation='linear'))
    model.add(layers.Flatten())
    model.add(layers.Dense(120,  activation='relu'))
    model.add(layers.Dense(70,  activation='relu'))
    model.add(layers.Dense(30,  activation='sigmoid'))
    model.add(layers.Dense(10,   activation='sigmoid'))
    model.add(layers.Dense(3,   activation='tanh'))
    return model


def trainRL1Sample(mdl, inputFrame, resultOfAction, targetFrame, did):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, decay=0.99)
    # resultOfAction = tf.convert_to_tensor(resultOfAction, dtype=tf.float32)
    # targetFrame = tf.convert_to_tensor(targetFrame, dtype=tf.float32)
    error = util.getMAE(tf.keras.losses.MeanAbsoluteError(), resultOfAction, targetFrame)

    with tf.GradientTape() as t:
        pred = mdl(inputFrame.reshape(-1, 108, 96, 3))
        error = tf.keras.losses.mean_squared_error(pred, error * did)
        grads = t.gradient(error, mdl.trainable_variables)
        optimizer.apply_gradients(zip(grads, mdl.trainable_variables))

    return mdl, error

if __name__=='__main__':
    pass
