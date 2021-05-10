import tensorflow as tf
from tensorflow.keras import layers, models
import util


def modelMake():
    # https://www.scitepress.org/papers/2018/67520/67520.pdf
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), strides=1, activation='relu', input_shape=(108, 96, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(32, (4, 4), strides=2, activation='relu'))
    model.add(layers.Conv2D(16, (5, 5), strides=2, activation='linear'))
    model.add(layers.Conv2D(8, (6, 6), strides=2, activation='linear'))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='sigmoid'))
    model.add(layers.Dense(70, activation='relu'))
    model.add(layers.Dense(30, activation='linear'))
    model.add(layers.Dense(14, activation='sigmoid'))
    return model


def trainRL1Sample(mdl, inputFrame, resultOfAction, did, norm):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, decay=0.99)  # defines a basic optimiser
    errorBetweenImages = util.maxBrightness(resultOfAction)
    did = tf.reshape(tf.convert_to_tensor(did, tf.float16), (1, 14))


    with tf.GradientTape() as t:
        pred = mdl(inputFrame.reshape(-1, 108, 96, 3))  # output from the model
        lossMethod = tf.keras.losses.BinaryCrossentropy()
        catCross = lossMethod(pred, did)  # find the difference between predicted vs did
        loss = catCross * tf.cast(errorBetweenImages, tf.float16)  # * that by the scalar that is normalised
        loss = norm.logNorm(loss)
        grads = t.gradient(loss, mdl.trainable_variables)  # calculate the gradients
        optimizer.apply_gradients(zip(grads, mdl.trainable_variables))  # apply them to the model

    return mdl, loss


if __name__ == '__main__':
    pass
