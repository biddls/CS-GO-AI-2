import tensorflow as tf
import os
import NN


def saveWeight(model, adress):
    model.save(adress)


def loadWeights(adress):
    model = tf.keras.models.load_model(adress)
    return model


def loadInit(address):
    if os.path.exists(address):
        return loadWeights(address)
    else:
        model = NN.modelMake()
        saveWeight(model, address)
        return model
