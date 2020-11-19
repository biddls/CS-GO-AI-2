import tensorflow as tf

def saveWeight(model, adress):
    model.save(adress)

def loadWeights(adress):
    model = tf.keras.models.load_model(adress)
    return model