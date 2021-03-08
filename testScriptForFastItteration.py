import os
import NN
import numpy as np
import LDSV
import util
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def step(mdl, inputFrame, resultOfAction, targetFrame):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01, decay=0.99)
    resultOfAction = tf.convert_to_tensor(resultOfAction, dtype=tf.float32)
    targetFrame = tf.convert_to_tensor(targetFrame, dtype=tf.float32)
    err = util.getMAE(tf.keras.losses.MeanAbsoluteError(), resultOfAction, targetFrame)

    with tf.GradientTape() as t:
        pred = mdl(inputFrame.reshape(-1, 108, 96, 3))
        err = tf.keras.losses.categorical_crossentropy(pred, - err * pred)
        grads = t.gradient(err, mdl.trainable_variables)  # its empty
        optimizer.apply_gradients(zip(grads, mdl.trainable_variables))

    return mdl, err


if __name__ == "__main__":
    if os.path.exists("RLCS.h5"):
        model = LDSV.loadWeights("RLCS.h5")
    else:
        model = NN.modelMake()
        LDSV.saveWeight(model, "RLCS.h5")
    target = np.load("HLP\\avHLP.npy")
    os.chdir("test data set")
    dirList = os.listdir()
    dirList.sort(key=lambda x: int(x.split(".npy")[0]))  # sorts the simulated data set

    arr = []
    for x in tqdm(range(100)):
        temp = []
        for index, value in enumerate(dirList[:-1]):  # for each itteration in the data set do this
            modelInput = np.load(dirList[index])  # gets observation from "ENV"
            actionOutcome = np.load(dirList[index + 1])  # gets the outcome of the AIs action on the ENV

            model, error = step(model, modelInput, actionOutcome, target)  # 1 training step on the data
            temp.append(error)

        arr.append(sum(temp) / len(temp))

    print(arr)
