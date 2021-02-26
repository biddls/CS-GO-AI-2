import os
from numba import jit
import tensorflow as tf

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

import timeit
import NN
import threading
import random
from time import sleep
import time
from getdat import getalldat
from getdat import ctrls
import numpy as np
import keyboard as kbd
import LDSV
import d3dshot
import cv2
from win32gui import GetWindowText, GetForegroundWindow
from matplotlib import pyplot as plt
import util

def trainRL1Sample(mdl, inputFrame, resultOfAction, targetFrame, index):
    output = mdl.predict(inputFrame.reshape(-1, 108, 96, 3))
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01, decay=0.99)
    # format outcome frame correctly
    with tf.GradientTape() as t:
        # gets MAE between the outcome img and
        mae = tf.keras.losses.MeanAbsoluteError()
        error = util.getMAE(mae, resultOfAction, targetFrame)
        print("so given input of:", index, "\tits action on the env returns an error of:", error.numpy().round(5), "\tits action is:", output)
        print(error)
        # update weights to minimise the error
        grads = t.gradient(error, mdl.trainable_variables)
        print(grads)  # its empty
        optimizer.apply_gradients(zip(grads, mdl.trainable_variables))

    return mdl


if __name__ == "__main__":
    if os.path.exists("RLCS.h5"):
        model = LDSV.loadWeights("RLCS.h5")
    else:
        model = NN.modelMake()
        LDSV.saveWeight(model, "RLCS.h5")

    # print(model.summary())

    os.chdir("HLP")
    target = np.load("testHLP.npy")  # what i want the AI to train to
    os.chdir("..")
    os.chdir("test data set")
    dirList = os.listdir()
    dirList.sort(key=lambda x: int(x.split(".npy")[0]))  # sorts the simulated data set

    for index, value in enumerate(dirList[:-1]):  # for each itteration in the data set do this
        modelInput = np.load(dirList[index])  # gets observation from "ENV"
        actionOutcome = np.load(dirList[index + 1])  # gets the outcome of the AIs action on the ENV

        model = trainRL1Sample(model, modelInput, actionOutcome, target, value)  # 1 training step on the data
