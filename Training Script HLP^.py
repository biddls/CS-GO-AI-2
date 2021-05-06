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
from util import Capture
from util import WatchGSI
from util import softmax
from util import discount_rewards


class agentBeginnerMouseOnlyTraining():

    def __init__(self, NNmodel):
        self.HyperParams = {'discount factor': 0.98}  # discount factor
        self.images = []
        # self.NNOut = []
        # self.NNOutputList = []
        self.did = []
        self.DidList = []
        self.RewardList = []
        self.model = NNmodel

    def start(self):
        # self.restart()#restarts cs go game

        counter = 0
        start = time.time()
        while 1:
            if GetWindowText(GetForegroundWindow()) == "Counter-Strike: Global Offensive":

                observation = capture.getImg()  # grabs new screen data

                # # if its of the game
                # self.NNOut = self.model.predict(observation.reshape([-1, 108, 96, 3]))[0]  # gets NN outputted

                self.did = ctrls.actionMouseOnly(self.model.predict(observation)[0])  # ignore fn puts part of it though a soft max
                ctrls.sendInputsMouseOnly(self.did)  # send inputs to cs go

                if get_data.new == False:  # same as earlier but the reward is 0 as it needs to be back filled
                    self.RewardList.append(0)
                    self.DidList.append(self.did)
                    # self.NNOutputList.append(self.NNOut)
                    self.images.append(observation)

                if self.RewardList != None and get_data.new == True:  # if theres a change in the GSI we care about
                    get_data.new = False  # let it knows its going to process the reward
                    self.RewardList.append(get_data.reward)
                    self.DidList.append(self.did)
                    # self.NNOutputList.append(self.NNOut)
                    self.images.append(observation)
                    ctrls.move('none')
                    self.RewardList = self.discount_rewards(self.RewardList, self.HyperParams['discount factor'])  # back propagates rewards w decay

                while temp := capture.getImg().all() == observation.all():
                    pass

                # self.model = NN.trainRL1Sample(self.model, temp, self.did)  # trains NN 1 step for each observation

                counter += 1
                if (time.time() - start) > 1:
                    print("FPS: ", counter / (time.time() - start))
                    counter = 0
                    start = time.time()
                    pass


if __name__ == '__main__':
    print("just give it a few seconds to warm up")
    time.sleep(5)
    open('C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\data.txt',
         "w+").close()  # resets text file
    getalldat.GSIstart()  # starts GSI server

    # Start new Threads
    get_data = WatchGSI(1, 'data boi time')  # starts the code to perpetually look for newly updated txt file
    get_data.start()

    capture = Capture(1, 'image boi')
    capture.start()

    if os.path.exists("RLCS.h5"):
        model = LDSV.loadWeights("RLCS.h5")
    else:
        model = NN.modelMake()
        LDSV.saveWeight(model, "RLCS.h5")

    print(model.summary())

    while 1:
        open('C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\data.txt',
             "w+").close()  # resets text file
        agentBeginnerMouseOnlyTraining(model).start()
        LDSV.saveWeight(model, "RLCS.h5")
