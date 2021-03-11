import tensorflow as tf
import NN
import threading
import random
import time
from getdat import ctrls
import numpy as np
import LDSV
import d3dshot
import cv2
from win32gui import GetWindowText, GetForegroundWindow
from util import Capture

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


class agentBeginnerMouseOnlyTraining():

    def __init__(self, NNmodel):
        self.model = NNmodel
        self.target = np.load('HLP\\avHLP.npy')
        self.did = None

    def start(self):
        # self.restart()#restarts cs go game

        counter = 0
        start = time.time()
        while 1:
            if GetWindowText(GetForegroundWindow()) == "Counter-Strike: Global Offensive":

                observation = capture.getImg()  # grabs new screen data
                # ignore fn puts part of it though a soft max
                self.did = ctrls.actionMouseOnly(self.model.predict(observation)[0])
                self.did = ctrls.sendInputsMouseOnly(self.did)  # send inputs to cs go
                while capture.getImg() is observation:
                    pass

                # train model NN 1 step for each observation
                self.model, error = NN.trainRL1Sample(self.model, observation, capture.getImg(), self.target, self.did)

            counter += 1
            if (time.time() - start) > 1:
                print("FPS: ", counter / (time.time() - start))
                counter = 0
                start = time.time()
                pass


if __name__ == '__main__':
    print("just give it a few seconds to warm up")

    model = LDSV.loadInit('RLCS.h5')
    # Start new Threads
    capture = Capture(1, 'image boi')
    capture.start()

    time.sleep(5)
    agent = agentBeginnerMouseOnlyTraining(model)
    agent.start()
    while 1:
        time.sleep(10)
        LDSV.saveWeight(agent.model, "RLCS.h5")

# todo: https://github.com/403-Fruit/csctl
