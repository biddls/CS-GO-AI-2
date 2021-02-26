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

# key = ['time', 'ct rounds', 't rounds', 'round phase', 'bomb phase', 'players team', 'health', 'flashed', 'smoked', 'burning', 'round kills', 'round kills hs', 'kills', 'assists', 'deaths', 'mvps', 'score']
# outputs = ['a', 'w', 's', 'd', 'aw', 'wd', 'as', 'ad', 'none', '+ or - size for x and y']
outputs = ['a', 'w', 's', 'd', 'aw', 'wd', 'as', 'ad', 'none']  # , '+ or - size for x and y']


class myThread(threading.Thread):  # GSI listener and parser
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.path = 'C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\data.txt'
        self.old = []
        self.reward = 0
        self.new = False
        print("Starting " + self.name)

    def run(self):
        while True:
            # parsing in file
            file = open(self.path)
            data = file.read().split('\n')

            for line in range(len(data)):
                data[line] = data[line].split(', ')
                for index in range(len(data[line])):
                    try:
                        data[line][index] = float(data[line][index])
                    except:
                        pass

            data = np.array(data)[:-1]
            # if there is data
            if len(data) > 0:
                difference = []
                if len(data) > 1:

                    new = data[-1]
                    old = data[-2]
                    # if there is a change
                    if new != self.old:
                        zip_object = zip(new, old)
                        for list1_i, list2_i in zip_object:
                            difference.append(list1_i - list2_i)

                    self.old = new

                # if theres 1 thing in list then thats the new difrence
                elif len(data) == 1:
                    if max(data[0]) == 1:
                        difference = data[0]
                # calulates reward
                if len(difference) != 0 and sum(difference) < 2:
                    self.reward = difference[0] - difference[1]
                    self.reward = (self.reward / abs(self.reward))
                    self.new = True


class Capture(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.image = None
        self.d = d3dshot.create(capture_output="numpy", frame_buffer_size=1)

    def run(self):
        print("Starting " + self.name)
        # counter = 0
        # start = time.time()
        while 1:
            self.image = (cv2.resize(self.d.screenshot(), dsize=(96, 108), interpolation=cv2.INTER_NEAREST) / 255).reshape([-1, 108, 96, 3])

            # counter += 1
            # if (time.time() - start) > 1:
            #     print("FPS: ", counter / (time.time() - start))
            #     counter = 0
            #     start = time.time()

    def getImg(self):
        return self.image


@jit
def softmax(x):  # softmax funciton
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def action(probs):  # chooses what action to make
    r = random.random()
    index = 0
    while (r >= 0 and index < len(probs)):
        r -= probs[index]
        index += 1
    index -= 1

    probs = np.zeros(9)
    probs[index] = 1
    return probs


def actionMouseOnly(actions):  # chooses what action to make
    r = random.random()
    if r <= actions[2]:
        actions[2] = 1
    else:
        actions[2] = 0
    return actions


def sendinputs(do, shape):  # send inputs to cs
    if shape == (1200, 1600, 3):  # makes sure CS is open
        height = shape[0]
        width = shape[1]
        move = do[:-3]
        shoot = do[-3:]
        ctrls.move(outputs[np.argmax(move)])
        r = random.random()
        if r <= shoot[2]:
            ctrls.shoot(width * shoot[0], height * shoot[1])
        else:
            ctrls.moveMouse(width * shoot[0], height * shoot[1])


def sendinputsMouseOnly(do):  # send inputs to cs
        if do[2] == 1:
            ctrls.shoot(1920 * do[0], 1080 * do[1])
        else:
            ctrls.moveMouse(1920 * do[0], 1080 * do[1])

@jit
def discount_rewards(r, gamma):  # backpropergates rewards w discount factor
    pointer = 0
    length = 0
    for x in range(len(r) - 1, 0, -1):
        if r[x] != 0:
            pointer = r[x]
        else:
            r[x] = pointer * gamma ** (length + 1)
            length += 1

    return r


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

                self.did = actionMouseOnly(self.model.predict(observation)[0])  # ignore fn puts part of it though a soft max
                sendinputsMouseOnly(self.did)  # send inputs to cs go

                # if get_data.new == False:  # same as earlier but the reward is 0 as it needs to be back filled
                #     self.RewardList.append(0)
                #     self.DidList.append(self.did)
                #     # self.NNOutputList.append(self.NNOut)
                #     self.images.append(observation)
                #
                # if self.RewardList != None and get_data.new == True:  # if theres a change in the GSI we care about
                #     get_data.new = False  # let it knows its going to process the reward
                #     self.RewardList.append(get_data.reward)
                #     self.DidList.append(self.did)
                #     # self.NNOutputList.append(self.NNOut)
                #     self.images.append(observation)
                #     ctrls.move('none')
                #     self.RewardList = self.discount_rewards(self.RewardList, self.HyperParams['discount factor'])  # back propagates rewards w decay

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
    get_data = myThread(1, 'data boi time')  # starts the code to perpetually look for newly updated txt file
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

# todo: make more efficient by running the AI in a separate thread and it
# todo: just returning its action and all this extra stuff is done else where
# todo: https://github.com/403-Fruit/csctl
