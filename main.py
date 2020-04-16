import NN
import traceback
import threading
import random
import time
from time import sleep
from getdat import getalldat
from getdat import ctrls
import numpy as np
import pyautogui as pygu
from getdat import screen_grab
import cv2
import keyboard as kbd



class myThread(threading.Thread):
    def __init__(self, threadID, name, ):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.path = 'C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\data.txt'
        self.old = []
        self.reward = 0
        self.new = False

    def run(self):
        print("Starting " + self.name)
        while True:
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
            if len(data) > 0:
                difference = []
                if len(data) > 1:

                    new = data[-1]
                    old = data[-2]

                    if new != self.old:
                        zip_object = zip(new, old)
                        for list1_i, list2_i in zip_object:
                            difference.append(list1_i - list2_i)

                    self.old = new

                elif len(data) == 1:
                    if max(data[0]) == 1:
                        difference = data[0]

                if len(difference) != 0 and sum(difference) < 2:
                    self.reward = difference[0] - difference[1]
                    self.reward = - (self.reward / abs(self.reward))
                    self.new = True

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def action(probs):

    r = random.random()
    index = 0
    while (r >= 0 and index < len(probs)):
        r -= probs[index]
        index += 1
    index -= 1

    probs = np.zeros(9)
    probs[index] = 1
    return probs

def sendinputs(do, shape):
    if shape == (1200, 1600, 3):
        width = shape[1]
        height = shape[0]
        move = do[:-3]
        shoot = do[-3:]
        ctrls.move(outputs[np.argmax(do)])
        r = random.random()
        if r <= move[-1]:
            ctrls.shoot(width * shoot[0], height * shoot[1])
        else:
            ctrls.moveMouse(width * (shoot[0]-0.5), height * (shoot[1]-0.5))

def discount_rewards(r, gamma):
    pointer = 0
    length = 0
    for x in range(len(r)-1, 0, -1):
        if r[x] != 0:
            pointer = r[x]
        else:
            r[x] = pointer * gamma ** (length + 1)
            length += 1

    return r

def restart():
    if screen_grab.grab_screen().shape == (1200, 1600, 3):
        ctrls.cscmd('mp_restartgame 1')
        sleep(0.1)
        kbd.press('esc')
        sleep(1.9)
        ctrls.tap('x')

#key = ['time', 'ct rounds', 't rounds', 'round phase', 'bomb phase', 'players team', 'health', 'flashed', 'smoked', 'burning', 'round kills', 'round kills hs', 'kills', 'assists', 'deaths', 'mvps', 'score']
#outputs = ['a', 'w', 's', 'd', 'aw', 'wd', 'as', 'ad', 'none', '+ or - size for x and y']
outputs = ['a', 'w', 's', 'd', 'aw', 'wd', 'as', 'ad', 'none']#, '+ or - size for x and y']

hyperparams = {'discount factor': 0.98}

def setup():
    open('C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\data.txt', "w+").close()
    getalldat.GSIstart()
    get_data = myThread(1, 'data boi time')

    # Start new Threads
    get_data.start()
    model = NN.modelmake()

    a = np.zeros(24)
    rwd = ['a']
    passed = False
    observations = None
    started = False

    restart()

    while True:
        observation = screen_grab.grab_screen()

        if observation.shape == (1200, 1600, 3):
            s = time.time()
            nnout = model.predict(observation.reshape([-1, 1600, 1200, 3]))[0]
            did = np.append(action(softmax(np.array(nnout[:-3]))), nnout[-3:])
            sendinputs(did, observation.shape)

            if get_data.new == True:
                if type(rwd[0]) == str:
                    rwd = [get_data.reward]
                    didl = [did]
                    nnoutl = [nnout]
                else:
                    rwd.append(get_data.reward)
                    didl.append(did)
                    nnoutl.append(nnout)

                if get_data.reward == -1:
                    ctrls.tap('x')

                if passed == True:
                    passed = False

                    rwd = discount_rewards(rwd, hyperparams['discount factor'])

                    model = NN.train(model, rwd, didl, nnoutl)

                    rwd = ['a']
                    didl = ['a']
                    nnoutl = ['a']

                    restart()

                get_data.new = False

            else:
                if type(rwd[0]) == str:
                    rwd = [0]
                    didl = [did]
                    nnoutl = [nnout]
                else:
                    rwd.append(0)
                    didl.append(did)
                    nnoutl.append(nnout)

            if len(rwd)%2 == 0:
                passed = True

            #print('fps:', 1 / (time.time() - s))


if __name__ == '__main__':
    setup()