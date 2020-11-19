import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
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

# key = ['time', 'ct rounds', 't rounds', 'round phase', 'bomb phase', 'players team', 'health', 'flashed', 'smoked', 'burning', 'round kills', 'round kills hs', 'kills', 'assists', 'deaths', 'mvps', 'score']
# outputs = ['a', 'w', 's', 'd', 'aw', 'wd', 'as', 'ad', 'none', '+ or - size for x and y']
outputs = ['a', 'w', 's', 'd', 'aw', 'wd', 'as', 'ad', 'none']  # , '+ or - size for x and y']

class myThread(threading.Thread):#GSI listener and parser
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
            #parsing in file
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
            #if there is data
            if len(data) > 0:
                difference = []
                if len(data) > 1:

                    new = data[-1]
                    old = data[-2]
                    #if there is a change
                    if new != self.old:
                        zip_object = zip(new, old)
                        for list1_i, list2_i in zip_object:
                            difference.append(list1_i - list2_i)

                    self.old = new

                #if theres 1 thing in list then thats the new difrence
                elif len(data) == 1:
                    if max(data[0]) == 1:
                        difference = data[0]
                #calulates reward
                if len(difference) != 0 and sum(difference) < 2:
                    self.reward = difference[0] - difference[1]
                    self.reward = (self.reward / abs(self.reward))
                    self.new = True
class agent():

    def __init__(self):
        self.hyperparams = {'discount factor': 0.98}  # dicount factor
        self.images = None
        self.nnout = None
        self.nnoutl = None
        self.did = None
        self.didl = None
        self.rwdl = None


    def softmax(self, x):#softmax funciton
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def action(self, probs):#chooses what action to make
        r = random.random()
        index = 0
        while (r >= 0 and index < len(probs)):
            r -= probs[index]
            index += 1
        index -= 1

        probs = np.zeros(9)
        probs[index] = 1
        return probs

    def sendinputs(self, do, shape):#send inputs to cs
        if shape == (1200, 1600, 3):#makes sure CS is open
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

    def discount_rewards(self, r, gamma):#backpropergates rewards w discount factor
        pointer = 0
        length = 0
        for x in range(len(r)-1, 0, -1):
            if r[x] != 0:
                pointer = r[x]
            else:
                r[x] = pointer * gamma ** (length + 1)
                length += 1

        return r

    def restart(self):#restarts the round
        #exec RL1
        if np.shape(d.screenshot()) == (1200, 1600, 3):
            sleep(1)
            kbd.press('h')
            sleep(1.5)
            ctrls.tap('x')

    def start(self, model):

        while np.shape(d.screenshot()) != (1200, 1600, 3):#if in game
            pass

        #self.restart()#restarts cs go game

        while 1:
            start = time.time()
            observation = d.screenshot()#grabs new screen data

            if observation.shape == (1200, 1600, 3):#if its of the game
                self.nnout = model.predict(observation.reshape([-1, 1600, 1200, 3]))[0]#gets NN ouputad
                self.did = np.append(self.action(self.softmax(np.array(self.nnout[:-3]))), self.nnout[-3:])#puts part of it though a soft max
                #self.sendinputs(self.did, observation.shape)#send inputs to cs go

                if get_data.new == False:#same as earlier but the reward is 0 as it needs to be back filled
                    if self.rwdl == None:
                        self.rwdl = [0]
                        self.didl = [self.did]
                        self.nnoutl = [self.nnout]
                        self.images = [observation]
                    else:
                        self.rwdl.append(0)
                        self.didl.append(self.did)
                        self.nnoutl.append(self.nnout)
                        self.images.append(observation)

                if self.rwdl != None and get_data.new == True:#if theres a cahnge in the GSI we care about
                    get_data.new = False#let it knows its going to process the reward
                    self.rwdl.append(get_data.reward)
                    self.didl.append(self.did)
                    self.nnoutl.append(self.nnout)
                    self.images.append(observation)
                    ctrls.move('none')
                    self.rwdl = self.discount_rewards(self.rwdl, self.hyperparams['discount factor'])#back prpergates rewards w decay
                    return NN.trainRL(model, self.rwdl, self.didl, self.nnoutl)#trains NN 1 step for each observation
            print(round(1/(time.time() - start),1)," FPS")

if __name__ == '__main__':
    open('C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\data.txt', "w+").close()#resets text file
    getalldat.GSIstart()#starts GSI server
    get_data = myThread(1, 'data boi time')#starts the code to perpertualy look for newly updated txt file

    # Start new Threads
    get_data.start()
    model = LDSV.loadWeights("RLCS.h5")

    d = d3dshot.create(capture_output="numpy", frame_buffer_size=1)

    while 1:
        open('C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\data.txt',"w+").close()#resets text file
        model, stats = agent().start(model)
        LDSV.saveWeights(model, "RLCS.h5")

#todo: make more effiecient by running the AI in a seperate thread and it
# just returning its action and all this extra stuff is done else wher