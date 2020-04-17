import NN
import matplotlib.pyplot as plt
import threading
import random
import time
from time import sleep
from getdat import getalldat
from getdat import ctrls
import numpy as np
from getdat import screen_grab
import keyboard as kbd
from statistics import mean


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

def softmax(x):#softmax funciton
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def action(probs):#chooses what action to make
    r = random.random()
    index = 0
    while (r >= 0 and index < len(probs)):
        r -= probs[index]
        index += 1
    index -= 1

    probs = np.zeros(9)
    probs[index] = 1
    return probs

def sendinputs(do, shape):#send inputs to cs
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

def discount_rewards(r, gamma):#backpropergates rewards w discount factor
    pointer = 0
    length = 0
    for x in range(len(r)-1, 0, -1):
        if r[x] != 0:
            pointer = r[x]
        else:
            r[x] = pointer * gamma ** (length + 1)
            length += 1

    return r

def restart():#resrtats the round
    if screen_grab.grab_screen().shape == (1200, 1600, 3):
        kbd.press('h')
        sleep(1.5)
        ctrls.tap('x')

#key = ['time', 'ct rounds', 't rounds', 'round phase', 'bomb phase', 'players team', 'health', 'flashed', 'smoked', 'burning', 'round kills', 'round kills hs', 'kills', 'assists', 'deaths', 'mvps', 'score']
#outputs = ['a', 'w', 's', 'd', 'aw', 'wd', 'as', 'ad', 'none', '+ or - size for x and y']
outputs = ['a', 'w', 's', 'd', 'aw', 'wd', 'as', 'ad', 'none']#, '+ or - size for x and y']

hyperparams = {'discount factor': 0.98}#dicount factor

def setup():
    open('C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\data.txt', "w+").close()#resets text file
    getalldat.GSIstart()#stats GSI server
    get_data = myThread(1, 'data boi time')#stats the code to perpertualy look for newly updated txt file

    # Start new Threads
    get_data.start()
    model = NN.modelmake()

    rwd = ['a']#reward list
    passed = True #past threshold to begin waiting for a kill or death

    runs = []

    if screen_grab.grab_screen().shape == (1200, 1600, 3):#if in game
        restart()#restartes cs go game

    for x in range(1000):#100 loops though
        observation = screen_grab.grab_screen()#grabs new screen data

        if observation.shape == (1200, 1600, 3):#if its of the game
            nnout = model.predict(observation.reshape([-1, 1600, 1200, 3]))[0]#gets NN ouput
            did = np.append(action(softmax(np.array(nnout[:-3]))), nnout[-3:])#puts part of it though a soft max
            sendinputs(did, observation.shape)#send inputs to cs go

            if get_data.new == True:#if theres a new record from the GSI
                get_data.new = False #let it knows its processed the reward

                if type(rwd[0]) == str: #initlaises tracking vars
                    rwd = [get_data.reward]
                    didl = [did]
                    nnoutl = [nnout]

                else:
                    rwd.append(get_data.reward)
                    didl.append(did)
                    nnoutl.append(nnout)

                if get_data.reward == -1:
                    ctrls.tap('x')

                if passed == True: #if the number of itterations has passed a batch size
                    passed = False
                    if rwd != [1.0]:
                        print(rwd)
                        ctrls.move('none')
                        rwd = discount_rewards(rwd, hyperparams['discount factor'])#back prpergates rewards w decay
                        model, losses = NN.train(model, rwd, didl, nnoutl)#trains NN 1 step for each observation

                        runs.append(mean(losses))

                        #resets vars
                        rwd = ['a']
                        didl = ['a']
                        nnoutl = ['a']

                        ctrls.tap('h')
                        sleep(3)
                        ctrls.tap('x')


            else:#same as earlier but the reward is 0 as it needs to be back filled
                if type(rwd[0]) == str:
                    rwd = [0]
                    didl = [did]
                    nnoutl = [nnout]
                else:
                    rwd.append(0)
                    didl.append(did)
                    nnoutl.append(nnout)

            if len(rwd)%400 == 0:#trigger for batch size
                passed = True
            print(x)
            #print('fps:', 1/(time.time()-s))#prints fps


    plt.plot(runs)
    plt.show()



if __name__ == '__main__':
    setup()