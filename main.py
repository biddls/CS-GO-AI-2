import traceback
import threading
import time
from time import sleep
from getdat import getalldat
from getdat import ctrls
import numpy as np
import pyautogui as pygu
from getdat import screen_grab
import cv2

###plan###
# get image in
# load in GSI

"""path = 'C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\images'
    count = 0

    while True:
        count += 1
        np.save(path + '\\' + str(count) + '.npy', np.array([round(time.time() * 100000000), grab_screen()]))"""


class myThread(threading.Thread):
    def __init__(self, threadID, name, ):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.path = 'C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\game data\\data.txt'
        self.new = []
        self.old = []
        self.reward = 0
        self.new = False

    def run(self):
        print("Starting " + self.name)
        while True:
            try:
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
                    try:
                        difference = []
                        if len(data) > 1:

                            self.new = data[-1]
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

                        if len(difference) != 0:
                            self.reward = difference[0] - difference[1]
                            self.reward = self.reward / abs(self.reward)
                            self.new = True
                            new = True
                            reward = 0

                    except Exception:
                        traceback.print_exc()
                        pass
            except:
                pass

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

key = ['time', 'ct rounds', 't rounds', 'round phase', 'bomb phase', 'players team', 'health', 'flashed', 'smoked', 'burning', 'round kills', 'round kills hs', 'kills', 'assists', 'deaths', 'mvps', 'score']
outputs = ['a', 'w', 's', 'd', 'aw', 'wd', 'as', 'ad', 'none', '+ or - size for x and y']


def setup():
    reward = 0
    new = True
    open('C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\game data\\data.txt', "w+").close()
    getalldat.GSIstart()
    get_data = myThread(1, 'data boi time')

    # Start new Threads
    get_data.start()

    while True:
        observation = screen_grab.grab_screen()
        #if observation.shape != (1200, 1600, 3):
        action = np.random.rand(11)

        did = softmax(np.array(action[:-2]))
        if new == True:
            reward = reward
            new = False

            print(action, did, reward)


if __name__ == '__main__':
    setup()