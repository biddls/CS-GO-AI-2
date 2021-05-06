import math
import os
import threading
import PIL
import d3dshot
import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import random


class WatchGSI(threading.Thread):  # GSI listener and parser
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
            self.image = (cv2.resize(self.d.screenshot(), dsize=(96, 108),
                                     interpolation=cv2.INTER_NEAREST) / 255).reshape([-1, 108, 96, 3])
            # counter += 1
            # if (time.time() - start) > 1:
            #     print("FPS: ", counter / (time.time() - start))
            #     counter = 0
            #     start = time.time()

    def getImg(self):
        return self.image


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


def softmax(x):  # softmax funciton
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def trimPeramsXY(img, x=0.41, y1=0.46, y2=0.35):
    if len(img.shape) == 4:
        shape = img.shape[1:]
        img = img[0]
    elif len(img.shape) == 3:
        shape = img.shape
    else:
        return None

    y1 = math.floor(shape[0] * y1)
    y2 = math.floor(shape[0] * (1 - y2))
    x1 = math.floor(shape[1] * x)
    x2 = math.floor(shape[1] * (1 - x))

    img = img[y1:y2, x1:x2]

    return img


def grayFilter(img, minNumb=80, maxNumb=110):
    for indexx, x in enumerate(img):
        for indexy, y in enumerate(x):
            if minNumb <= y[0] <= maxNumb and minNumb <= y[1] <= maxNumb and minNumb <= y[2] <= maxNumb:
                img[indexx][indexy] = [0, 0, 0]

    return img


def getMAE(mae, img1, img2):  # returns the av difference between 2 images
    img1 = trimPeramsXY(img1)
    img1 = grayFilter(img1)
    return mae(img1, img2/255)

def maxBrightness(img1):
    img1 = trimPeramsXY(img1)
    img1 = grayFilter(img1)
    img1 = np.average(img1)
    return img1


def imgPrep(path):
    os.chdir(path)
    files = [x for x in os.listdir() if x[-4:] == '.png']
    temp = None
    for file in files:
        frame = PIL.Image.open(file)
        frame = frame.convert("RGB")
        frame = np.asarray(frame)
        frame = cv2.resize(frame, dsize=(96, 108), interpolation=cv2.INTER_LANCZOS4)
        frame = trimPeramsXY(frame)
        if temp is None:
            temp = frame / len(files)
        else:
            temp += frame / len(files)

    temp = grayFilter(temp)

    temp = temp.astype(int)
    np.save("avHLP.npy", temp)
    plt.imshow(temp)
    plt.show()


def sample(dist):
    a = random()
    total = 0
    for index, x in enumerate(dist):
        if total + x < a:
            total += x
        else:
            temp = [0 for x in range(7)]
            temp[index] = 1
            return [-0.2, -0.05, -0.01, 0, 0.01, 0.05, 0.2][index], temp


if __name__ == '__main__':
    # imgPrep("HLP")
    img = np.load('HLP\\avHLP.npy')
    plt.imshow(img)
    plt.show()
