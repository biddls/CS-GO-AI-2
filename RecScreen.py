import math
import threading
import time
import d3dshot
import cv2
from win32gui import GetWindowText, GetForegroundWindow
import numpy as np
import util


class Capture(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.image = None
        self.d = d3dshot.create(capture_output="numpy", frame_buffer_size=1)

    def run(self):
        print("Starting " + self.name)
        while 1:
            self.image = (cv2.resize(self.d.screenshot(), dsize=(96, 108), interpolation=cv2.INTER_NEAREST) / 255)\
                .reshape([-1, 108, 96, 3])

    def getImg(self):
        return self.image

def start():
    capture = Capture(1, 'image boi')
    capture.start()
    time.sleep(1)
    print("Ready")

    time.sleep(2)

    for x in range(50):
        while GetWindowText(GetForegroundWindow()) != "Counter-Strike: Global Offensive":
            pass

        time.sleep(1)
        img = capture.getImg()

        np.save("test data set/{}.npy".format(x), img)
        print(x, img.shape)

    print("DONE")

if __name__ == '__main__':
    start()
