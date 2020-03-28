from screen_grab import grab_screen
import cv2
import numpy as np
import os
from useless.getkeys import key_check
import time

#collect dat at a certian fps and save it
file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []

def main(fps):

    if fps == None:
        fps = 0
    else:
        fps = 1/fps

    for i in range(4,0,-1):
        print(i)
        time.sleep(1)

    paused = False
    ind = time.time()

    while True:

        if not paused and time.time() >= (ind + fps):
            print('in')
            ind += 1
            # 800x600 windowed mode
            screen = grab_screen(region=(1, 26, 800 ,625))
            cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            training_data.append([screen])

            if len(training_data) % 100 == 0:
                print(len(training_data))
                np.save(file_name, training_data)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
fps = 0
try:
    fps = float(input('fps: '))
except:
    pass
main(fps)