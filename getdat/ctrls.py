import random

import keyboard as kbd
import pyautogui
import pydirectinput
import numpy as np
from time import sleep

pyautogui.FAILSAFE = False

# width = 1920
# height = 1080

def actionMouseOnly(actions):  # chooses what action to make
    r = random.random()
    if r <= actions[2]:
        actions[2] = 1
    else:
        actions[2] = 0
    return actions


def sendInputsMouseOnly(do):  # send inputs to cs
    do[:2] = [(x/2) for x in do[:2]]
    if do[2] == 1:
        shoot(1920 * do[0], 1080 * do[1])
    else:
        moveMouse(1920 * do[0], 1080 * do[1])
    return do


def cscmd(cmd):
    try:
        cmd = str(cmd)
        sleep(0.01)
        kbd.press_and_release('=')
        sleep(0.01)
        pyautogui.hotkey('ctrl', 'a')
        sleep(0.1)
        kbd.write(cmd)
        sleep(0.01)
        kbd.press('enter')
        kbd.press('esc')
    except:
        print('could not use input passed')
        try:
            print('input was', str(cmd))
        except:
            print('input was', cmd)


def move(keys):
    release('wasd')
    if keys != 'none':
        for x in keys:
            try:
                kbd.press(x)
            except:
                raise Exception('not a valid key, you tired to pass: {}'.format(x))


def tap(keys):
    for x in keys:
        try:
            kbd.press(x)
            kbd.release(x)
        except:
            raise Exception('not a valid key, you tired to pass: {}'.format(x))


def release(keys):
    for x in keys:
        try:
            kbd.release(x)
        except:
            raise Exception('not a valid key, you tired to pass: {}'.format(x))


def shoot(x, y):
    moveMouse(x, y)
    # pydirectinput.click()


def moveMouse(x, y):
    pydirectinput.moveRel(int(x), int(y))


def restart(img):  # restarts the round
    # exec RL1
    if np.shape(img) == (108, 144, 3):
        sleep(1)
        kbd.press('h')
        sleep(1.5)
        tap('x')
