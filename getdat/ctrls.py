import random
import tensorflow as tf
import pyautogui
import pydirectinput
from util import sample

pyautogui.FAILSAFE = False

# width = 1920
# height = 1080


def actionMouseOnly(actions):  # chooses what action to make
    r = random.random()
    if r <= actions[2]:
        actions[2] = 1
    else:
        actions[2] = 0
    do = actions

    do[:2] = [(x/2) for x in do[:2]]
    if do[2] == 1:
        shoot(1920 * do[0], 1080 * do[1])
    else:
        moveMouse(1920 * do[0], 1080 * do[1])
    return do


def shoot(x, y):
    moveMouse(x, y)
    # pydirectinput.click()


def moveMouse(x, y):
    pydirectinput.moveRel(int(x), int(y))


def classification(preds):  # chooses what action to make
    # ranges are: 0, 0.01, 0.05, 0.2 and also negative
    x, nnOutX = sample(tf.nn.softmax(preds[:6]))
    y, nnOutY = sample(tf.nn.softmax(preds[7:]))

    moveMouse(1920 * x, 1080 * y)
    return nnOutX + nnOutY


# def move(keys):
#     release('wasd')
#     if keys != 'none':
#         for x in keys:
#             try:
#                 kbd.press(x)
#             except:
#                 raise Exception('not a valid key, you tired to pass: {}'.format(x))
#
#
# def tap(keys):
#     for x in keys:
#         try:
#             kbd.press(x)
#             kbd.release(x)
#         except:
#             raise Exception('not a valid key, you tired to pass: {}'.format(x))
#
#
# def release(keys):
#     for x in keys:
#         try:
#             kbd.release(x)
#         except:
#             raise Exception('not a valid key, you tired to pass: {}'.format(x))