import keyboard as kbd
import pyautogui as pygu
from time import sleep

pygu.FAILSAFE = False


# width = 1920
# height = 1080

def cscmd(cmd):
    try:
        cmd = str(cmd)
        sleep(0.01)
        kbd.press_and_release('=')
        sleep(0.01)
        pygu.hotkey('ctrl', 'a')
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
    for x in keys:
        try:
            kbd.press(x)
        except:
            raise Exception('not a valid key, you tired to pass: {}'.format(x))


def release(keys):
    for x in keys:
        try:
            kbd.release(x)
        except:
            raise Exception('not a valid key, you tired to pass: {}'.format(x))


def shoot(x, y):
    pygu.dragRel(x, y)


def moveMouse(x, y):
    pygu.moveRel(x, y)
