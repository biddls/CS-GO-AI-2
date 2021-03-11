import math
import os
import PIL
import numpy as np
import cv2
import matplotlib.pyplot as plt


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


def getMAE(mae, img1, img2):
    img1 = trimPeramsXY(img1)
    img1 = grayFilter(img1)
    return mae(img1, img2/255)


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
    # np.save("avHLP.npy", temp)
    plt.imshow(temp)
    plt.show()

if __name__ == '__main__':
    imgPrep("HLP")