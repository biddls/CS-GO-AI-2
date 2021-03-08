import math
import os
import PIL
import numpy as np
import cv2
import matplotlib.pyplot as plt


def trimPerams(img, off):
    if len(img.shape) == 4:
        shape = img.shape[1:]
        img = img[0]
    elif len(img.shape) == 3:
        shape = img.shape
    else:
        return None

    y1 = math.floor(shape[0] * off)
    y2 = math.floor(shape[0] * (1 - off))
    x1 = math.floor(shape[1] * off)
    x2 = math.floor(shape[1] * (1 - off))

    dims = [y1, y2, x1, x2]
    return img[dims[0]:dims[1], dims[2]:dims[3]]


def getMAE(mae, img1, img2):
    img1 = trimPerams(img1, 0.2)
    return mae(img1, img2/255)


def imgPrep():
    os.chdir("HLP")
    files = [x for x in os.listdir() if x[-4:] == '.png']
    temp = None
    for file in files:
        frame = PIL.Image.open(file)
        frame = frame.convert("RGB")
        frame = np.asarray(frame)
        frame = cv2.resize(frame, dsize=(96, 108), interpolation=cv2.INTER_LANCZOS4)
        frame = trimPerams(frame, 0.2)
        if temp is None:
            temp = frame / len(files)
        else:
            temp += frame / len(files)

    temp = temp.astype(int)
    np.save("avHLP.npy", temp)
    plt.imshow(temp)
    plt.show()
