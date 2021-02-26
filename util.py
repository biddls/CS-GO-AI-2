import math
import numpy as np

def trimPerams(img, off):
    if len(img.shape) == 4:
        shape = img.shape[1:]
        img = img[0]
    elif len(img.shape) == 3:
        pass
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
    return mae(img1, img2)
