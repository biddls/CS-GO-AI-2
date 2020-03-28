import numpy as np
import os
import cv2
#depericated
labels = {'T': 0,
         'CT':1}

file_name = 'training_data.npy'

training_data = list(np.load(file_name))

#turns into images to lable
#for index, img in enumerate(training_data):
#    cv2.imwrite('images\{}.jpg'.format(index), cv2.cvtColor(img[0], cv2.COLOR_GRAY2RGB))

os.chdir("dat saved/vott-csv-export")
file = glob.glob("*.csv")

temp = training_data
data = []

#split images into 100X100 squares

for x in training_data:
    for y in x:
        divsize = 100
        rowcount = y.shape[0]
        colcount = y.shape[1]

        rows = np.array(np.split(y, rowcount/divsize, axis=0))
        x = np.empty([int(rowcount / divsize), int(colcount / divsize)], dtype=object)

        for index, row in enumerate(rows):
            row = np.array(np.split(row, colcount/divsize, axis=1))

            for index2, col in enumerate(row):
                x[index, index2] = col  #cant get images to be put int a 6X8 arryay w each elemetn holding a 100X100 image FUCK
        data.append(x)

pure_images = temp

data = np.array(data)

training_data = []

#does data aranging
for img in data:
    temp = np.empty([int(img.shape[0]-1), int(img.shape[1])-1], dtype=object)
    for y in range(0,8-1):
        for x in range(0,6-1):
            boxx = np.append(arr = img[x,y], values = img[x,y+1], axis = 1)
            boxy = np.append(arr = img[x+1,y], values = img[x+1,y+1], axis = 1)
            boxy = np.append(arr = boxx, values = boxy, axis = 0)
            temp[x,y] = np.array(boxy)
    img = temp

#runs at 1.5 miliseconds