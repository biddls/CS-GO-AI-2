import threading
import time
from time import sleep
from getdat import getalldat
from getdat import ctrls
import numpy as np

###plan###
# get image in
# load in GSI

"""path = 'C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\images'
    count = 0

    while True:
        count += 1
        np.save(path + '\\' + str(count) + '.npy', np.array([round(time.time() * 100000000), grab_screen()]))"""


class myThread(threading.Thread):
    def __init__(self, threadID, name, ):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.path = 'C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\game data\\data.txt'
        self.file = None

    def run(self):
        print("Starting " + self.name)
        while True:
            try:
                self.file = self.parse(open(self.path))
            except:
                pass

    def parse(self, file):
        data = file.read().split('\n')

        for line in range(len(data)):
            data[line] = data[line].split(', ')
            for index in range(len(data[line])):
                try:
                    data[line][index] = float(data[line][index])
                except:
                    pass

        data = np.array(data)
        if len(data) > 1:
            return data[:-1]
        else:
            return None

class GSI(threading.Thread):
    def __init__(self, threadID, name, ):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.previous = None
        self.current = None
        self.score = 0
        self.deaths = 0

    def run(self):
        print("Starting " + self.name)

        while True:
            sleep(1)
            if self.previous is None:
                self.previous = self.current
            elif self.previous[-1][0] != self.current[-1][0]:
                self.previous = self.reduce()

    def reduce(self):
        print('hi')
        new = self.current
        old = self.previous[0]
        print(new,'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        for index, x in enumerate(new):
            print(index, x)
            if x == old:
                new = new[:-index]

        print(new, '###################################')

        """swapped = 1
        while swapped > 0:
            swapped = 0
            for n1, n2 in zip(reversed(array[1:]), reversed(array[:-1])):
                if n1 == n2:
                    swapped += 1
                    #remove element should result in array being only filled w uniqe entrys where in each entry something that is being measured gets changed"""



key = ['time', 'ct rounds', 't rounds', 'round phase', 'bomb phase', 'players team', 'health', 'flashed', 'smoked', 'burning', 'round kills', 'round kills hs', 'kills', 'assists', 'deaths', 'mvps', 'score']


def setup():
    open('C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS GO AI 2\\data\\game data\\data.txt', "w+").close()
    getalldat.GSIstart()
    get_data = myThread(1, 'get data')
    csinfo = GSI(2, 'GSI')

    # Start new Threads
    get_data.start()
    csinfo.start()

    while True:
        sleep(3)
        csinfo.current = get_data.file

if __name__ == '__main__':
    setup()